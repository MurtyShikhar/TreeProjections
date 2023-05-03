from datasets import load_dataset
import os
from data_utils import build_datasets, build_datasets_semparse, build_datasets_pcfg
from transformer_helpers import create_model
import numpy as np
import argparse
import torch
from tree_projection_src import (
    get_word_vecs_from_subwords,
    get_chart_parse,
    get_pre_tokenized_info,
    random_parse,
)
from tqdm import tqdm
import pickle
import random
import collate


from scipy.spatial import distance
from graph_node import Graph


def get_parse_fn(args):
    def fn(text, model, tokenizer):
        return get_chart_parse(
            text,
            model,
            tokenizer,
            st_threshold=0,
            en_threshold=0,
            use_elementwise_dist=args.elemwise,
            sim_fn=args.sim_fn,
            verbose=True,
            normalize=True,
        )

    return fn


class ModelInfluences:
    def __init__(self, args, model, tokenizer, parse_fn):
        self.model = model
        self.tokenizer = tokenizer
        self.parse_fn = parse_fn
        self.args = args

    def dist_helper(self, v1, v2, sim=None):
        sim2fn = {"cosine": distance.cosine, "euclidean": distance.euclidean}
        if sim is not None:
            return sim2fn[sim](v1, v2)
        else:
            return sim2fn[self.args.sim_fn](v1, v2)

    def get_vectors_gaussian(self, input_str, gaussian_noise_idx):
        train_data_collator = collate.VarLengthCollate(None)

        def get_gaussian(inputs, gaussian_noise_idx):
            vec = torch.zeros(1, len(inputs), 512)
            # add 1 for start token
            vec[0][gaussian_noise_idx + 1] = (0.1**0.5) * torch.randn(512)
            return vec

        def tokenizer_helper(inp_slice):
            inp_list = [self.tokenizer(s) for s in inp_slice]
            in_lens = [len(s) for s in inp_list]

            inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
            inp = train_data_collator(inp_to_collate)
            in_len = inp["in_len"].long()
            return inp["in"].transpose(0, 1), in_len

        device = torch.device("cuda")

        sent_tokens, idxs = get_pre_tokenized_info(
            input_str, self.tokenizer, pretrained=False
        )
        model.eval()
        inputs, input_lens = tokenizer_helper([input_str])
        inputs = inputs.to(device)
        input_lens = input_lens.to(device)
        if gaussian_noise_idx != -1:
            gaussian_noise = get_gaussian(inputs[0], gaussian_noise_idx).to(device)
        else:
            gaussian_noise = None

        inp_len = inputs.shape[1]
        mask = self.model.generate_len_mask(inp_len, input_lens).to(device)
        mask_mult = mask.unsqueeze(-1)

        with torch.no_grad():
            outputs = self.model.encoder_only(
                inputs, mask, gaussian_noise=gaussian_noise
            )
        outputs = [hs * (~mask_mult) for hs in outputs]
        hidden_states = [[outputs[-1][0].cpu().numpy()]]

        vecs = get_word_vecs_from_subwords(
            [input_str], hidden_states, tokenizer, ([sent_tokens], [idxs])
        )
        return vecs[0][0]

    def model_independences_gaussian(self, input_text):
        words = input_text.split(" ")
        parses_and_scores = self.parse_fn(input_text, self.model, self.tokenizer)
        tree = parses_and_scores["pred_parse"]
        constituent_list = Graph(tree).get_constituents(self.args.max_depth)
        v1 = self.get_vectors_gaussian(input_text, -1)

        all_diffs_1 = []
        all_diffs_2 = []
        all_rel_dist_1 = []
        all_rel_dist_2 = []

        for (imp_idxs, perturb_idx, less_imp_idxs) in constituent_list:
            span_1 = imp_idxs + [perturb_idx]
            span_2 = less_imp_idxs
            seen = set()
            for _ in range(self.args.num_samples):
                if self.args.control:
                    cdone = False
                    for idx in range(100):
                        chosen_idxs = random.sample(span_1, k=2)
                        control_idx = random.choice(span_2)
                        if abs(control_idx - chosen_idxs[1]) == abs(
                            chosen_idxs[0] - chosen_idxs[1]
                        ):
                            cdone = True
                            break
                    if not cdone:
                        continue
                else:
                    chosen_idxs = random.sample(span_1, k=2)
                    control_idx = random.choice(span_2)

                if (chosen_idxs[0], chosen_idxs[1], control_idx) not in seen:
                    seen.add((chosen_idxs[0], chosen_idxs[1], control_idx))
                    v21 = self.get_vectors_gaussian(input_text, chosen_idxs[0])
                    v22 = self.get_vectors_gaussian(input_text, control_idx)
                    try:
                        diff_1 = self.dist_helper(
                            v1[chosen_idxs[1]], v21[chosen_idxs[1]], sim="euclidean"
                        )
                        diff_2 = self.dist_helper(
                            v1[chosen_idxs[1]], v22[chosen_idxs[1]], sim="euclidean"
                        )
                    except:
                        import pdb

                        pdb.set_trace()
                    all_diffs_1.append(diff_1)
                    all_diffs_2.append(diff_2)
                    all_rel_dist_1.append(abs(chosen_idxs[1] - chosen_idxs[0]))
                    all_rel_dist_2.append(abs(control_idx - chosen_idxs[1]))
        return (
            all_diffs_1,
            all_diffs_2,
            all_rel_dist_1,
            all_rel_dist_2,
            parses_and_scores["pred_parse_score"],
        )

    def __call__(self, sentences):
        d1 = []
        d2 = []
        rel_1 = []
        rel_2 = []
        scores = []
        for sent in tqdm(sentences):
            (
                d1_curr,
                d2_curr,
                rel_1_curr,
                rel_2_curr,
                score,
            ) = self.model_independences_gaussian(sent)
            d1 += d1_curr
            d2 += d2_curr
            rel_1 += rel_1_curr
            rel_2 += rel_2_curr
            scores.append(score)
        return d1, d2, rel_1, rel_2, np.mean(scores)


def get_model_and_tokenizer(args):
    model_name = args.model_name.split("/")[-1].split(".")[0]
    print(model_name)
    N_HEADS = 4
    VEC_DIM = 512
    ENCODER_LAYERS = args.encoder_depth
    DECODER_LAYERS = 2
    if args.dataset == "cogs":
        _, in_vocab, out_vocab, inp_sentences, _ = build_datasets()
    elif args.dataset == "geoquery":
        _, in_vocab, out_vocab, inp_sentences, _ = build_datasets_semparse(
            "semparse/geoquery.pickle"
        )
    else:
        _, in_vocab, out_vocab, inp_sentences, _ = build_datasets_pcfg(
            use_singleton=True, use_no_commas=True
        )
    model = create_model(
        len(in_vocab), len(out_vocab), VEC_DIM, N_HEADS, ENCODER_LAYERS, DECODER_LAYERS
    )
    model.load_state_dict(torch.load(args.model_name, map_location=torch.device("cpu")))

    def tokenizer_fn(model):
        def fn(s, add_special_tokens=True):
            if add_special_tokens:
                return [model.encoder_sos] + in_vocab(s) + [model.encoder_eos]
            else:
                return in_vocab(s)

        return fn

    tokenizer = tokenizer_fn(model)
    device = torch.device("cuda")
    model.to(device)
    return model, tokenizer, inp_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "understanding token independences that a model makes with trees!"
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument(
        "--masking_type", type=str, choices=["attention", "token"], default="attention"
    )
    parser.add_argument(
        "--measure", type=str, choices=["avg", "max", "bow"], default="bow"
    )
    parser.add_argument(
        "--sim_fn", type=str, choices=["euclidean", "cosine"], default="cosine"
    )
    parser.add_argument("--max_depth", type=int, default=10000)
    parser.add_argument("--mask_only", action="store_true")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--elemwise", action="store_true")
    parser.add_argument("--construct_data", action="store_true")
    parser.add_argument("--scores_only", action="store_true")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--vanilla", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--corr", action="store_true")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--parsing_style", choices=["bottom_up", "top_down"], default="top_down"
    )
    args = parser.parse_args()

    model, tokenizer, inp_sentences = get_model_and_tokenizer(args)
    inp_sentences = random.sample(inp_sentences, k=500)

    if args.randomize:
        parse_fn = lambda text, model, tokenizer: {
            "pred_parse": random_parse(text),
            "pred_parse_score": 0.0,
        }
    else:
        parse_fn = get_parse_fn(args)

    model_influence_obj = ModelInfluences(args, model, tokenizer, parse_fn)

    d1, d2, rel_1, rel_2, score = model_influence_obj(inp_sentences)
    num_times = sum([x1 > x2 for x1, x2 in zip(d1, d2)])

    print(num_times, len(d1))
    print(num_times / len(d1))
    print("tree score", score)
    folder_name = "vanilla_models"

    if args.mask_only:
        folder_name += "_mask"
    if args.elemwise:
        folder_name += "_elemwise"
    if args.sim_fn != "cosine":
        folder_name += "_{}_2".format(args.sim_fn)

    if args.randomize:
        folder_name += "_randomize"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    model_name = args.model_name.split("/")[-1]
    if args.vanilla:
        model_name = "{}_{}_{}".format(args.dataset, args.encoder_depth, model_name)
    with open(
        "{}/{}_{}_{}.pickle".format(
            folder_name, model_name, args.masking_type, args.measure
        ),
        "wb",
    ) as writer:
        pickle.dump((d1, d2, rel_1, rel_2), writer)
