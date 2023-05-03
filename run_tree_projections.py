from data_utils import build_datasets, build_datasets_pcfg, build_datasets_semparse
from data_utils.pcfg_helpers import is_leaf_fn_pcfg, get_parsing_accuracy_pcfg

from transformer_helpers import create_model
import torch
import random
import argparse
import os
import numpy as np
import pickle

from tree_projection_src import TreeProjection
from tree_projection_src import (
    left_branching_parse,
    right_branching_parse,
    random_parse,
    get_parsing_accuracy,
    read_inputs_and_parses,
)


def get_model(model_name, data, encoder_depth, **kwargs):
    if data == "cogs":
        _, in_vocab, out_vocab, _, _ = build_datasets()
    elif data == "geoquery":
        _, in_vocab, out_vocab, _, _ = build_datasets_semparse(
            "semparse/{}.pickle".format(args.data)
        )
    elif data == "pcfg":
        _, in_vocab, out_vocab, _, _ = build_datasets_pcfg(
            base_folder=kwargs["base_folder"],
            use_singleton=kwargs["singleton"],
            use_no_commas=kwargs["no_commas"],
        )

    N_HEADS = 4
    VEC_DIM = 512
    ENCODER_LAYERS = encoder_depth
    DECODER_LAYERS = 2

    if "mlm" in model_name:
        model = create_model(
            len(in_vocab),
            len(out_vocab),
            VEC_DIM,
            N_HEADS,
            ENCODER_LAYERS,
            DECODER_LAYERS,
            mode="mlm",
        )
    else:
        model = create_model(
            len(in_vocab),
            len(out_vocab),
            VEC_DIM,
            N_HEADS,
            ENCODER_LAYERS,
            DECODER_LAYERS,
        )
    if len(model_name) > 0:
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))

    def tokenizer_fn(model):
        def fn(s, add_special_tokens=True):
            if add_special_tokens:
                return [model.encoder_sos] + in_vocab(s) + [model.encoder_eos]
            else:
                return in_vocab(s)

        return fn

    tokenizer = tokenizer_fn(model)
    return model, tokenizer


def get_scores(args, input_strs, gold_parses, get_data_for_lw_parser=False):
    device = torch.device("cuda")
    model, tokenizer = get_model(
        args.model_name,
        args.data,
        args.encoder_depth,
        base_folder=args.base_folder,
        singleton=args.singleton,
        no_commas=args.no_commas,
    )

    if get_data_for_lw_parser:
        st_thresholds = [(args.encoder_depth) // 2]
    else:
        st_thresholds = [idx for idx in range((args.encoder_depth // 2) + 1)]
    model.to(device)
    model_name = args.model_name.split("/")[-1].split(".")[0]
    print(model_name)
    model_folder = args.model_name.split("/")[-2]
    print(model_folder)

    all_scores = [{}, {}, {}]
    all_parses = {}

    if args.data == "pcfg":
        is_leaf_fn = is_leaf_fn_pcfg
    else:
        is_leaf_fn = None

    tree_projector = TreeProjection(
        model, tokenizer, sim_fn=args.sim_fn, normalize=True
    )
    for st in st_thresholds:
        # input_str: The man is eating bananas
        if st == args.layer_id:
            break
        parses_and_scores = [
            tree_projector(
                input_str,
                st,
                ret_dict=True,
                layer_id=args.layer_id,
                is_leaf_fn=is_leaf_fn,
                is_invalid_fn=None,
            )
            for input_str in input_strs
        ]

        parses = [x["pred_parse"] for x in parses_and_scores]
        scores = [x["tscore"] for x in parses_and_scores]

        if args.print_parses:
            all_parses[st] = {sent: parse for sent, parse in zip(input_strs, parses)}
        else:
            if args.data == "pcfg":
                parsing_acc = get_parsing_accuracy_pcfg(
                    parses, gold_parses, take_best=False
                )
                all_scores[1][(st, 0)] = parsing_acc["f1"]
            else:
                parsing_acc = get_parsing_accuracy(parses, gold_parses)
                all_scores[1][(st, 0)] = parsing_acc["f1"]
            all_scores[0][(st, 0)] = np.mean(scores)
            print("tscore: ", all_scores[0][(st, 0)])
            print("tparseval: ", all_scores[1][(st, 0)])

    if get_data_for_lw_parser:
        return all_parses[st_thresholds[0]]
    if args.print_parses:
        folder_name = "all_parses_m{}_{}".format(args.encoder_depth, args.data)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open("{}/{}.pickle".format(folder_name, model_name), "wb") as writer:
            pickle.dump(all_parses, writer)
    else:
        folder_name = "all_scores_m{}_{}_{}".format(
            args.encoder_depth, args.data, model_folder
        )
        if args.layer_id != -1:
            folder_name += "_{}".format(args.layer_id)
        if args.singleton:
            folder_name += "_singleton"
        if args.no_commas:
            folder_name += "_no_commas"
        if args.base_folder != "m-pcfgset":
            folder_name += "_{}".format(args.base_folder)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        with open("{}/{}.pickle".format(folder_name, model_name), "wb") as writer:
            pickle.dump(all_scores, writer)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--print_parses", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument(
        "--sim_fn", type=str, default="cosine", choices=["cosine", "euclidean"]
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["cogs", "pcfg", "geoquery"],
        default="cogs",
    )
    parser.add_argument("--singleton", action="store_true")
    parser.add_argument("--no_commas", action="store_true")
    parser.add_argument("--base_folder", type=str, default="m-pcfgset")
    parser.add_argument("--layer_id", type=int, default=-1)
    parser.add_argument("--get_baselines", action="store_true")
    args = parser.parse_args()
    set_seed(args)

    DATA_DIR = os.getcwd()
    if args.data == "cogs":
        dat_folder = "{}/data_utils/COGS_TREES".format(DATA_DIR)
        data_file = "{}/train.pickle".format(dat_folder)
    elif args.data == "geoquery":
        dat_folder = "{}_trees".format(args.data)
        data_file = "{}/train.pickle".format(dat_folder)
    else:
        data_file = "{}/pcfg_train_singleton_no_commas.pickle".format(args.base_folder)

    model_name = args.model_name.split("/")[-1].split(".")[0]
    print(model_name)
    input_strs, gold_parses = read_inputs_and_parses(data_file)
    if args.data in ["cogs", "pcfg"]:
        sampled_idxs = random.sample(
            range(len(input_strs)), k=min(len(input_strs), 5000)
        )
        input_strs = [input_strs[idx] for idx in sampled_idxs]
        gold_parses = [gold_parses[idx] for idx in sampled_idxs]
    if args.get_baselines:
        lbranch = [left_branching_parse(s) for s in input_strs]
        rbranch = [right_branching_parse(s) for s in input_strs]
        random_tree = [random_parse(s) for s in input_strs]
        baseline2trees = {
            "left_branching": lbranch,
            "right_branching": rbranch,
            "random": random_tree,
        }
        for key in baseline2trees:
            parses = baseline2trees[key]
            if args.data == "pcfg":
                parsing_acc = get_parsing_accuracy_pcfg(parses, gold_parses)
            else:
                parsing_acc = get_parsing_accuracy(parses, gold_parses)
            print(key, parsing_acc)
    else:
        get_scores(args, input_strs, gold_parses)
