from data_utils import build_datasets_pcfg, build_datasets
from transformer_helpers import create_model, create_model_interface
import torch
from training_utils import get_opt, get_scheduler, get_grad_norm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm
import wandb
import random
from vocabulary import WordVocabulary
import argparse

from tree_projection_src import (
    read_inputs_and_parses,
    get_all_hidden_states_scratch,
    convert_into_linearized_rep,
    get_pre_tokenized_info,
)


from data_utils.pcfg_helpers import get_parsing_accuracy_pcfg


def convert_to_tuple_cogs(inp):
    def helper(idx, curr_chunk):
        if inp[idx] == "(":
            j, chunk = helper(idx + 1, "")
            if j + 1 < len(inp):
                j2, chunk2 = helper(j, "")
                return j2, (chunk, chunk2)
            else:
                return j, chunk
        elif inp[idx] == ")":
            return idx + 1, " ".join(curr_chunk)
        else:
            chunk = inp[idx]
            if idx + 1 < len(inp) and inp[idx + 1] not in ["(", ")"]:
                ret = idx + 2
                while ret < len(inp) and inp[ret] == ")":
                    ret += 1
                return ret, (chunk, inp[idx + 1])
            else:
                j, rhs = helper(idx + 1, "")
                if len(rhs) == 0:
                    return j, chunk
                else:
                    return j, (chunk, rhs)

    try:
        return helper(0, "")[1]
    except:
        return None


def convert_into_dataset(features, parses, labels):
    out_vocab = WordVocabulary(labels)

    class CustomDataset(Dataset):
        def __init__(self, feats, parses, labels, vocab):
            super().__init__()
            self.feats = feats
            self.labels = labels
            self.parses = parses
            self.vocab = vocab

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {"features": self.feats[idx], "labels": self.labels[idx]}

    all_idxs = [idx for idx, _ in enumerate(labels)]
    random.shuffle(all_idxs)
    train_split_idx = int(0.8 * len(all_idxs))
    train_idxs = all_idxs[:train_split_idx]
    val_idxs = all_idxs[train_split_idx:]

    splits = {"train": train_idxs, "val": val_idxs}

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    dataset = {}
    for split in splits:
        feat_subset = get_subset(features, splits[split])
        labels_subset = get_subset(labels, splits[split])
        parse_subset = get_subset(parses, splits[split])
        labels_subset_tokenized = [out_vocab(s) for s in labels_subset]
        dataset[split] = CustomDataset(
            feat_subset, parse_subset, labels_subset_tokenized, out_vocab
        )

    return dataset["train"], dataset["val"], out_vocab


def collator(batch):
    def pad_to_max_len(inps, max_len, padding):
        # vecs: list of vectors of shape s_len x vec_dim
        ### we are going to pad (on the right) with a vector of dim (max_len - s_len) x vec_dim
        ret_inp = []
        for inp in inps:
            residual = max_len - len(inp)
            if inp == [454, 0] and type(padding) == int:
                import pdb

                pdb.set_trace()
            if residual > 0:
                if type(inp) == list:
                    ret_inp.append(inp + [padding] * residual)
                else:
                    v = padding[None, :].repeat(residual, 1)
                    ret_inp.append(torch.cat([inp, v]))
            else:
                ret_inp.append(inp)
        return ret_inp, torch.tensor([len(inp) for inp in inps])

    features = [torch.tensor(b["features"]) for b in batch]
    labels = [b["labels"] for b in batch]
    max_feat_len = max(len(c) for c in features)
    max_label_len = max(len(c) for c in labels)

    features_padded, in_lens = pad_to_max_len(
        features, max_feat_len, torch.zeros(len(features[0][0]))
    )
    labels_padded, out_lens = pad_to_max_len(labels, max_label_len, 0)

    return {
        "in": torch.stack(features_padded, dim=0),
        "out": torch.tensor(labels_padded),
        "in_len": in_lens,
        "out_len": out_lens,
    }


def eval_func(args, model, dataset, device):
    model.model.eval()
    val_dataloader = DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=256, collate_fn=collator
    )

    def compare(out, dataset, out_lens, target, target_lens):
        acc = 0
        for idx, _ in enumerate(target):
            curr_target = target[idx]
            curr_target_len = target_lens[idx]
            curr_out = out[idx]
            curr_out_len = out_lens[idx]
            if curr_out_len == curr_target_len:
                o1 = curr_target[:curr_target_len]
                o2 = curr_out[:curr_out_len]
                acc += torch.all(o1 == o2).item()
        return acc

    curr_acc = 0
    total = 0
    parses = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch_gpu = {}
            for key in batch:
                batch_gpu[key] = batch[key].to(device)
            res = model(batch_gpu)
            out, lens = model.decode_outputs(res)
            pred_targets = out.argmax(axis=-1).transpose(0, 1)[:, :-1]
            for pred, _len in zip(pred_targets, lens):
                linearized_parse = dataset.vocab(pred[:_len].tolist())
                parses.append(convert_to_tuple_cogs(linearized_parse))
                # if args.data == 'pcfg':
                #    try:
                #        parses.append(convert_to_tuple(linearized_parse))
                #    except:
                #        parses.append(None)
                # else:
                #
                #     parses.append(convert_to_tuple_cogs(linearized_parse))
            curr_acc += compare(
                pred_targets, dataset, lens, batch_gpu["out"], batch_gpu["out_len"] - 1
            )
            total += len(batch_gpu["out_len"])

    filtered_gold = []
    filtered_preds = []
    for parse, gold_parse in zip(parses, dataset.parses):
        if parse is not None:
            filtered_preds.append(parse)
            filtered_gold.append(gold_parse)

    parsing_acc = get_parsing_accuracy_pcfg(filtered_preds, filtered_gold)
    return (curr_acc / total), parsing_acc["f1"]
    # return parsing_acc['f1'], (curr_acc / total)


def train_helper(args, model, train_dataset, val_dataset, run_name=None):
    if run_name:
        wandb.run.name = run_name
    num_steps = 0
    max_grad_norm = 1
    train_batch_size = 32
    accum_steps = 1
    eval_every = 1000
    max_steps = 30000
    patience = 10
    lr = 1e-3
    curr_patience = 0

    opt = get_opt(lr, model)
    scheduler = get_scheduler(opt, max_steps)

    device = torch.device("cuda")

    model.model.to(device)
    best_acc = 0.0
    while True:
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=train_batch_size,
            collate_fn=collator,
        )
        total_train_sz = len(train_dataset)
        if num_steps > max_steps or curr_patience >= patience:
            break
        with torch.enable_grad(), tqdm(total=total_train_sz) as progress_bar:
            losses = []
            for curr_batch_dict in train_dataloader:
                model.model.train()
                curr_batch_dict_gpu = {}
                for key in curr_batch_dict:
                    curr_batch_dict_gpu[key] = curr_batch_dict[key].to(device)
                loss_curr = model(curr_batch_dict_gpu).loss
                progress_bar.update(curr_batch_dict["in"].shape[0])
                loss_curr /= accum_steps
                loss_curr.backward()
                losses.append(loss_curr.item())
                if len(losses) == accum_steps:
                    num_steps += 1
                    torch.nn.utils.clip_grad_norm_(
                        model.model.parameters(), max_grad_norm
                    )
                    progress_bar.set_postfix(
                        {"loss": sum(losses), "num_steps": num_steps}
                    )
                    grad_norm = get_grad_norm(model.model)
                    wandb.log(
                        {
                            "loss": sum(losses),
                            "grad_norm": grad_norm,
                            "iteration": num_steps,
                        }
                    )
                    opt.step()
                    scheduler.step()
                    model.model.zero_grad()
                    losses = []
                    if num_steps % eval_every == 0:
                        print("Evaluating at step {}".format(num_steps))
                        _, curr_acc = eval_func(args, model, val_dataset, device)
                        # save model

                        to_log = {
                            "iteration": num_steps,
                            "gen acc": curr_acc,
                            "best acc": max(curr_acc, best_acc),
                        }
                        wandb.log(to_log)
                        print("curr acc", curr_acc)
                        if curr_acc > best_acc:
                            curr_patience = 0
                            best_acc = curr_acc
                        else:
                            curr_patience += 1
                    if curr_patience >= patience or num_steps > max_steps:
                        break
    print("Best Accuracies,", best_acc)
    return best_acc


def train_decoder_only_helper(args, features, labels, parses, run_name):
    N_HEADS = 4
    VEC_DIM = 512
    DECODER_LAYERS = 1

    out_vocab = None

    # convert parses into indices

    train_dataset, val_dataset, out_vocab = convert_into_dataset(
        features, parses, labels
    )

    model = create_model(
        -1, len(out_vocab), VEC_DIM, N_HEADS, -1, DECODER_LAYERS, is_null_encoder=True
    )

    model_interface = create_model_interface(model, is_null_encoder=True)
    val_acc = train_helper(args, model_interface, train_dataset, val_dataset, run_name)
    return val_acc


def train_decoder_only(args, input_strs, labels, parses, run_name=None):
    device = torch.device("cuda")
    model_path = args.model_path.split("/")[-1].split(".")[0]
    if args.data == "pcfg":
        _, in_vocab, out_vocab, _, _ = build_datasets_pcfg(
            use_singleton=True, use_no_commas=True
        )
    else:
        _, in_vocab, out_vocab, _, _ = build_datasets()

    N_HEADS = 4
    VEC_DIM = 512
    ENCODER_LAYERS = args.encoder_depth
    DECODER_LAYERS = 2

    model = create_model(
        len(in_vocab), len(out_vocab), VEC_DIM, N_HEADS, ENCODER_LAYERS, DECODER_LAYERS
    )
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    def tokenizer_fn(model):
        def fn(s, add_special_tokens=True):
            if add_special_tokens:
                return [model.encoder_sos] + in_vocab(s) + [model.encoder_eos]
            else:
                return in_vocab(s)

        return fn

    tokenizer = tokenizer_fn(model)
    model.to(device)

    # encode all strs with the model, and train a transformer decoder to output
    # the parse as a linearized tree

    sent_tokens_all = []
    idxs_all = []
    for input_str in input_strs:
        sent_tokens, idxs = get_pre_tokenized_info(
            input_str, tokenizer, pretrained=False
        )
        sent_tokens_all.append(sent_tokens)
        idxs_all.append(idxs)

    contextual_vectors = get_all_hidden_states_scratch(
        model,
        tokenizer,
        input_strs,
        tqdm_disable=False,
        pre_tokenized=(sent_tokens_all, idxs_all),
    )
    features = [v[0] for v in contextual_vectors]

    if run_name:
        run_name += "_{}_{}".format(model_path, args.encoder_depth)
    else:
        run_name = model_path
    best_acc = train_decoder_only_helper(args, features, labels, parses, run_name)
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--encoder_depth", type=int)
    parser.add_argument(
        "--data",
        type=str,
        choices=["cogs", "ptb", "pcfg", "geoquery"],
        default="cogs",
    )

    args = parser.parse_args()

    if args.data == "cogs":
        dat_folder = "COGS_TREES"
        data_file = "{}/train.pickle".format(dat_folder)
    elif args.data == "geoquery":
        dat_folder = "{}_trees".format(args.data)
        data_file = "{}/train.pickle".format(dat_folder)
    else:
        data_file = "{}/pcfg_train_singleton_no_commas.pickle".format(args.base_folder)

    input_strs, gold_parses = read_inputs_and_parses(data_file)
    sampled_idxs = random.sample(range(len(input_strs)), k=5000)
    input_strs = [input_strs[idx] for idx in sampled_idxs]
    gold_parses = [gold_parses[idx] for idx in sampled_idxs]
    if args.data == "pcfg":
        from data_utils.pcfg_helpers import (
            convert_to_linearized_rep_pcfg,
            tree_transformation,
        )

        gold_parses = [tree_transformation(parse) for parse in gold_parses]
        gold_parses_linearized = [
            convert_to_linearized_rep_pcfg(parse) for parse in gold_parses
        ]
    else:
        gold_parses_linearized = [
            convert_into_linearized_rep(parse, inp)
            for parse, inp in zip(gold_parses, input_strs)
        ]
    train_decoder_only(args, input_strs, gold_parses_linearized, gold_parses)
