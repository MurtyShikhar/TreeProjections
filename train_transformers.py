import numpy as np
import random
import os
import torch
from training_utils import *

import argparse
from data_utils import build_datasets, build_datasets_pcfg, build_datasets_semparse
from transformer_helpers import *
import torch.nn.functional as F

### Change this for your own system as appropriate
def working_dir():
    USER = os.environ["USER"]
    dir_name = f"/scr/biggest"

    def helper(dir_name):
        if os.path.exists(dir_name):
            sub_dir = "{}/{}/compositionality".format(dir_name, USER)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            return sub_dir
        else:
            return ""

    try:
        return helper(dir_name)
    except:
        dir_name = f"/scr/smurty/biggest"
        return helper(dir_name)


def get_base_transformer_model(args, in_vocab, out_vocab, model_name=None):
    model = create_model(
        len(in_vocab),
        len(out_vocab),
        args.vec_dim,
        args.n_heads,
        args.encoder_depth,
        args.decoder_depth,
        mode=args.mode,
    )
    if model_name:
        print("loading pretrained model from {}".format(model_name))
        model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    interface = create_model_interface(model)
    return model, interface


def main(args):
    if args.dataset == "cogs":
        datasets, in_vocab, out_vocab, input_strs, parse_splits = build_datasets()
    elif args.dataset == "pcfg":
        datasets, in_vocab, out_vocab, input_strs, parse_splits = build_datasets_pcfg(
            base_folder=args.base_folder,
            use_singleton=args.singleton,
            use_no_commas=args.no_commas,
            tree_transform=args.tree_transform,
        )
    elif args.dataset == "geoquery":
        (
            datasets,
            in_vocab,
            out_vocab,
            input_strs,
            parse_splits,
        ) = build_datasets_semparse("semparse/{}.pickle".format(args.dataset))
    else:
        raise Exception

    model, interface = get_base_transformer_model(
        args, in_vocab, out_vocab, model_name=args.model_load_path
    )

    if args.mode == "mlm":
        in_vocab._add_word("<mask>")
    device = torch.device("cuda:{}".format(args.gpu_id))
    model.to(device)
    if len(args.save_dir) > 0:
        dir_path = working_dir()
        args.save_dir = os.path.join(dir_path, args.save_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    eval_keys = ["val", "gen"]
    train_loop(
        args,
        interface,
        datasets["train"],
        {key: datasets[key] for key in eval_keys},
        device,
        args.save_dir,
        in_vocab=in_vocab,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="cogs")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--vec_dim", type=int, default=512)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--mode", type=str, default="enc_dec")
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no_commas", action="store_true")
    parser.add_argument("--singleton", action="store_true")
    # this is only used if args.dataset == pcfg
    parser.add_argument("--base_folder", type=str, default="m-pcfgset")
    parser.add_argument("--tree_transform", action="store_true")

    args = parser.parse_args()
    set_seed(args)
    wandb.run.name = "{}-{}".format(args.save_dir, args.seed)
    wandb.run.save()
    main(args)
