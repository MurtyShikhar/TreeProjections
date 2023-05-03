### ASSUME COGS IS DOWNLOADED in cogs/
import csv
from datasets import Dataset as HFDataset
from vocabulary import WordVocabulary
import os
import pickle
import sequence
from tree_projection_src import get_node_brackets
import sys
sys.path.append('{}/tree_projection_src/'.format(os.getcwd()))

DATA_DIR=os.getcwd() + '/data_utils'
def read_data(splits):
    in_sentences = []
    out_sentences = []
    type_map = {}
    type_list = []

    index_map = {split: [] for split in splits}

    for split in splits:
        with open("{}/cogs/{}.tsv".format(DATA_DIR,split), "r") as reader:
            d = csv.reader(reader, delimiter="\t")
            for line in d:
                i, o, t = line
                index_map[split].append(len(in_sentences))
                in_sentences.append(i)
                out_sentences.append(o)
                tind = type_map.get(t)
                if tind is None:
                    type_map[t] = tind = len(type_map)
                type_list.append(tind)

        assert len(in_sentences) == len(out_sentences)
    return in_sentences, out_sentences, type_map, type_list, index_map


def get_all_parses(in_sentences):
    def process(ex):
        if ex[-1] == ".":
            ex = ex[:-1].strip()
        return ex

    with open("{}/COGS_TREES/all.pickle".format(DATA_DIR), "rb") as reader:
        data_f = pickle.load(reader)
    parses = [get_node_brackets(data_f[process(sent)]) for sent in in_sentences]
    return parses


def build_datasets():
    splits = ["train", "val", "test", "gen"]
    in_sentences, out_sentences, type_map, type_list, index_map = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    parses = get_all_parses(in_sentences)

    in_vocab = WordVocabulary(in_sentences)
    out_vocab = WordVocabulary(out_sentences)

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    def process(ex):
        if ex[-1] == ".":
            ex = ex[:-1].strip()
        return ex

    dataset = {}
    inv = {idx: val for val, idx in type_map.items()}
    type_names = [inv[idx] for idx in range(len(type_map))]
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_sentences, index_map[split])
        type_subset = get_subset(type_list, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        out_subset_tokenized = [out_vocab(s) for s in out_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        out_lens = [len(s) for s in out_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "out": out_subset_tokenized,
            "in_len": in_lens,
            "out_len": out_lens,
            "idxs": index_map[split],
            "type": type_subset,
        }
        dataset_curr = HFDatasetWithTypes.from_dict(data)
        dataset_curr.set_ds(in_vocab, out_vocab, type_names)
        dataset[split] = dataset_curr
    return dataset, in_vocab, out_vocab, [process(x) for x in in_sentences], parses


class HFDatasetWithTypes(HFDataset):
    def set_ds(self, in_vocab, out_vocab, type_names, **kwargs):
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.type_names = type_names
        for k, val in kwargs.items():
            if k == "inv_role_marker_map":
                self.inv_role_marker_map = val

    def start_test(self):
        return sequence.TypedTextSequenceTestState(
            lambda x: " ".join(self.in_vocab(x)),
            lambda x: " ".join(self.out_vocab(x)),
            self.type_names,
        )
