from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
import sequence
import pickle
from tree_projection_src import get_node_brackets


def read_data_semparse(path_name, splits):
    in_sentences = []
    out_sentences = []
    parses = []
    index_map = {split: [] for split in splits}

    with open(path_name, "rb") as reader:
        data = pickle.load(reader)

    for split in splits:
        if split == "gen":
            all_inps = [inp for inp, _ in data["test"]]
            all_tgts = [tgt for _, tgt in data["test"]]
        else:
            all_inps = [inp for inp, _ in data[split]]
            all_tgts = [tgt for _, tgt in data[split]]
        for inp, tgt in zip(all_inps, all_tgts):
            index_map[split].append(len(in_sentences))
            in_sentences.append(inp)
            out_sentences.append(tgt)
        assert len(in_sentences) == len(out_sentences)
    return in_sentences, out_sentences, index_map


def get_all_parses(in_sentences):
    def process(ex):
        if ex[-1] == ".":
            ex = ex[:-1].strip()
        return ex.strip()

    with open("geoquery_trees/all.pickle", "rb") as reader:
        data_f = pickle.load(reader)
    parses = [get_node_brackets(data_f[process(sent)]) for sent in in_sentences]
    return parses


def build_datasets_semparse(path_name):
    splits = ["train", "val", "gen"]
    in_sentences, out_sentences, index_map = read_data_semparse(path_name, splits)
    gold_parses = get_all_parses(in_sentences)
    print("num examples: {}".format(len(in_sentences)))
    in_vocab = WordVocabulary(in_sentences)
    out_vocab = WordVocabulary(out_sentences)

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]

    def process(ex):
        if ex[-1] == ".":
            ex = ex[:-1].strip()
        return ex.strip()

    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        out_subset_tokenized = [out_vocab(s) for s in out_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        out_lens = [len(s) for s in out_subset_tokenized]
        data = {
            "in": in_subset_tokenized,
            "idxs": index_map[split],
            "out": out_subset_tokenized,
            "in_len": in_lens,
            "out_len": out_lens,
        }
        dataset_curr = HFDatasetNoTypes.from_dict(data)
        dataset_curr.set_ds(in_vocab, out_vocab)
        dataset[split] = dataset_curr
    return dataset, in_vocab, out_vocab, [process(x) for x in in_sentences], gold_parses


class HFDatasetNoTypes(HFDataset):
    def set_ds(self, in_vocab, out_vocab, **kwargs):
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab

    def start_test(self):
        return sequence.TextSequenceTestState(
            lambda x: " ".join(self.in_vocab(x)), lambda x: " ".join(self.out_vocab(x))
        )
