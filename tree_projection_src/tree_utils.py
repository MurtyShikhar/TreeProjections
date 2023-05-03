import pickle
import random
from .node import get_node_brackets


def read_inputs_and_parses(input_file):
    def binarize(tree):
        if type(tree) == str:
            return tree
        elif len(tree) == 1:
            return binarize(tree[0])
        else:
            lchild = binarize(tree[0])
            rchild = binarize(tree[1:])
            return (lchild, rchild)

    with open(input_file, "rb") as reader:
        data = pickle.load(reader)
    if "ptb" in input_file or "pcfg" in input_file:
        return data[0], data[1]
    else:
        strs, parses = [], []
        for l in data:
            strs.append(l)
            parses.append(data[l])
    return strs, parses


def right_branching_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            p1 = add_brackets(st, st)
            p2 = add_brackets(st + 1, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def left_branching_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            p1 = add_brackets(st, en - 1)
            p2 = add_brackets(en, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def random_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            random_point = random.choice([k for k in range(st, en)])
            p1 = add_brackets(st, random_point)
            p2 = add_brackets(random_point + 1, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def balanced_parse(input_str):
    input_words = input_str.split(" ")
    num_words = len(input_words)

    def add_brackets(st, en):
        if st == en:
            return input_words[st]
        else:
            mid_point = (en - st + 1) // 2
            p1 = add_brackets(st, st + mid_point - 1)
            p2 = add_brackets(st + mid_point, en)
            return (p1, p2)

    return add_brackets(0, num_words - 1)


def get_parsing_accuracy(predicted_parses, gold_parses):
    def get_brackets(parse):
        p_set = set()

        def get_brackets_helpers(t, st):
            if type(t) == str:
                return 1
            else:
                l1_len = get_brackets_helpers(t[0], st)
                l2_len = get_brackets_helpers(t[1], st + l1_len)
                p_set.add((st, st + l1_len + l2_len - 1))
                return l1_len + l2_len

        get_brackets_helpers(parse, 0)
        return p_set

    if type(gold_parses[0]) != tuple:
        gold_brackets = [get_node_brackets(parse) for parse in gold_parses]
    else:
        gold_brackets = [get_brackets(parse) for parse in gold_parses]
    pred_brackets = [get_brackets(parse) for parse in predicted_parses]

    def get_score(set_1, set_2):
        score = 0.0
        for p in set_2:
            if p in set_1:
                score += 1
        return score

    precision = sum(
        [get_score(gold, pred) for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    recall = sum(
        [get_score(pred, gold) for gold, pred in zip(gold_brackets, pred_brackets)]
    )
    precision /= 1.0 * sum(len(b) for b in pred_brackets)
    recall /= 1.0 * sum(len(b) for b in gold_brackets)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall + 1e-10),
    }
