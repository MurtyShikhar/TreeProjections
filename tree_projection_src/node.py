class Node():
    def __init__(self, tag, child_nodes):
        self.tag = tag
        self.content = child_nodes

    def get_tag(self):
        return self.tag

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        if self.is_terminal():
            return 1
        else:
            return len(self.content)

    def is_terminal(self):
        return type(self.content) == str

    def __repr__(self):
        content_repr = str(self.content)
        return "NODE {}:{}".format(self.tag, content_repr)


def convert_into_linearized_rep(parse, input_str):
    words = input_str.split(" ")
    brackets = get_node_brackets(parse)

    def fn(st, en):
        if st == en:
            return words[st]
        else:
            split_point = brackets[(st, en)]
            lc = fn(st, split_point)
            rc = fn(split_point + 1, en)
            return "( {} {} )".format(lc, rc)

    return fn(0, len(words) - 1)


def get_node_brackets(tree_tuple, only_nps=False, right_branching=True):
    split_vals = {}

    def terminal(t):
        if type(t) in [tuple, list]:
            if len(t) == 1 and t[0].is_terminal():
                return True
        elif t.is_terminal():
            return True
        return False

    NOUN_LIST = ["NN", "NNP", "NNPS", "NNS", "NP"]

    def get_len(t, st):
        if terminal(t):
            return 1
        # if type(t[0]) == str and type(t[1]) == str:
        #    split_vals[(st, st+1)] = st
        #    return 2
        # elif type(t[0]) == str:
        #    l2_len = get_len(t[1], st+1)
        #    split_vals[(st, st + l2_len)] = st
        #    return 1 + l2_len
        elif len(t) == 1:
            return get_len(t[0], st)
        else:
            l1_len = get_len(t[0], st)
            l2_len = get_len(t[1:], st + l1_len)
            if only_nps:
                if (type(t) == node and t.get_tag() in NOUN_LIST) or (
                    type(t) == tuple and t[0].get_tag() in NOUN_LIST
                ):
                    split_vals[(st, st + l1_len + l2_len - 1)] = st + l1_len - 1
            else:
                split_vals[(st, st + l1_len + l2_len - 1)] = st + l1_len - 1
            return l1_len + l2_len

    get_len(tree_tuple, 0)
    return split_vals
