from tqdm import tqdm
import torch
import numpy as np
import random
import collate
from scipy.spatial import distance

device = torch.device("cuda")


def get_all_hidden_states_scratch(
    model,
    tokenizer,
    input_list,
    input_masks=None,
    sum_all=False,
    tqdm_disable=False,
    pre_tokenized=None,
    start_relax_layer=0,
    layer_id=-1,
):
    """
    This function returns hidden states for every input in input_list,
    by calling the encoder of model, and optionally uses masks specified according to
    input_masks (where start_relax_layer decides the layer from which the mask is "activated").
    PARAMS:
    model: a transformer encoder-decoder model (see models/transformer_enc_dec.py for the implementation.)
    tokenizer: converts strings to lists of tokens with special SOS/EOS tokens added
    input_masks: if not None, specifies an attetntion mask
    sum_all: If True, we sum all hidden states (set this to true, if input_masks
             if specified to get context-free vectors of spans.)
    tqdm_disable: If True, uses tqdm progress bars.
    pre_tokenized: If not None, specifies a mapping from word tokens
                   to sub-words.
    start_relax_layer: If not 0, masks in input_masks are only used
                       for layers after start_relax_layer. This is
                       how the t-shaped attention mask is implemented.
    layer_id: If not -1, we output contextual vectors at this layer.

    OUTPUT:
    contextual vectors (if input_masks is None) or
    context free vectors (if input_masks is not None)
    """

    def t_shaped_mask(mask, relax_mask, num_layers):
        ### relax mask only masks padded stuff
        #### relax mask from 0 ... start_relax_layer-1,
        #### mask from start_relax_layer to num_layers - 1
        return [relax_mask] * start_relax_layer + [mask] * (
            num_layers - start_relax_layer
        )

    hidden_states_all = []
    batch_size = 32
    st = 0

    train_data_collator = collate.VarLengthCollate(None)

    def tokenizer_helper(inp_slice):
        inp_list = [tokenizer(s) for s in inp_slice]
        in_lens = [len(s) for s in inp_list]

        inp_to_collate = [{"in": x, "in_len": y} for x, y in zip(inp_list, in_lens)]
        inp = train_data_collator(inp_to_collate)
        in_len = inp["in_len"].long()
        return inp["in"].transpose(0, 1), in_len

    num_layers = model.get_encoder_layers()
    with tqdm(total=len(input_list), disable=tqdm_disable) as progress_bar:
        while st < len(input_list):
            en = min(len(input_list), st + batch_size)
            cslice = input_list[st:en]
            inputs, input_lens = tokenizer_helper(cslice)
            inputs = inputs.to(device)
            input_lens = input_lens.to(device)
            inp_len = inputs.shape[1]
            # input masks specify the inner context
            if input_masks is not None:
                masks_curr = input_masks[st:en]
                masks_padded = []
                for mask in masks_curr:
                    ### in this code base, 1 means MASK, 0 means do not MASK
                    mask_padded = mask + [1] * (inp_len - len(mask))
                    masks_padded.append(mask_padded)
                tree_mask = torch.tensor(masks_padded).to(device)
                relax_mask = model.generate_len_mask(inp_len, input_lens).to(device)
                mask = t_shaped_mask(tree_mask, relax_mask, num_layers)
                mask_mult = tree_mask.unsqueeze(-1)
            else:
                mask = model.generate_len_mask(inp_len, input_lens).to(device)
                mask_mult = mask.unsqueeze(-1)
            model.eval()
            with torch.no_grad():
                outputs = [model.encoder_only(inputs, mask, layer_id=layer_id)]

            # remove vectors for masked stuff
            # REMEMBER: mask is 1 if the token is not attended to, and 0 if the token is attended to.
            outputs = [hs * (~mask_mult) for hs in outputs]
            for idx, _ in enumerate(cslice):
                hidden_states = [outputs[0][idx].cpu().numpy()]
                if sum_all:
                    # the first thing is the [CLS] or [start id] which we ignore
                    # the secnd thing is the [EOS] token which we also ignore.
                    hidden_states = [hs[1:-1].sum(axis=0) for hs in hidden_states]
                hidden_states_all.append(hidden_states)
            progress_bar.update(en - st)
            st = en

    if sum_all:
        return hidden_states_all
    else:
        # NOTE:
        # contextual vecs (hidden_states_all) may possibly correspond to subword tokens.
        # This sub-routine converts sub-word contextual vectors to word contextual vectors
        return get_word_vecs_from_subwords(
            input_list, hidden_states_all, tokenizer, pre_tokenized
        )


def get_idxs(phrase_tokens, sent_tokens, st):
    while st < len(sent_tokens):
        en = st + len(phrase_tokens)
        if sent_tokens[st:en] == phrase_tokens:
            return (st, en)
        st += 1
    ### should not get to this point!
    return (-1, -1)


def get_word_vecs_from_subwords(
    input_list, hidden_states_all, tokenizer, pre_tokenized=None
):
    def get_cumulants(hs, idx_list):
        all_vecs = []
        for (st, en) in idx_list:
            curr_vec = hs[st:en].sum(
                axis=0
            )  # sum up all the subword token representations!
            all_zeros = not curr_vec.any()
            if all_zeros:
                continue
            all_vecs.append(curr_vec)
        return np.stack(all_vecs, axis=0)

    cumulants = []
    if pre_tokenized:
        sent_token_list, word_tokens_all = pre_tokenized
    else:
        sent_token_list = tokenizer(input_list, padding=True).input_ids
    for idx, input_str in enumerate(input_list):
        sent_tokens = sent_token_list[idx]
        curr_hidden_states = hidden_states_all[idx]
        if pre_tokenized:
            idxs = word_tokens_all[idx]
        else:
            words = input_str.split(" ")
            idxs = []
            # go in order.
            st = 0
            for word in words:
                word_tokenized = tokenizer(word, add_special_tokens=False).input_ids
                st_curr, en_curr = get_idxs(word_tokenized, sent_tokens, st)
                idxs.append((st_curr, en_curr))
                st = en_curr

        hs_cumulated = [get_cumulants(hs, idxs) for hs in curr_hidden_states]
        cumulants.append(hs_cumulated)
    return cumulants


def measure_sim_factory(distance_fn):
    def measure_sim(m1, m2):
        if m1.ndim == 2:
            assert len(m1) == len(m2)
            return sum(measure_sim(m1[idx], m2[idx]) for idx in range(len(m1))) / (
                1.0 * len(m1)
            )
        elif distance_fn == distance.cosine:
            return 1.0 - distance_fn(m1, m2)
        else:
            return -1.0 * distance_fn(m1, m2)

    return measure_sim


def get_pre_tokenized_info(tokenizer, input_str):
    """
    e.g.
        input_str: The man is eating bananas
        [The, man, is, eating, bananas]
        [The, man, is, eat##, ##ing, bananas]
        [(0, 0), (1, 1), (2, 2), (3, 4), (5, 5)]
    """
    sent_tokens = tokenizer(input_str)
    words = input_str.split(" ")
    idxs = []
    # go in order.
    st = 0
    for word in words:
        word_tokenized = tokenizer(word, add_special_tokens=False)
        st_curr, en_curr = get_idxs(word_tokenized, sent_tokens, st)
        idxs.append((st_curr, en_curr))
        st = en_curr
    return sent_tokens, idxs


class TreeProjection:
    def __init__(self, model, tokenizer, sim_fn="cosine", normalize=True):
        """
        model: Transformer model to compute tree projections for
        tokenizer: tokenizer that takes sentences
                   and converts it into tokens
        sim_fn: used to construct "d" from the paper
        normalize: If true, we subtract a baseline from the SCI scores
                   to prevent trivial non-contextual encoders from
                   getting high tree scores
        """

        self.model = model
        self.tokenizer = tokenizer
        self.sim_fn = sim_fn
        self.normalize = normalize

    def _mask_all_possible(self, input_str):
        """
        input_str: the input sentence for which we compute all O(n^2) different masks.
        Each mask at index (i, j) specifies an attention pattern such that only words from token_i
        to token_j can be attended to, by the model.
        """
        all_tokens = input_str.split(" ")
        tokenized_inp = self.tokenizer(input_str)
        word_tokenized = [
            self.tokenizer(word, add_special_tokens=False) for word in all_tokens
        ]

        # the starting point of each word
        st_p = 0
        while st_p < len(tokenized_inp):
            en = st_p + len(word_tokenized[0])
            if tokenized_inp[st_p:en] == word_tokenized[0]:
                break
            st_p += 1
        cumulants = [st_p]
        for w_tokenized in word_tokenized:
            cumulants.append(cumulants[-1] + len(w_tokenized))

        # the final thing might be a special token too. 0_0
        en_p = len(tokenized_inp) - cumulants[-1]
        assert en_p > 0

        def generate_attention_mask(st, en):
            ### generate an attention mask such that only words st through en-1 can be attended to.
            # NOTE:
            # 1. for this codebase, True => cannot attend, False => can attend.
            # 2. we always want to attend to the SOS and EOS tokens.
            block_len = cumulants[en] - cumulants[st]
            out = (
                [False] * st_p
                + [True] * (cumulants[st] - st_p)
                + [False] * block_len
                + [True] * (cumulants[-1] - cumulants[en])
                + [False] * en_p
            )
            assert len(out) == len(tokenized_inp)
            return out

        sz = len(all_tokens)
        # l = [1, ... sz]
        # st = [0, 1, 2, ..., sz - l-1]
        all_inputs = {}
        # get new attention masks.
        for l in range(1, sz + 1):
            for st in range(sz - l + 1):
                en = st + l - 1
                # only the stuff from st:en can be attended to.
                all_inputs[(st, en)] = generate_attention_mask(st, en + 1)
        return all_inputs

    def _get_masking_info(self, input_strs):
        masked_strs = []
        curr = 0
        sentence2idx_tuple = []
        input_masks = []
        for inp in input_strs:
            input_dict = self._mask_all_possible(inp)
            curr_keys = [k for k in input_dict]
            masked_strs += [inp] * len(input_dict)
            input_masks += [input_dict[key] for key in curr_keys]
            relative_idxs = [(curr + p, key) for p, key in enumerate(curr_keys)]
            curr += len(curr_keys)
            sentence2idx_tuple.append(relative_idxs)

        return sentence2idx_tuple, masked_strs, input_masks

    ### implements Algorithm-1 from the Appendix
    def _tree_projection(
        self,
        chart_values,
        input_str,
        get_score_parse=False,
        normalize=False,
        is_leaf_fn=None,
        is_invalid_fn=None,
    ):
        num_words = len(input_str.split(" "))

        def tree_projection_recurse(word_list, st, en, randomize=False):
            if is_leaf_fn is not None and is_leaf_fn(word_list, st, en):
                return " ".join(word_list[st : en + 1])
            elif st == en:
                return word_list[st], 0.0
            else:
                curr_split = st
                best_val = -10000
                if randomize:
                    curr_split = random.choice(range(st, en))
                else:
                    for k in range(st, en):
                        if is_invalid_fn is not None and is_invalid_fn(
                            word_list, st, k, en
                        ):
                            continue
                        curr_val = chart_values[(st, k)] + chart_values[(k + 1, en)]
                        if curr_val > best_val:
                            best_val = curr_val
                            curr_split = k
                p1, s1 = tree_projection_recurse(word_list, st, curr_split)
                p2, s2 = tree_projection_recurse(word_list, curr_split + 1, en)
                if normalize:
                    rand_split = random.choice(range(st, en))
                    rand_val = (
                        chart_values[(st, rand_split)]
                        + chart_values[(rand_split + 1, en)]
                    )
                    best_val -= rand_val

                return (p1, p2), s1 + s2 + best_val

        word_list = input_str.split(" ")
        parse, score = tree_projection_recurse(word_list, 0, num_words - 1)

        if get_score_parse:
            return score
        else:
            return chart_values, parse, score

    def __call__(
        self,
        input_str,
        st_threshold,
        ret_dict=False,
        layer_id=-1,
        is_leaf_fn=None,
        is_invalid_fn=None,
    ):
        """
        PARAMS:
        input_str: A string for which we want to compute a tree projection for
        st_threshold: decides the layer at which the t-shaped attention mask
                      blocks outer context attention
        layer_id: The layer at which we get contextual vectors from (usually the last layer)

        OUTPUT:
            if ret_dict is set to true, we output a dictionary containing:
                1. pred_parse: \hat{T}_{proj} obtained via SCI minimization (Eq. 4)
                2. pred_parse_score: the normalized SCI score (Eq. 5) for the tree computed above
                3. sci_chart: contents of the entire chart (for analysis)
            if ret_dict is set to false, we simply output \hat{T}_{proj}
        """
        sent_tokens, idxs = get_pre_tokenized_info(self.tokenizer, input_str)
        sentence2idx_tuple, masked_strs, input_masks = self._get_masking_info(
            [input_str]
        )
        outer_context_vecs = get_all_hidden_states_scratch(
            self.model,
            self.tokenizer,
            [input_str],
            tqdm_disable=True,
            pre_tokenized=([sent_tokens], [idxs]),
            layer_id=layer_id,
        )
        inner_context_vecs = get_all_hidden_states_scratch(
            self.model,
            self.tokenizer,
            masked_strs,
            input_masks,
            sum_all=True,
            start_relax_layer=st_threshold,
            tqdm_disable=True,
            pre_tokenized=([sent_tokens] * len(masked_strs), [idxs] * len(masked_strs)),
            layer_id=layer_id,
        )
        keys = sentence2idx_tuple[0]

        if self.sim_fn == "euclidean":
            measure_sci = measure_sim_factory(distance.euclidean)
        else:
            measure_sci = measure_sim_factory(distance.cosine)

        sci_chart = {}
        all_vector_idxs = outer_context_vecs[0][-1]
        for idx, key in keys:
            st, en = key
            contextual_vectors = all_vector_idxs[st : en + 1].sum(axis=0)
            context_free_vectors = inner_context_vecs[idx][
                -1
            ]  # only consider the words inside the context
            sci_chart[(st, en)] = measure_sci(context_free_vectors, contextual_vectors)
        _, parse, score = self._tree_projection(
            sci_chart,
            input_str,
            get_score_parse=False,
            normalize=self.normalize,
            is_leaf_fn=is_leaf_fn,
            is_invalid_fn=is_invalid_fn,
        )
        if ret_dict:
            return {
                "sci_chart": sci_chart,
                "tscore": score,
                "pred_parse": parse,
            }
        else:
            return parse
