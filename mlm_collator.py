from collate import VarLengthCollate
from collections import defaultdict as ddict
import numpy as np

from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)
from data_utils import build_datasets_semparse
from transformer_helpers import create_model, create_model_interface


default_collate_fn = VarLengthCollate(tokenizer=None)
wwm_probability = 0.2
mask_token_id = None

class MaskCollator():
    def __init__(self, default_collator, wwm_probability, mask_token_id):
        self.default_collator = default_collator
        self.wwm_probability = wwm_probability
        self.mask_token_id = mask_token_id
    
    def __call__(self, features):
        for feature in features:
            # Randomly mask words
            input_ids = feature["in"]
            mask = np.random.binomial(1, self.wwm_probability, (len(input_ids),))
            new_labels = [ex for ex in feature['in']]
            for word_id in np.where(mask)[0]:
                word_id = word_id.item()
                input_ids[word_id] = self.mask_token_id
            feature['labels'] = new_labels
        return self.default_collator(features)


if __name__ == '__main__':
    datasets, in_vocab, out_vocab, input_strs, parse_splits = build_datasets_semparse('semparse/geoquery.pickle')
    print(len(in_vocab))
    in_vocab._add_word('<mask>')
    print(len(in_vocab))
    mask_token_id = in_vocab['<mask>']
    print(datasets['train'])
    train_dataloader = DataLoader(
            datasets['train'],
            sampler=SequentialSampler(datasets['train']),
            batch_size=32,
            collate_fn=whole_word_masking_data_collator
        )


    model = create_model(len(in_vocab), len(out_vocab), 256, 2, 4, 2, mode='mlm')
    interface = create_model_interface(model)

    for batch in train_dataloader:
        out = interface(batch)
        print(out.loss)
