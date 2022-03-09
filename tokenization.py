import os, sys
import json
import random
import numpy as np
import string
import re
from vocabulary import Vocabulary

unk_words = 0
num_words = 0
unk = []
def numerize(sentence):
    global unk_words, num_words, unk
    new_sentence = []
    for word in sentence:
        # if word.isnumeric():
        #     new_sentence.append(vocab.word2index['<UNK>'])
        # else:
        num_words +=1 
        try:
            new_sentence.append(vocab.word2index[word])
        except KeyError:
            new_sentence.append(vocab.word2index['<UNK>'])
            unk_words += 1
            unk.append(word)
    assert len(sentence) == len(new_sentence), "Numerization failed."
    return new_sentence


DATASET = ['train', 'dev']
vocab = Vocabulary()
sentence_lengths = []
target_indices = []
for dataset in DATASET:
    unk_words = 0 
    num_words = 0
    datapath = 'preprocessed_{}.json'.format(dataset)
    with open(datapath, 'r') as reader:
        data = json.load(reader)
        reader.close()
    new_data = {}
    for j, i in data.items():
        sentence1 = i['sentence1']
        sentence2 = i['sentence2']
        target1 = i['index1']
        target2 = i['index2']
        label = i['label']
        new_sentence1 = numerize(sentence1)
        new_sentence2 = numerize(sentence2)

        # Target Masks
        target_weight = 1.1
        others_weight = 1.0
        # others_weight = (1 - target_weight) / (len(new_sentence1) - 1)
        mask1 = np.ones_like(new_sentence1) * others_weight
        mask1[target1] = target_weight
        mask1 = list(mask1)
        # others_weight = (1 - target_weight) / (len(new_sentence2) - 1)
        mask2 = np.ones_like(new_sentence2) * others_weight
        mask2[target2] = target_weight
        mask2 = list(mask2)

        # PADDING
        ## The max length of a sentence is 88
        ## The max target index is 64
        ## Pad to the max length : 88
        pad_length = 48
        pad = (pad_length - len(new_sentence1))*[0]
        # new_sentence1 += pad
        # mask1 += pad

        pad = (pad_length - len(new_sentence2))*[0]
        # new_sentence2 += pad
        # mask2 += pad

        assert new_sentence1[target1] == new_sentence2[target2], f'Target mismatch: {new_sentence1[target1]} {new_sentence2[target2]}'

        new_data[j] = {
            'sentence1' : new_sentence1,
            'sentence2' : new_sentence2,
            'target1'   : target1,
            'target2'   : target2,
            'mask1'     : mask1,
            'mask2'     : mask2,
            'label'     : label
        }
    print('UNK Words ', unk_words)
    print('Num Words ', num_words)
    print(list(set(unk)))
    with open(f'data/tokenized_{dataset}.json', 'w') as writer:
        json.dump(new_data, writer)
        writer.close()
