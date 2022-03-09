from multiprocessing import Value
import os, sys
import json
import random
import numpy as np
import string
import re
import torch
from vocabulary import Vocabulary
from torch.nn.utils.rnn import pad_sequence


class DataLoader:

    def __init__(self,
                 datapath,
                 batch_size,
                 shuffle=False,
                 random_unk=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_unk = random_unk
        if random_unk:
            vocab = Vocabulary()
            self.unk_tag = vocab.word2index['<UNK>']
        with open(datapath, 'r') as reader:
            data_dict = json.load(reader)
            reader.close()
        self.data_len = len(data_dict.keys())
        self.sentences1, self.sentences2 = [], []
        self.sentence_lengths = []
        self.target_pairs = []
        self.masks1, self.masks2 = [], []
        self.labels = []
        for values in data_dict.values():
            self.sentences1.append(values['sentence1'])
            self.sentences2.append(values['sentence2'])
            self.sentence_lengths.append(list(self.unpadded_lengths(values['sentence1'], values['sentence2'])))
            self.target_pairs.append([values['target1'], values['target2']])
            self.masks1.append(values['mask1'])
            self.masks2.append(values['mask2'])
            self.labels.append(values['label'])
        del data_dict
        print('INFO: Data parsed.')
        self.sentences1 = np.array(self.sentences1)
        self.sentences2 = np.array(self.sentences2)
        self.sentence_lengths = np.array(self.sentence_lengths)
        self.target_pairs = np.array(self.target_pairs)
        self.masks1 = np.array(self.masks1)
        self.masks2 = np.array(self.masks2)
        self.labels = np.array(self.labels)
        if shuffle:
            self.shuffle_data()
        self.counter = 0
        self.n_batch = (self.data_len + self.batch_size - 1) // self.batch_size

    def shuffle_data(self):
        permutation = np.random.permutation(self.data_len)
        self.sentences1 = self.sentences1[permutation]
        self.sentences2 = self.sentences2[permutation]
        self.sentence_lengths = self.sentence_lengths[permutation]
        self.target_pairs = self.target_pairs[permutation]
        self.masks1 = self.masks1[permutation]
        self.masks2 = self.masks2[permutation]
        self.labels = self.labels[permutation]
        print("INFO: Data shuffled.")

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.data_len:
            last = min(self.counter + self.batch_size, self.data_len)
            # sentence_batch = torch.tensor(self.sentence_pairs[self.counter : last])
            sentence1_batch = [torch.tensor(sentences).long() for sentences in self.sentences1[self.counter : last]]
            sentence2_batch = [torch.tensor(sentences).long() for sentences in self.sentences2[self.counter : last]]

            sentence1_batch = pad_sequence(sentence1_batch, batch_first=True, padding_value=0)
            sentence2_batch = pad_sequence(sentence2_batch, batch_first=True, padding_value=0)
            # print(self.sentence_lengths[self.counter : last].shape)
            if self.random_unk:
                perm1 = [np.random.randint(0, l) for l in self.sentence_lengths[self.counter : last][:,0]]
                perm2 = [np.random.randint(0, l) for l in self.sentence_lengths[self.counter : last][:,1]]

                sentence1_batch[range(self.batch_size), 0, perm1] = self.unk_tag
                sentence1_batch[range(self.batch_size), 1, perm2] = self.unk_tag

            lengths_batch = torch.tensor(self.sentence_lengths[self.counter : last])
            target_batch = torch.tensor(self.target_pairs[self.counter : last])
            mask1_batch = [torch.tensor(mask).long() for mask in self.masks1[self.counter : last]]
            mask2_batch = [torch.tensor(mask).long() for mask in self.masks2[self.counter : last]]
            mask1_batch = pad_sequence(mask1_batch, batch_first=True, padding_value=0)
            mask2_batch = pad_sequence(mask2_batch, batch_first=True, padding_value=0)

            label_batch = torch.tensor(self.labels[self.counter : last])

            self.counter = last

            return sentence1_batch, sentence2_batch, lengths_batch, target_batch, mask1_batch, mask2_batch, label_batch

        else:
            self.counter = 0
            if self.shuffle: self.shuffle_data()
            raise StopIteration

    def unpadded_lengths(self, sentence1, sentence2):
        try:
            l1 = sentence1.index(0)
        except ValueError:
            l1 = len(sentence1)
        try:
            l2 = sentence2.index(0)
        except ValueError:
            l2 = len(sentence2)
        return l1, l2

if __name__ == '__main__':
    datapath = 'data/tokenized_train.json'
    dataloader = DataLoader(datapath,
                            batch_size=1,
                            shuffle=False)

    counter = 0
    for sentences, targets, masks, label in dataloader:
        print(masks)
        print(torch.tensor(masks).float())
        print(sentences.shape)
        print(targets.shape)
        print(masks.shape)
        print(label.shape)
        counter += 1
        if counter == 30: break