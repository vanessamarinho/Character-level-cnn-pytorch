# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import sys
import csv
from torch.utils.data import Dataset
csv.field_size_limit(sys.maxsize)


class GermanDataset(Dataset):
    def __init__(self, data_path, class_path=None, max_length=1014):
        self.data_path = data_path
        # Include the german umlauts (Ä,Ö,Ü) and ligature (ß)
        self.vocabulary = list(u"abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}äöüß")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file)
            for idx, line in enumerate(reader):
                label = int(line[0])
                # No need to decode the string in Python 3
                texts.append(line[1])
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        if class_path:
            self.num_classes = sum(1 for _ in open(class_path))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        # Original paper says it usually gives worse results when we distinguish between upper-case and lower-case
        raw_text = raw_text.lower()
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label