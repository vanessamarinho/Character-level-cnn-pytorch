# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.utils import *
from src.dataset import GermanDataset


def get_args():
    parser = argparse.ArgumentParser(
        """Use pre-trained model to predict new data""")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-g", "--gpu", action="store_true", default=False)
    parser.add_argument("-i", "--input", type=str, default="inference/test.csv", help="path to input file")
    parser.add_argument("-p", "--trained_model", type=str, default="inference/trained_model",
                        help="path to pre-trained model")
    parser.add_argument("-o", "--output", type=str, default="inference/prediction.csv", help="path to output file")
    args = parser.parse_args()
    return args


def inference(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    test_set = GermanDataset(opt.input)
    test_generator = DataLoader(test_set, **test_params)
    model = torch.load(opt.trained_model)
    model.eval()
    test_true = []
    test_prob = []
    for batch in test_generator:
        _, n_true_label = batch
        if opt.gpu:
            batch = [Variable(record).cuda() for record in batch]
        else:
            batch = [Variable(record) for record in batch]
        t_data, _ = batch
        t_predicted_label = model(t_data)
        t_predicted_label = F.softmax(t_predicted_label, dim=1)
        test_prob.append(t_predicted_label)
        test_true.extend(n_true_label)
    test_prob = torch.cat(test_prob, 0)
    test_prob = test_prob.cpu().data.numpy()
    test_true = np.array(test_true)
    test_pred = np.argmax(test_prob, -1)
    fieldnames = ['True label', 'Predicted label', 'Content']
    with open(opt.output, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(test_true, test_pred, test_set.texts):
            writer.writerow(
                {'True label': i + 1, 'Predicted label': j + 1, 'Content': k})

    test_metrics = get_evaluation(test_true, test_prob,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    inference(opt)
