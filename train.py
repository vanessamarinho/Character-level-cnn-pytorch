# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.utils import *
from src.dataset import GermanDataset
from src.character_level_cnn import CharacterLevelCNN


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Character-level convolutional networks for text classification""")
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    parser.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-n", "--num_epochs", type=int, default=10)
    parser.add_argument("-l", "--lr", type=float, default=0.001)  # recommended learning rate for sgd is 0.01, while for adam is 0.001
    parser.add_argument("-g", "--gpu", action="store_true", default=False)
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["agnews", "dbpedia", "yelp_review", "yelp_review_polarity", "amazon_review",
                                 "amazon_polarity", "sogou_news", "yahoo_answers", "german"], default="german",
                        help="public dataset used for experiment. If this parameter is set, parameters input and output are ignored")
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    args = parser.parse_args()
    return args


def train(opt):
    if opt.dataset in ["agnews", "dbpedia", "yelp_review", "yelp_review_polarity", "amazon_review",
                       "amazon_polarity", "sogou_news", "yahoo_answers", "german"]:
        opt.input, opt.output = get_default_folder(opt.dataset, opt.feature)

    if opt.dataset == "german":
        opt.alphabet = u"abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}äöüß"

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    output_file = open(opt.output + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    training_set = GermanDataset(opt.input + os.sep + "train.csv", opt.input + os.sep + "classes.txt", opt.max_length)
    test_set = GermanDataset(opt.input + os.sep + "test.csv", opt.input + os.sep + "classes.txt", opt.max_length)
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)

    if opt.feature == "small":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=256, n_fc_neurons=1024)

    elif opt.feature == "large":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=1024, n_fc_neurons=2048)
    else:
        sys.exit("Invalid feature mode!")

    if opt.gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    model.train()
    num_iter_per_epoch = len(training_generator)
    best_accuracy = 0

    for epoch in range(opt.num_epochs):
        for iter, batch in enumerate(training_generator):
            _, n_true_label = batch
            if opt.gpu:
                batch = [Variable(record).cuda() for record in batch]
            else:
                batch = [Variable(record) for record in batch]
            t_data, t_true_label = batch

            optimizer.zero_grad()
            t_predicted_label = model(t_data)
            n_prob_label = t_predicted_label.cpu().data.numpy()

            loss = criterion(t_predicted_label, t_true_label)
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(n_true_label, n_prob_label, list_metrics=["accuracy", "loss"])
            print(
                "Training: Iteration: {}/{} Epoch: {}/{} Loss: {} Accuracy: {}".format(iter + 1, num_iter_per_epoch,
                                                                                           epoch + 1, opt.num_epochs,
                                                                                           training_metrics["loss"],
                                                                                           training_metrics[
                                                                                               "accuracy"]))

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
        model.train()

        test_metrics = get_evaluation(test_true, test_prob,
                                      list_metrics=["accuracy", "loss", "confusion_matrix"])

        output_file.write(
            "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, opt.num_epochs,
                training_metrics["loss"],
                training_metrics["accuracy"],
                test_metrics["loss"],
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print (
            "\tTest:Epoch: {}/{} Loss: {} Accuracy: {}\r".format(epoch + 1, opt.num_epochs, test_metrics["loss"],
                                                                 test_metrics["accuracy"]))
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(model, opt.output + os.sep + "trained_model")
        if opt.optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr


if __name__ == "__main__":
    opt = get_args()
    train(opt)
