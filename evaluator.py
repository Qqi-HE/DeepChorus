import copy
from itertools import count
import librosa
import numpy as np
import os
from tensorflow.keras import Input, Model
from constant import *
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import joblib
import tensorflow as tf


def get_song_label(chorus_list, feature_length):
    song_label = np.zeros(round(feature_length / FRAME_PER_SEC))
    for seg in chorus_list:
        for i in range(seg[0], seg[1]):
            if i < len(song_label):
                song_label[i] = 1
    return song_label


def result_to_np(src_result):
    result = np.zeros(len(src_result))
    for i in range(len(src_result)):
        result[i] = src_result[i]
    return result


def scale_result(result):
    # print(result)
    new_np = np.zeros(len(result))
    max_score = np.max(result)
    min_score = np.min(result)
    overlap = max_score - min_score
    for i in range(len(result)):
        new_np[i] = (result[i] - min_score) / overlap
    return new_np


def median_filter(result_np, sample):

    begin = side = int(sample // 2)
    for i in range(begin):
        result_np = np.insert(result_np, 0, result_np[2 * i])
        result_np = np.append(result_np, result_np[-(2 * i)])

    filtered_np = copy.deepcopy(result_np)

    for i in range(begin, (len(filtered_np) - side)):
        group_s = i - side
        group_e = i + side + 1
        window = result_np[group_s: group_e]
        # print(len(window))
        r_max = np.max(window)
        # print(r_max)
        r_min = np.min(window)
        # print(r_min)
        mid_value = float((sum(window) - r_min - r_max) / (len(window) - 2))
        filtered_np[i] = mid_value
    for i in range(begin):
        filtered_np = np.delete(filtered_np, 0)
        filtered_np = np.delete(filtered_np, len(filtered_np) - 1)
    return filtered_np


def get_result_dict(model, features_dict, labels_dict):

    @tf.function(experimental_relax_shapes=True)
    def predict(t):
        return model(t)
    counter = 0
    np_labels = {}
    orig_dict = {}

    for key in labels_dict.keys():
        feature = features_dict[key]
        feature_length = len(feature[1])
        remain = feature_length % 9
        remain_np = np.zeros([128, 9 - remain, 1])
        feature_crop = np.concatenate((feature, remain_np), axis=1)

        counter += 1
        print('Predicting ({}/ {}): {}\t'.format(counter, len(labels_dict), key), end='\r')
        
        song_label = get_song_label(labels_dict[key], len(feature_crop[1]))
        feature_crop = np.expand_dims(feature_crop, axis=0)

        result_np = predict(feature_crop)[0]

        remain_crop = abs(len(result_np) - len(song_label))
        if len(result_np) < len(song_label):
            for i in range(remain_crop):
                result_np = np.append(result_np, 0)
        if len(result_np) > len(song_label):
            for i in range(remain_crop):
                song_label = np.append(song_label, 0)

        np_labels[key] = song_label
        orig_dict[key] = copy.deepcopy(result_np)

    return orig_dict, np_labels


def get_rpf(result_dict, label_dict):
    temp_r, temp_p, temp_f = 0, 0, 0
    total = len(result_dict)
    for key in result_dict.keys():
        temp_p += precision_score(label_dict[key], result_dict[key], average='binary')
        temp_r += recall_score(label_dict[key], result_dict[key], average='binary')
        temp_f += f1_score(label_dict[key], result_dict[key], average='binary')

    r = temp_r / total
    p = temp_p / total
    f = temp_f / total
    return r, p, f


def test_dict_result(result_dict, gt_dict):
    res_list = []
    new_dict = {}
    counter = 0
    for key in result_dict.keys():
        result_np = result_dict[key]

        result_np = scale_result(result_np)
        result_np = median_filter(result_np, 9)

        if gt_dict[key].sum() != 0:
            counter += 1
            auc = roc_auc_score(gt_dict[key], result_np)
            print('Testing ({}/ {}): {}\t'.format(counter, len(result_dict), key, auc), end='\r')
            for i in range(len(result_np)):
                if result_np[i] < 0.5:
                    result_np[i] = 0
                else:
                    result_np[i] = 1

            res_list.append(auc)
            new_dict[key] = result_np

    auc = 0

    for i in range(len(new_dict)):
        auc += res_list[i] / len(new_dict)
    r, p, f = get_rpf(new_dict, gt_dict)

    return r, p, f, auc
