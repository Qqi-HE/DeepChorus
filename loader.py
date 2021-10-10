import joblib
from constant import *


def load_labels(label_list):
    labels = {}
    round_labels = {}
    for label_joblib in label_list:
        labels.update(joblib.load(label_joblib))
    for key, label in labels.items():
        round_labels[key] = []
        for row in label:
            round_labels[key] += [[round(float(row[0])), round(float(row[1]))]]
    return round_labels


def load_features(files):
    features = {}
    for feature_file in files:
        features.update(joblib.load(feature_file))
    return features
