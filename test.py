import os
import argparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import joblib
import importlib
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from constant import *
from loader import load_labels, load_features
from evaluator import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('-n', "--network")
parser.add_argument('-m', "--mark")
args = parser.parse_args()

# Loading Network
network_module = importlib.import_module('network.' + args.network)
create_model = network_module.create_model

# Loading Model
model_file = 'model/' + args.network + '_' + args.mark + '.h5'
model = create_model(input_shape=SHAPE, chunk_size=CHUNK_SIZE)
model.compile(loss='binary_crossentropy', optimizer=(Adam(lr=LR)),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.Recall()])
model.load_weights(model_file)
model.summary()

# Loading Data
features = load_features(test_feature_files)
print('Loaded features for {} files from {}.'.format(len(features), test_feature_files))
labels = load_labels(test_annotation_files)
print('Loaded labels for {} files from {}.'.format(len(labels), test_annotation_files))


# Testing Result
print('Testing...')
predictions_dict, target_dict = get_result_dict(model, features, labels)

joblib.dump(target_dict, 'result/ground_truth.joblib')
joblib.dump(predictions_dict, 'result/result.joblib')

R, P, F, AUC = test_dict_result(predictions_dict, target_dict)
print("                                            ", end='\r')
print("R: {:.4f}\nP: {:.4f}\nF: {:.4f}\nAUC: {:.4f}".format(R, P, F, AUC))
