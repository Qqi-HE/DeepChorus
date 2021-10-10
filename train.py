import os
import tensorflow as tf
import argparse
import importlib
import shutil
import time
import numpy as np
import random
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from constant import *
from loader import load_labels, load_features
from generator import create_data_generator
from evaluator import get_result_dict, test_dict_result

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


parser = argparse.ArgumentParser()
parser.add_argument('-n', "--network")
parser.add_argument('-m', "--mark")
args = parser.parse_args()

# Loading Network
network_module = importlib.import_module('network.' + args.network)
create_model = network_module.create_model

# Loading Model
checkpoint_model_file = 'model/' + args.network + '_' + args.mark + '.h5'

# Logging
log_file_name = checkpoint_model_file.replace('model/', 'log/').replace('.h5', '.log')
log_file = open(log_file_name, 'wb')


def log(message):
    message_bytes = '{}\n'.format(message).encode(encoding='utf-8')
    log_file.write(message_bytes)
    print(message)


def valid_songs(model, songs_num, features_dict, labels_dict):
    v_features = {}
    v_labels = {}
    key_list = list(labels_dict.keys())
    random.shuffle(key_list)
    for i in range(songs_num):
        key = key_list[i]
        v_features[key] = features_dict[key]
        v_labels[key] = labels_dict[key]
    res, gt = get_result_dict(model, v_features, v_labels, show_msg=False)
    v_r, v_p, v_f, v_auc = test_dict_result(res, gt)
    return v_r, v_p, v_f, v_auc


##--- Loading Labels ---##
train_labels = load_labels(train_annotation_file)
valid_labels = load_labels(valid_annotation_file)
log('Loaded {} training labels from {}.'.format(len(train_labels), train_annotation_file))
log('Loaded {} validation labels from {}.'.format(len(valid_labels), valid_annotation_file))

##--- Loading Data ---##
print('loading features...')
features = load_features(feature_files)
log('Loaded features for {} files.'.format(len(features)))
valid_features = load_features(valid_feature)
log('Loaded valid features for {} files.'.format(len(valid_features)))

##--- Data Generator ---##
log('\nCreating generators...')
train_generator = create_data_generator(features, train_labels)
valid_generator = create_data_generator(valid_features, valid_labels)

##--- Network ---##
log('\nCreating model...')
model = create_model(input_shape=SHAPE, chunk_size=CHUNK_SIZE)
model.compile(loss='mean_squared_error', optimizer=(Adam(lr=LR)),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), tf.keras.metrics.Recall()])
# model.summary()


##--- Training ---##
log('\nTaining...')
log('params={}'.format(model.count_params()))

i_epoch, i_iter = 1, 1
best_acc, best_epoch, best_R, best_epoch_R = 0, 0, 0, 0
mean_loss, mean_acc, mean_recall = 0, 0, 0
steps_per_epoch = 50

valid_acc_list = []
v_para = 10

time_start = time.time()
while i_epoch < EPOCHS:
    _, X, y, _ = next(train_generator)
    i_iter = i_iter % steps_per_epoch + 1
    loss, acc, recall = model.train_on_batch(X, y)
    mean_loss += loss / steps_per_epoch
    mean_acc += acc / steps_per_epoch
    mean_recall += recall / steps_per_epoch

    print('Epoch:[{} / {}] - Round:[{} / {}] - loss: {:.4f} - accuracy: {:.4f} - recall: {:.4f}'.format(
        i_epoch, EPOCHS, i_iter, steps_per_epoch, loss, acc, recall), end='\r')
    if i_iter == 1:
        time_train = time.time() - time_start
        valid_mean_acc, valid_mean_recall = 0, 0, 0
        fig_vis_num = 0

        v_r, v_p, v_f, v_auc = valid_songs(model, v_para, valid_features, valid_labels)
        log('Epoch:[{} / {}] - Time: {:.1f}s - loss: {:.4f} - accuracy: {:.4f} - recall: {:.4f} - '
            'f1: {:.4f} - auc: {:.4f} - mean_recall: {:.4f}'.format(
            i_epoch, EPOCHS, time_train, mean_loss, mean_acc, mean_recall,
            v_f, v_auc, v_r))
        valid_acc_list.append(v_f)

        # check point
        if v_f >= best_acc:
            best_acc = v_f
            best_epoch = i_epoch
            model.save_weights(checkpoint_model_file)

        log('best_auc: {:.4f} - best_epoch: {}'.format(best_acc, best_epoch))


        # early stopping
        if i_epoch - best_epoch > PATIENCE:
            log('Early Stopping.')
            break

        i_epoch += 1
        mean_acc, mean_loss, mean_recall = 0, 0, 0
        time_start = time.time()

log_file.close()
