import random
import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform
from constant import *


def create_seq_label(feature_crop, chorus_list, begin):
    begin = round(begin / FRAME_PER_SEC)

    crop_sec = round(len(feature_crop[1]) / FRAME_PER_SEC)
    seq_label = np.zeros(crop_sec)
    if crop_sec < N_CHUNK:
        seq_label = np.zeros(N_CHUNK)
    end = begin + N_CHUNK

    song_sec = round(chorus_list[-1][-1] / SEC_PER_CHUNK)
    total_seq_label = np.zeros(song_sec)
    for chorus_seg in chorus_list:

        sec_chorus_seg = [round(chorus_seg[0] / SEC_PER_CHUNK),
                            round(chorus_seg[1] / SEC_PER_CHUNK)]

        for i in range(sec_chorus_seg[0], sec_chorus_seg[1]):
            total_seq_label[i] = 1

    for i in range(begin, end):
        if i < len(total_seq_label) and i - begin < len(seq_label):
            seq_label[i - begin] = total_seq_label[i]

    return seq_label


def scale_data(feature, label, augmentation):
    label_scaled = []
    if augmentation:
        scale_factor = random.choice([x / 100.0 for x in range(80, 124, 4)])
        if scale_factor != 1:
            feature_scaled = apply_affine_transform(feature, zy=scale_factor, fill_mode='nearest').astype(np.float32)
            for row in range(len(label)):
                label_scaled.append(
                    [round(label[row][0] * scale_factor), round(label[row][1] * scale_factor)])
        else:
            feature_scaled = feature
            label_scaled = label
    else:
        feature_scaled = feature
        label_scaled = label
    return feature_scaled, label_scaled


def pitch_shift_spectrogram(feature):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    scale_factor = random.choice([x / 100.0 for x in range(80, 124, 4)])
    if scale_factor != 1:
        feature_scaled = apply_affine_transform(feature, zx=scale_factor, fill_mode='reflect').astype(np.float32)
    else:
        feature_scaled = feature

    return feature_scaled


def create_data_generator(feature_dict, labels_dict):

    label_idx = 0
    keys = list(labels_dict.keys())
    length = len(keys)
    random.shuffle(keys)
    offset = -1

    while True:
        if label_idx + BATCH_SIZE > length:
            label_idx = 0
            random.shuffle(keys)
        X, y = [], []
        key_list = []
        offset_list = []
        for i in range(label_idx, label_idx + BATCH_SIZE):  # 打包
            feature = feature_dict[keys[i]]
            label_list = labels_dict[keys[i]]
            feature, label_list = scale_data(feature, label_list, augmentation=True)
            feature = pitch_shift_spectrogram(feature)

            if feature.shape[1] < N_FRAME:
                remain = N_FRAME - feature.shape[1]
                remain_np = np.zeros([128, remain, 1])
                feature_crop = np.concatenate((feature, remain_np), axis=1)
            else:
                offset = random.randint(0, feature.shape[1] - N_FRAME)
                feature_crop = feature[:, offset: offset + N_FRAME, :]

            label = create_seq_label(feature_crop, label_list, offset)

            X.append(feature_crop)
            y.append(label)
            key_list.append(keys[i])
            offset_list.append((round(offset / FRAME_PER_SEC)))
        yield key_list, np.stack(X, axis=0), np.stack(y, axis=0), offset_list
        label_idx += BATCH_SIZE


def transformer_generator(feature_dict, labels_dict):

    label_idx = 0
    keys = list(labels_dict.keys())
    length = len(keys)
    random.shuffle(keys)
    offset = -1

    while True:
        if label_idx + BATCH_SIZE > length:
            label_idx = 0
            random.shuffle(keys)
        X, y = [], []
        key_list = []
        offset_list = []
        for i in range(label_idx, label_idx + BATCH_SIZE):
            feature = feature_dict[keys[i]]
            label_list = labels_dict[keys[i]]

            feature, label_list = scale_data(feature, label_list, augmentation=True)
            feature = pitch_shift_spectrogram(feature)
            if feature.shape[1] < N_FRAME:
                remain = N_FRAME - feature.shape[1]
                remain_np = np.zeros([128, remain, 1])
                feature_crop = np.concatenate((feature, remain_np), axis=1)
            else:
                offset = random.randint(0, feature.shape[1] - N_FRAME)
                feature_crop = feature[:, offset: offset + N_FRAME, :]
            print(keys[i])
            print(label_list)
            label = create_seq_label(feature_crop, label_list, offset)

            X.append(feature_crop)
            y.append(label)
            key_list.append(keys[i])
            offset_list.append((round(offset / FRAME_PER_SEC)))

        yield key_list, np.stack(X, axis=0), np.stack(y, axis=0), offset_list
        label_idx += BATCH_SIZE
