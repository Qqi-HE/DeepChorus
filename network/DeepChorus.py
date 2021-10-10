import os
import tensorflow as tf
import tensorflow.keras.backend as K
from .utils import *
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, Conv1D, BatchNormalization, AveragePooling2D,
    Add, Reshape, Flatten, Concatenate, Reshape, Concatenate
)


def mix_module(x_in, channel, out_branch):
    # resample
    x_out = []
    in_branch = len(x_in)
    for i in range(in_branch):
        for j in range(out_branch):
            if j == i:
                x_tmp = Conv1D(channel, 3, strides=1, padding='same')(x_in[i])
            elif j > i:  # downsample
                x_tmp = Conv1D(channel, 3, strides=3, padding='same')(x_in[i])
                # print("down samp.")
                # print(K.int_shape(x_tmp))
                for k in range(j - i - 1):
                    x_tmp = Conv1D(channel, 3, strides=3, padding='same')(x_tmp)
                    # print("down samp.")
                    # print(K.int_shape(x_tmp))

            elif j < i:  # upsample
                # print("before up samp.")
                # print(K.int_shape(x_in[i]))
                x_tmp = Conv1DTranspose(channel, 3, strides=3, padding='same')(x_in[i])
                # print("up samp.")
                # print(K.int_shape(x_tmp))
                for k in range(i - j - 1):
                    x_tmp = Conv1DTranspose(channel, 3, strides=3, padding='same')(x_tmp)
                    # print("up samp.")
                    # print(K.int_shape(x_tmp))
            if i == 0:
                x_out.append(x_tmp)
            else:
                x_out[j] = Concatenate(axis=-1)([x_out[j], x_tmp])
    # convolution
    for i in range(out_branch):
        x_out[i] = Conv1D(channel, 3, padding='same', activation='elu')(x_out[i])
        x_out[i] = BatchNormalization()(x_out[i])
        x_out[i] = Conv1D(channel, 3, padding='same', activation='elu')(x_out[i])
        x_out[i] = BatchNormalization()(x_out[i])

    return x_out


def self_att(x, head_num=1):
    res_branch = MultiHeadAttention(head_num=head_num, activation='elu')(x)
    x = Add()([res_branch, x])
    x = BatchNormalization()(x)
    return x


def create_model(input_shape=(128, None, 1), chunk_size=None):
    input = Input(shape=input_shape)
    x = input

    # preprocess
    c_list = [64, 128, 256, 256]
    p_list = [(4, 1), (4, 1), (4, 1), (2, 1)]
    for i in range(4):
        x = Conv2D(c_list[i], (3, 3), padding='same', activation='elu')(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(p_list[i])(x)
    x = Reshape((-1, 256))(x)

    # HR-Net
    x_s = [x]
    x_s = mix_module(x_s, 128, 2)
    # print("mix 2: ")
    x_s = mix_module(x_s, 128, 3)
    x_s = mix_module(x_s, 128, 3)
    # print("mix 3: ")
    x_s = mix_module(x_s, 128, 2)
    x_s = mix_module(x_s, 128, 1)

    # classifier
    x = x_s[0]

    # SA
    x = self_att(x, head_num=1)
    x = Conv1D(128, 3, padding='same', activation='elu')(x)
    x = self_att(x, head_num=1)
    x = Conv1D(128, 3, padding='same', activation='elu')(x)
    x = self_att(x, head_num=1)

    x = Conv1D(128, 3, padding='same', activation='elu')(x)
    x = Conv1D(64, chunk_size, strides=chunk_size, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv1D(32, 3, padding='same', activation='elu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1, 3, padding='same', activation='sigmoid')(x)
    x = Flatten()(x)

    return Model(inputs=input, outputs=x)
