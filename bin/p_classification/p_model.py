# -*- coding: utf-8 -*-
########################################################
# Copyright (c) 2019, Baidu Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
########################################################
"""
This module to define a neural network
"""

import json
import os
import sys
import argparse
import configparser
import numpy as np
import tensorflow as tf

def build_graph(data_reader, conf_dict, word, postag, seq_len):
    # Placeholders
    hidden_dim = conf_dict['hidden_dim']

    word_emb_init = random_embedding(data_reader.get_dict_size('wordemb_dict'), conf_dict['word_dim'])
    _word_embedding = tf.Variable(word_emb_init,
                                   dtype=tf.float32,
                                   name="word_embedding")

    postag_emb_init = random_embedding(data_reader.get_dict_size('postag_dict'), conf_dict['postag_dim'])
    _postag_embedding = tf.Variable(postag_emb_init,
                                 dtype=tf.float32,
                                 name="postag_embedding")


    word_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(_word_embedding, np.float32), word)
    postag_embedding = tf.nn.embedding_lookup(tf.convert_to_tensor(_postag_embedding, np.float32), postag)

    with tf.variable_scope("word_fc"):
        word_inputs = full_connect(tf.reshape(word_embedding,[-1,conf_dict['word_dim']]), hidden_dim)

    with tf.variable_scope("postag_fc"):
        postag_inputs = full_connect(tf.reshape(postag_embedding,[-1,conf_dict['postag_dim']]), hidden_dim)

    word_rnn_inputs = word_inputs + postag_inputs
    word_rnn_inputs_formatted = tf.reshape(word_rnn_inputs, [-1, conf_dict['max_len'], hidden_dim])

    with tf.variable_scope("rnn1"):
        word_rnn_outputs = sequence(word_rnn_inputs_formatted, hidden_dim, seq_len)
        word_rnn_outputs = tf.concat(word_rnn_outputs, 2)

    with tf.variable_scope("rnn2"):
        word_rnn_outputs = sequence(word_rnn_outputs, hidden_dim, seq_len)
        word_rnn_outputs = tf.concat(word_rnn_outputs, 2)

    with tf.variable_scope("rnn3"):
        word_rnn_outputs = sequence(word_rnn_outputs, hidden_dim, seq_len)
        word_rnn_outputs = tf.concat(word_rnn_outputs, 2)

    with tf.variable_scope("rnn4"):
        word_rnn_outputs = sequence(word_rnn_outputs, hidden_dim, seq_len)
        word_rnn_outputs = tf.concat(word_rnn_outputs, 2)

    max_pool = tf.reduce_max(word_rnn_outputs, axis=1)

    with tf.variable_scope("out_weights"):
        weights_out = tf.get_variable(name="output_w", dtype=tf.float32, shape=[hidden_dim * 2, conf_dict['class_dim']])
        biases_out = tf.get_variable(name="output_bias", dtype=tf.float32, shape=[conf_dict['class_dim']])

    dense = tf.matmul(max_pool, weights_out) + biases_out

    return dense

def random_embedding(dict_size, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (dict_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def sequence(rnn_inputs, hidden_size, seq_lens):
    cell_fw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print(('build fw cell: ' + str(cell_fw)))
    cell_bw = tf.nn.rnn_cell.GRUCell(hidden_size)
    print(('build bw cell: ' + str(cell_bw)))
    rnn_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                               cell_bw,
                                                               inputs=rnn_inputs,
                                                               sequence_length=seq_lens,
                                                               dtype=tf.float32
                                                               )
    print(('rnn outputs: ' + str(rnn_outputs)))
    print(('final state: ' + str(final_state)))

    return rnn_outputs

def full_connect(inputs, output_dim):
    weight_shape = [inputs.get_shape().as_list()[-1], output_dim]
    weight = tf.get_variable("weight", weight_shape, initializer=tf.contrib.layers.xavier_initializer(inputs.dtype))
    bias = tf.get_variable("bias", [output_dim], initializer=tf.zeros_initializer(inputs.dtype))
    values = tf.matmul(inputs, weight) + bias
    return values