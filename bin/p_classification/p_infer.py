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
This module to infer with a p classification model
"""

import json
import os
import sys
import argparse
import configparser
import math
import tensorflow as tf
import p_model

import numpy as np


import p_data_reader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib


def predict_infer(conf_dict, data_reader, predict_data_path, \
        predict_result_path, model_path):
    """
    Predict with trained models 
    """
    if len(predict_result_path) > 0:
        result_writer = open(predict_result_path, 'w', encoding='utf-8')
    else:
        result_writer = sys.stdout

    # input layer
    word = tf.placeholder(tf.int32, [None, conf_dict['max_len']])
    postag = tf.placeholder(tf.int32, [None, conf_dict['max_len']])
    seq_len = tf.placeholder(tf.int32, [None])

    # label
    label_ids = tf.placeholder(tf.float32, [None, conf_dict['class_dim']])

    # NN: embedding + lstm + pooling
    feature_out = p_model.build_graph(data_reader, conf_dict, word, postag, seq_len)

    sess = restore_tf_model(model_path)
    batch_id = 0

    for data in data_reader.batch_yield(data_reader.test_data, conf_dict):
        # print(data)
        feed_dict = {word: data[0], postag: data[1], seq_len: data[2], label_ids: data[3]}
        label_scores = sess.run(feature_out, feed_dict=feed_dict)

        input_data = data[4]
        infer_a_batch(label_scores, input_data, result_writer, data_reader)

        batch_id += 1


def infer_a_batch(label_scores, input_data, result_writer, data_reader):
    """Infer the results of a batch"""
    #print(label_scores,input_data)
    for sent_idx, label in enumerate(label_scores):

        p_label = []
        label = list(map(float, label))
        for p_idx, p_score in enumerate(label):
            if sigmoid(p_score) > 0.5:
                p_label.append(data_reader.get_label_output(p_idx))
        for p in p_label:
            output_fields = [input_data[sent_idx], p]
            result_writer.write('\t'.join(output_fields))
            result_writer.write('\n')


def sigmoid(x):
    """sigmode function"""
    return math.exp(x) / (1 + math.exp(x))

def restore_tf_model(epoch):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "model/p_model/model.ckpt-" + str(epoch))
    return sess


def main(conf_dict, model_path, predict_data_path, 
            predict_result_path, use_cuda=False):
    """Predict main function"""
    data_generator = p_data_reader.RcDataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['train_data_path'],
        test_data_list_path=conf_dict['test_data_path'])

    data_generator.load_test_data()
    
    predict_infer(conf_dict, data_generator, predict_data_path, \
            predict_result_path, model_path)


if __name__ == '__main__':
    # Load configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str,
            help="conf_file_path_for_model. (default: %(default)s)",
            required=True)
    parser.add_argument("--model_path", type=str,
            help="model_path", required=True)
    parser.add_argument("--predict_file", type=str,
            help="the_file_to_be_predicted", required=True)
    parser.add_argument("--result_file", type=str,
            default='', help="the_file_of_predicted_results")
    args = parser.parse_args()
    conf_dict = conf_lib.load_conf(args.conf_path)
    model_path = args.model_path
    predict_data_path = args.predict_file
    predict_result_path = args.result_file
    for input_path in [predict_data_path]:
        if not os.path.exists(input_path):
            raise ValueError("%s not found." % (input_path))
    main(conf_dict, model_path, predict_data_path, predict_result_path)
