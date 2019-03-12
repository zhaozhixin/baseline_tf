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
This module to train a so labeling model
"""

import json
import os
import time
import sys
import argparse
import configparser

import tensorflow as tf
import six

import spo_data_reader
import spo_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lib")))
import conf_lib

def train(conf_dict, data_reader, use_cuda=False):
    """
    Training of so labeling model
    """

    # input layer
    word = tf.placeholder(tf.int32, [None, conf_dict['max_len']])
    postag = tf.placeholder(tf.int32, [None, conf_dict['max_len']])
    seq_len = tf.placeholder(tf.int32, [None])
    p_word = tf.placeholder(tf.int32, [None, conf_dict['max_len']])

    # label
    label_ids = tf.placeholder(tf.int32, [None, conf_dict['max_len']])

    # embedding + lstm
    feature_out = spo_model.build_graph(data_reader, conf_dict, word, postag, p_word, seq_len)

    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=feature_out,
                                                                tag_indices=label_ids,
                                                                sequence_lengths=seq_len)
    cost = -tf.reduce_mean(log_likelihood)
    loss = tf.reduce_sum(cost)
    basic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_step = basic_optimizer.minimize(loss)


    def train_loop():
        """start train"""
        init = tf.global_variables_initializer()
        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(init)
        start_time = time.time()
        batch_id = 0
        for pass_id in six.moves.xrange(conf_dict['pass_num']):
            pass_start_time = time.time()
            cost_sum, cost_counter = 0, 0
            for data in data_reader.batch_yield(data_reader.train_data, conf_dict):
                feed_dict = {word: data[0], postag: data[1], seq_len: data[2], p_word: data[4], label_ids: data[3]}
                _loss, _ = sess.run([loss, train_step], feed_dict=feed_dict)

                cost_sum += _loss
                cost_counter += 1
                if batch_id % 10 == 0 and batch_id != 0:
                    print("batch %d finished, second per batch: %02f, loss: %02f" % (
                        batch_id, (time.time() - start_time) / batch_id, _loss), file=sys.stderr)
                    #save_tf_model(sess, pass_id)

                # cost expected, training over
                if float(_loss) < 0.01:
                    pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
                    print("%d pass end, cost time: %02f, avg_cost: %f" % (
                        pass_id, time.time() - pass_start_time, pass_avg_cost), file=sys.stderr)
                    save_tf_model(sess, pass_id)
                    return
                batch_id = batch_id + 1

            # save the model once each pass ends
            pass_avg_cost = cost_sum / cost_counter if cost_counter > 0 else 0.0
            print("%d pass end, cost time: %02f, avg_cost: %f" % (
                pass_id, time.time() - pass_start_time, pass_avg_cost), file=sys.stderr)

            save_tf_model(sess, pass_id)

    train_loop()


def save_tf_model(sess, epoch):
    saver = tf.train.Saver()
    saver.save(sess, "model/spo_model/model.ckpt", global_step=epoch)



def main(conf_dict, use_cuda=False):
    """Train main function"""
    data_generator = spo_data_reader.DataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['so_label_dict_path'],
        p_eng_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['spo_train_data_path'],
        test_data_list_path=conf_dict['spo_test_data_path'])

    data_generator.load_train_data()
    train(conf_dict, data_generator, use_cuda=use_cuda)


if __name__ == '__main__':
    # Load the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_path", type=str,
        help="conf_file_path_for_model. (default: %(default)s)",
        required=True)
    args = parser.parse_args()
    conf_dict = conf_lib.load_conf(args.conf_path)
    use_gpu = True if conf_dict.get('use_gpu', 'False') == 'True' else False
    main(conf_dict, use_cuda=use_gpu)
