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
This module to infer with a trained so model
"""

import json
import os
import sys
import argparse
import configparser
import codecs
from tensorflow.python import debug as tf_debug

import numpy as np
import tensorflow as tf

import spo_data_reader
import spo_model

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
    p_word = tf.placeholder(tf.int32, [None, conf_dict['max_len']])

    # label
    label_ids = tf.placeholder(tf.int32, [None, conf_dict['max_len']])

    # embedding + lstm
    feature_out = spo_model.build_graph(data_reader, conf_dict, word, postag, p_word, seq_len)

    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=feature_out,
                                                                          tag_indices=label_ids,
                                                                          sequence_lengths=seq_len)

    sess = restore_tf_model(model_path)
    text_spo_dic = {}  #final triples

    # batch
    batch_id = 0
    for data in data_reader.batch_yield(data_reader.test_data, conf_dict):

        feed_dict = {word: data[0], postag: data[1], seq_len: data[2], p_word: data[4], label_ids: data[3]}

        logits, _transition_params = sess.run([feature_out, transition_params], feed_dict=feed_dict)

        label_list = []
        for logit, _seq_len in zip(logits, data[2]):
            viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:_seq_len], _transition_params)
            label_list.append(viterbi_seq)

        input_data = data[5]

        tag_split_idx = label_list
        label_tag_scores = np.array(logits)
        # sentence
        print('batch_id=', batch_id, file=sys.stderr)
        for sent_idx, tag_list in enumerate(label_list):
            input_sent = input_data[sent_idx].split('\t')[0]
            input_p = input_data[sent_idx].split('\t')[1]
            tag_list = [data_reader._reverse_dict['so_label_dict'].get(t, 0) for t in tag_list]
            predicted_s_list, predicted_o_list = refine_predict_seq(input_sent, tag_list)
            #tag_list_str = json.dumps(tag_list, ensure_ascii=False)
            if len(predicted_s_list) == 0 or len(predicted_o_list) == 0:
                continue
            else:
                text = json.loads(input_sent)["text"]
                predicted_s_list = list(set(predicted_s_list))
                predicted_o_list = list(set(predicted_o_list))
                for predicted_s in predicted_s_list:
                    for predicted_o in predicted_o_list:
                        if text not in text_spo_dic:
                            text_spo_dic[text] = set()
                        text_spo_dic[text].add((predicted_s, input_p, predicted_o))

        batch_id += 1
        break

    output(text_spo_dic, result_writer)


def refine_predict_seq(input_sent, tag_list):
    """
    Generate s-o list based on the annotation results 
    predicted by the model
    """
    sent_info = json.loads(input_sent)
    word_seq = [item["word"] for item in sent_info["postag"]]
    s_list, o_list= [], []
    token_idx = 0
    while token_idx < len(tag_list):
        if tag_list[token_idx] == 'O':
            token_idx += 1
        elif tag_list[token_idx].endswith('SUB') and tag_list[token_idx].startswith('B'):
            cur_s = word_seq[token_idx]
            token_idx += 1
            while token_idx < len(tag_list) and tag_list[token_idx].endswith('SUB'):
                cur_s += word_seq[token_idx]
                token_idx += 1
            s_list.append(cur_s)
        elif tag_list[token_idx].endswith('OBJ') and tag_list[token_idx].startswith('B'):
            cur_o = word_seq[token_idx]
            token_idx += 1
            while token_idx < len(tag_list) and tag_list[token_idx].endswith('OBJ'):
                cur_o += word_seq[token_idx]
                token_idx += 1
            o_list.append(cur_o)
        else:
            token_idx += 1
    return s_list, o_list


def get_schemas(schema_file):
    """"Read the original schema file"""
    schema_dic = {}
    with codecs.open(schema_file, 'r', 'utf-8') as fr:
        for line in fr:
            dic = json.loads(line.strip())
            predicate = dic["predicate"]
            subject_type = dic["subject_type"]
            object_type = dic["object_type"]
            schema_dic[predicate] = (subject_type, object_type)
    return schema_dic


def output(text_spo_dic, result_writer):
    """
    Output final SPO triples
    """
    schema_dic = {}
    schema_dic = get_schemas('./data/all_50_schemas')
    for text in text_spo_dic:
        text_dic = {"text": text}
        text_dic["spo_list"] = []
        for spo in text_spo_dic[text]:
            dic = {"subject": spo[0], "predicate": spo[1], \
                    "object": spo[2], "subject_type": schema_dic[spo[1]][0], \
                    "object_type": schema_dic[spo[1]][1]}
            text_dic["spo_list"].append(dic)
        result_writer.write(json.dumps(text_dic, ensure_ascii=False))
        result_writer.write('\n')


def main(conf_dict, model_path, predict_data_path, predict_result_path, \
        use_cuda=False):
    """Predict main function"""
    data_generator = spo_data_reader.DataReader(
        wordemb_dict_path=conf_dict['word_idx_path'],
        postag_dict_path=conf_dict['postag_dict_path'],
        label_dict_path=conf_dict['so_label_dict_path'],
        p_eng_dict_path=conf_dict['label_dict_path'],
        train_data_list_path=conf_dict['spo_train_data_path'],
        test_data_list_path=conf_dict['spo_test_data_path'])

    data_generator.load_test_data()
    predict_infer(conf_dict, data_generator, predict_data_path, \
            predict_result_path, model_path)

def restore_tf_model(epoch):
    sess = tf.Session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    saver = tf.train.Saver()
    saver.restore(sess, "model/spo_model/model.ckpt-" + str(epoch))
    return sess


if __name__ == '__main__':
    # Load the configuration file
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
