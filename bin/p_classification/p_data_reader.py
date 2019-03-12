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
This module to define a class for p classfication data reader
"""

import json
import os
import codecs
import sys
import random


class RcDataReader(object):
    """
    class for p classfication data reader
    """
    def __init__(self,
                wordemb_dict_path,
                postag_dict_path,
                label_dict_path,
                train_data_list_path='',
                test_data_list_path=''):
        self._wordemb_dict_path = wordemb_dict_path
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self.train_data_list_path = train_data_list_path
        self.test_data_list_path = test_data_list_path
        self._p_map_eng_dict = {}
        # load dictionary
        self._dict_path_dict = {'wordemb_dict': self._wordemb_dict_path,
                                'postag_dict': self._postag_dict_path,
                                'label_dict': self._label_dict_path}
        # check if the file exists
        for input_dict in [wordemb_dict_path, postag_dict_path, \
                label_dict_path, train_data_list_path, test_data_list_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % (input_dict))
                return

        self._feature_dict = {}
        self._feature_dict['postag_dict'] = \
                self._load_dict_from_file(self._dict_path_dict['postag_dict'])
        self._feature_dict['wordemb_dict'] = \
                self._load_dict_from_file(self._dict_path_dict['wordemb_dict'])
        self._feature_dict['label_dict'] = \
                self._load_label_dict(self._dict_path_dict['label_dict'])
        self._reverse_dict = {name: self._get_reverse_dict(name) for name in
                              list(self._dict_path_dict.keys())}
        self._reverse_dict['eng_map_p_dict'] = self._reverse_p_eng(self._p_map_eng_dict)
        self._UNK_IDX = 0
        self.train_data = []
        self.test_data = []

    def load_train_data(self):
        self.train_data = self.read_corpus(self.train_data_list_path)

    def load_test_data(self):
        self.test_data = self.read_corpus(self.test_data_list_path)

    def _load_label_dict(self, dict_name):
        """load label dict from file"""
        label_dict = {}
        with codecs.open(dict_name, 'r', 'utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                label_dict[p_eng] = idx
                self._p_map_eng_dict[p] = p_eng
        return label_dict

    def _load_dict_from_file(self, dict_name, bias=0):
        """
        Load vocabulary from file.
        """
        dict_result = {}
        with codecs.open(dict_name, 'r', 'utf-8') as f_dict:
            for idx, line in enumerate(f_dict):
                line = line.strip()
                dict_result[line] = idx + bias
        return dict_result

    def _cal_mark_slot(self, spo_list, sentence):
        """
        Calculate the value of the label 
        """
        mark_list = [0] * len(self._feature_dict['label_dict'])
        for spo in spo_list:
            predicate = spo['predicate']
            p_idx = self._feature_dict['label_dict'][self._p_map_eng_dict[predicate]]
            mark_list[p_idx] = 1
        return mark_list

    def _is_valid_input_data(self, input_data):
        """is the input data valid"""
        try:
            dic = json.loads(input_data)
        except:
            return False
        if "text" not in dic or "postag" not in dic or \
                type(dic["postag"]) is not list:
            return False
        for item in dic['postag']:
            if "word" not in item or "pos" not in item:
                return False
        return True
    
    def _get_feed_iterator(self, line, need_input=False, need_label=True):
        # verify that the input format of each line meets the format
        if not self._is_valid_input_data(line):
            print('Format is error', file=sys.stderr)
            return None
        dic = json.loads(line)
        sentence = dic['text']
        sentence_term_list = [item['word'] for item in dic['postag']]
        sentence_pos_list = [item['pos'] for item in dic['postag']]
        sentence_emb_slot = [self._feature_dict['wordemb_dict'].get(w, self._UNK_IDX) \
                for w in sentence_term_list]
        sentence_pos_slot = [self._feature_dict['postag_dict'].get(pos, self._UNK_IDX) \
                for pos in sentence_pos_list]
        if 'spo_list' not in dic:
            label_slot = [0] * len(self._feature_dict['label_dict'])
        else:
            label_slot = self._cal_mark_slot(dic['spo_list'], sentence)
        # verify that the feature is valid
        if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 \
                or len(label_slot) == 0:
            return None
        feature_slot = [sentence_emb_slot, sentence_pos_slot]
        input_fields = json.dumps(dic, ensure_ascii=False)
        output_slot = feature_slot
        if need_input:
            output_slot = [input_fields] + output_slot
        if need_label:
            output_slot = output_slot + [label_slot]
        return output_slot

    def get_dict(self, dict_name):
        """Return dict"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return self._feature_dict[dict_name]

    def get_all_dict_name(self):
        """Get name of all dict"""
        return list(self._feature_dict.keys())

    def get_dict_size(self, dict_name):
        """Return dict length"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return len(self._feature_dict[dict_name])

    def _get_reverse_dict(self, dict_name):
        dict_reverse = {}
        for key, value in self._feature_dict[dict_name].items():
            dict_reverse[value] = key
        return dict_reverse
    
    def _reverse_p_eng(self, dic):
        dict_reverse = {}
        for key, value in dic.items():
            dict_reverse[value] = key
        return dict_reverse

    def get_label_output(self, label_idx):
        """Output final label, used during predict and test"""
        dict_name = 'label_dict'
        if len(self._reverse_dict[dict_name]) == 0:
            self._get_reverse_dict(dict_name)
        p_eng = self._reverse_dict[dict_name][label_idx]
        return self._reverse_dict['eng_map_p_dict'][p_eng]

    def read_corpus(self, corpus_path):
        """
        read corpus and return the list of samples
        :param corpus_path:
        :return: data
        """
        data = []
        with open(corpus_path, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                if not self._is_valid_input_data(line):
                    print('Format is error', file=sys.stderr)
                    return None
                dic = json.loads(line)
                sentence = dic['text']
                sentence_term_list = [item['word'] for item in dic['postag']]
                sentence_pos_list = [item['pos'] for item in dic['postag']]
                sentence_emb_slot = [self._feature_dict['wordemb_dict'].get(w, self._UNK_IDX) \
                                     for w in sentence_term_list]
                sentence_pos_slot = [self._feature_dict['postag_dict'].get(pos, self._UNK_IDX) \
                                     for pos in sentence_pos_list]
                if 'spo_list' not in dic:
                    label_slot = [0] * len(self._feature_dict['label_dict'])
                else:
                    label_slot = self._cal_mark_slot(dic['spo_list'], sentence)
                # verify that the feature is valid
                if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 \
                        or len(label_slot) == 0:
                    continue
                feature_slot = [sentence_emb_slot, sentence_pos_slot]
                input_fields = json.dumps(dic, ensure_ascii=False)
                output_slot = feature_slot
                output_slot = [input_fields] + output_slot
                output_slot = output_slot + [label_slot]
                data.append(output_slot)
        return data

    def batch_yield(self, data, conf_dict, shuffle=False):
        """

        :param data:
        :param batch_size:
        :param vocab:
        :param tag2label:
        :param shuffle:
        :return:
        """
        if shuffle:
            random.shuffle(data)

        word = []
        postag = []
        seq_len = []
        label_ids = []
        input_raw_list = []

        for sent in data:
            input_raw, w_id, p_id, l_id = sent
            w_id, p_id = w_id[0:conf_dict['max_len']], p_id[0:conf_dict['max_len']]
            s_len = len(w_id)

            w_id = w_id + [0] * (conf_dict['max_len'] - len(w_id))
            p_id = p_id + [0] * (conf_dict['max_len'] - len(p_id))

            word.append(w_id)
            postag.append(p_id)
            seq_len.append(s_len)
            label_ids.append(l_id)
            input_raw_list.append(input_raw)

            if len(word) == conf_dict['batch_size']:
                yield word, postag, seq_len, label_ids, input_raw_list
                word = []
                postag = []
                seq_len = []
                label_ids = []
                input_raw_list = []

        if len(word) != 0:
            yield word, postag, seq_len, label_ids, input_raw_list
