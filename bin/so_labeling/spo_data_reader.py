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
This module to define a class for data reader
"""

import json
import random
import os
import re
import codecs
import sys


class DataReader(object):
    """
    class for data reader 
    """
    def __init__(self,
                wordemb_dict_path,
                postag_dict_path,
                label_dict_path,
                p_eng_dict_path,
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
                                'so_label_dict': self._label_dict_path}
        # check if the file exists
        for input_dict in [wordemb_dict_path, postag_dict_path, \
                label_dict_path, train_data_list_path, test_data_list_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % (input_dict))
                return

        self._feature_dict = {}
        self._feature_dict = {name: self._load_dict_from_file(self._dict_path_dict[name]) \
                for name in list(self._dict_path_dict.keys())}
        self._p_map_eng_dict = self._load_p_eng_dict(p_eng_dict_path)
        self._reverse_dict = {name: self._get_reverse_dict(name) for name in
                              list(self._dict_path_dict.keys())}
        self._UNK_IDX = 0
        self.train_data = []
        self.test_data = []

    def load_train_data(self):
        self.train_data = self.read_corpus(self.train_data_list_path)

    def load_test_data(self):
        self.test_data = self.read_corpus(self.test_data_list_path)

    def _load_p_eng_dict(self, dict_name):
        """load label dict from file"""
        p_eng_dict = {}
        with codecs.open(dict_name, 'r', 'utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                p_eng_dict[p] = p_eng
        return p_eng_dict

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

    def _add_item_offset(self, token, sentence):
        """Get the start and end offset of a token in a sentence"""
        s_pattern = re.compile(re.escape(token), re.I)
        token_offset_list = []
        for m in s_pattern.finditer(sentence):
            token_offset_list.append((m.group(), m.start(), m.end()))
        return token_offset_list
    
    def _cal_item_pos(self, target_offset, idx_list):
        """Get the index list where the token is located"""
        target_idx = []
        for target in target_offset:
            start, end = target[1], target[2]
            cur_idx = []
            for i, idx in enumerate(idx_list):
                if idx >= start and idx < end:
                    cur_idx.append(i)
            if len(cur_idx) > 0:
                target_idx.append(cur_idx)
        return target_idx
    
    def _get_token_idx(self, sentence_term_list, sentence):
        """Get the start offset of every token"""
        token_idx_list = []
        start_idx = 0
        for sent_term in sentence_term_list:
            if start_idx >= len(sentence):
                break
            token_idx_list.append(start_idx)
            start_idx += len(sent_term)
        return token_idx_list

    def _cal_mark_slot(self, spo_list, sentence, p, token_idx_list):
        """Calculate the value of the label"""
        mark_list = ['O'] * len(token_idx_list)
        for spo in spo_list:
            predicate = spo['predicate']
            if predicate != p:
                continue
            sub = spo['subject']
            obj = spo['object']
            s_idx_list = self._cal_item_pos(self._add_item_offset(sub, sentence), \
                    token_idx_list)
            o_idx_list = self._cal_item_pos(self._add_item_offset(obj, sentence), \
                    token_idx_list)
            if len(s_idx_list) == 0 or len(o_idx_list) == 0:
                continue
            for s_idx in s_idx_list:
                if len(s_idx) == 1:
                    mark_list[s_idx[0]] = 'B-SUB'
                elif len(s_idx) == 2:
                    mark_list[s_idx[0]] = 'B-SUB'
                    mark_list[s_idx[1]] = 'E-SUB'
                else:
                    mark_list[s_idx[0]] = 'B-SUB'
                    mark_list[s_idx[-1]] = 'E-SUB'
                    for idx in range(1, len(s_idx) - 1):
                        mark_list[s_idx[idx]] = 'I-SUB'
            for o_idx in o_idx_list:
                if len(o_idx) == 1:
                    mark_list[o_idx[0]] = 'B-OBJ'
                elif len(o_idx) == 2:
                    mark_list[o_idx[0]] = 'B-OBJ'
                    mark_list[o_idx[1]] = 'E-OBJ'
                else:
                    mark_list[o_idx[0]] = 'B-OBJ'
                    mark_list[o_idx[-1]] = 'E-OBJ'
                    for idx in range(1, len(o_idx) - 1):
                        mark_list[o_idx[idx]] = 'I-OBJ'
        return mark_list

    def _is_valid_input_data(self, input_line):
        """is the input data valid"""
        try:
            dic, p = input_line.strip().split('\t')
            dic = json.loads(dic)
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
        dic, p = line.strip().decode('utf-8').split('\t')
        dic = json.loads(dic)
        sentence = dic['text']
        sentence_term_list = [item['word'] for item in dic['postag']]
        token_idx_list = self._get_token_idx(sentence_term_list, sentence)
        sentence_pos_list = [item['pos'] for item in dic['postag']]
        sentence_emb_slot = [self._feature_dict['wordemb_dict'].get(w, self._UNK_IDX) \
                for w in sentence_term_list]
        sentence_pos_slot = [self._feature_dict['postag_dict'].get(pos, self._UNK_IDX) \
                for pos in sentence_pos_list]
        p_emb_slot = [self._feature_dict['wordemb_dict'].get(p, self._UNK_IDX)] * len(sentence_term_list)
        if 'spo_list' not in dic:
            label_slot = [self._feature_dict['so_label_dict']['O']] * \
                    len(sentence_term_list)
        else:
            label_slot = self._cal_mark_slot(dic['spo_list'], sentence, p, token_idx_list)
            label_slot = [self._feature_dict['so_label_dict'][label] for label in label_slot]
        feature_slot = [sentence_emb_slot, sentence_pos_slot, p_emb_slot]
        input_fields = "\t".join([json.dumps(dic, ensure_ascii=False), p])
        output_slot = feature_slot
        #verify the feature is valid or not
        if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 \
                or len(label_slot) == 0:
            return None
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

    def get_label_output(self, tensor_list):
        """Output final label, used during predict and test"""
        dict_name = 'so_label_dict'
        if len(self._reverse_dict[dict_name]) == 0:
            self._get_reverse_dict(dict_name)

        max_idx = tensor_list.argmax()
        return self._reverse_dict[dict_name].get(max_idx, 0)

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
                dic, p = line.strip().split('\t')
                dic = json.loads(dic)
                sentence = dic['text']
                sentence_term_list = [item['word'] for item in dic['postag']]
                token_idx_list = self._get_token_idx(sentence_term_list, sentence)
                sentence_pos_list = [item['pos'] for item in dic['postag']]
                sentence_emb_slot = [self._feature_dict['wordemb_dict'].get(w, self._UNK_IDX) \
                                     for w in sentence_term_list]
                sentence_pos_slot = [self._feature_dict['postag_dict'].get(pos, self._UNK_IDX) \
                                     for pos in sentence_pos_list]
                p_emb_slot = [self._feature_dict['wordemb_dict'].get(p, self._UNK_IDX)] * len(sentence_term_list)
                if 'spo_list' not in dic:
                    label_slot = [self._feature_dict['so_label_dict']['O']] * \
                                 len(sentence_term_list)
                else:
                    label_slot = self._cal_mark_slot(dic['spo_list'], sentence, p, token_idx_list)
                    label_slot = [self._feature_dict['so_label_dict'][label] for label in label_slot]
                feature_slot = [sentence_emb_slot, sentence_pos_slot, p_emb_slot]
                input_fields = "\t".join([json.dumps(dic, ensure_ascii=False), p])
                output_slot = feature_slot
                # verify the feature is valid or not
                if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 \
                        or len(label_slot) == 0:
                    continue

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
        p_word_list = []
        input_raw_list = []

        for sent in data:
            input_raw, w_id, p_id, pw_id, l_id = sent
            w_id, p_id, pw_id, l_id = w_id[0:conf_dict['max_len']], p_id[0:conf_dict['max_len']], pw_id[0:conf_dict['max_len']], l_id[0:conf_dict['max_len']]
            s_len = len(w_id)

            w_id = w_id + [0] * (conf_dict['max_len'] - len(w_id))
            p_id = p_id + [0] * (conf_dict['max_len'] - len(p_id))
            pw_id = pw_id + [0] * (conf_dict['max_len'] - len(pw_id))
            l_id = l_id + [0] * (conf_dict['max_len'] - len(l_id))

            word.append(w_id)
            postag.append(p_id)
            seq_len.append(s_len)
            label_ids.append(l_id)
            p_word_list.append(pw_id)
            input_raw_list.append(input_raw)

            if len(word) == conf_dict['batch_size']:
                yield word, postag, seq_len, label_ids, p_word_list, input_raw_list
                word = []
                postag = []
                seq_len = []
                label_ids = []
                p_word_list = []
                input_raw_list = []

        if len(word) != 0:
            yield word, postag, seq_len, label_ids, p_word_list, input_raw_list


