# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import numpy as np

logger = logging.getLogger(__name__)
from sklearn.externals import joblib


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, tag_names=None, tfidf_feat=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.tag_names = tag_names
        self.tfidf_feat = tfidf_feat.tolist()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, head_positions, tail_positions, mfeats, from_vec):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.head_positions = head_positions
        self.tail_positions = tail_positions
        self.mfeats = mfeats
        self.from_vec = from_vec


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, data_dir):
        super(DataProcessor, self).__init__()
        self.dv_tag_names = self.read_tag_names(os.path.join(data_dir, './divorce/selectedtags.txt'))
        self.lb_tag_names = self.read_tag_names(os.path.join(data_dir, './labor/selectedtags.txt'))
        self.ln_tag_names = self.read_tag_names(os.path.join(data_dir, './loan/selectedtags.txt'))
        self.tfidf = TfidfVectorizer(tokenizer=jieba.lcut, max_df=0.7, min_df=5, max_features=500)
        logger.info(self.dv_tag_names)
        logger.info(np.sum([len(v) for v in self.dv_tag_names]))
        logger.info(self.lb_tag_names)
        logger.info(np.sum([len(v) for v in self.lb_tag_names]))
        logger.info(self.ln_tag_names)
        logger.info(np.sum([len(v) for v in self.ln_tag_names]))

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""

        # Initialize the list to store examples
        examples = []

        # Load the training data from JSON files
        dv_X, dv_y = self.read_json(os.path.join(data_dir, './divorce/train.json'))
        lb_X, lb_y = self.read_json(os.path.join(data_dir, './labor/train.json'))
        ln_X, ln_y = self.read_json(os.path.join(data_dir, './loan/train.json'))

        cnt = 0  # counter for unique identifier for each example

        # Compute tf-idf features for the training data
        tfidf_features = self.tfidf.fit_transform(dv_X + lb_X + ln_X).todense()

        # Log the shape of the tf-idf features
        logger.info(tfidf_features[0].shape)

        # Create InputExample objects from the divorce training data and append to examples list
        for text, label in zip(dv_X, dv_y):
            examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, tag_names=self.dv_tag_names,
                                         tfidf_feat=tfidf_features[cnt]))
            cnt += 1

        # Create InputExample objects from the labor training data and append to examples list
        for text, label in zip(lb_X, lb_y):
            examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, tag_names=self.lb_tag_names,
                                         tfidf_feat=tfidf_features[cnt]))
            cnt += 1

        # Create InputExample objects from the loan training data and append to examples list
        for text, label in zip(ln_X, ln_y):
            examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, tag_names=self.ln_tag_names,
                                         tfidf_feat=tfidf_features[cnt]))
            cnt += 1

        # Uncomment the following lines to use additional data
        # dv_X_plus, dv_y_plus = self.read_json(os.path.join(data_dir, './new_data/new_divorce_pred.json'))
        # lb_X_plus, lb_y_plus = self.read_json(os.path.join(data_dir, './new_data/new_labor_pred.json'))
        # ln_X_plus, ln_y_plus = self.read_json(os.path.join(data_dir, './new_data/new_loan_pred.json'))

        # for text, label in zip(dv_X_plus, dv_y_plus):
        #     examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, task_name='DV'))
        #     cnt += 1

        # for text, label in zip(lb_X_plus, lb_y_plus):
        #     examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, task_name='LB'))
        #     cnt += 1

        # for text, label in zip(ln_X_plus, ln_y_plus):
        #     examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, task_name='LN'))
        #     cnt += 1

        # Calculate the lengths of each type of training data
        len_dv = len(dv_X)
        len_lb = len(lb_X)
        len_ln = len(ln_X)
        # Log the count of each type of training data
        logger.info('dv cnt: {}, lb cnt: {}, ln cnt: {}'.format(len_dv, len_lb, len_ln))
        return examples[:1000]

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""

        # Initialize the list to store examples
        examples = []

        # Load the testing data from JSON files
        dv_X, dv_y = self.read_json(os.path.join(data_dir, './divorce/test.json'))
        lb_X, lb_y = self.read_json(os.path.join(data_dir, './labor/test.json'))
        ln_X, ln_y = self.read_json(os.path.join(data_dir, './loan/test.json'))

        # Log the lengths of each type of testing data
        logger.info('{}, {}, {}'.format(len(dv_X), len(lb_X), len(ln_X)))

        # Compute tf-idf features for the testing data
        tfidf_features = self.tfidf.transform(dv_X + lb_X + ln_X).todense()

        cnt = 0  # counter for unique identifier for each example

        # Create InputExample objects from the divorce testing data and append to examples list
        for text, label in zip(dv_X, dv_y):
            examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, tag_names=self.dv_tag_names,
                                         tfidf_feat=tfidf_features[cnt]))
            cnt += 1

        # Create InputExample objects from the labor testing data and append to examples list
        for text, label in zip(lb_X, lb_y):
            examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, tag_names=self.lb_tag_names,
                                         tfidf_feat=tfidf_features[cnt]))
            cnt += 1

        # Create InputExample objects from the loan testing data and append to examples list
        for text, label in zip(ln_X, ln_y):
            examples.append(InputExample(guid=cnt, text_a=text, text_b=None, label=label, tag_names=self.ln_tag_names,
                                         tfidf_feat=tfidf_features[cnt]))
            cnt += 1

        # Return the examples list
        return examples

    def get_labels(self, x: list):
        """
        This function converts a list of labels into a binary representation.

        :param x: A list of labels.
        :return: A binary list representing the presence of each label.
        """

        # Initialize a list of 20 zeros. This is the binary representation of labels.
        labels = [0] * 20

        # If the input list is empty, return the all-zero list
        if len(x) == 0:
            return labels
        else:
            # For each label in the input list
            for v in x:
                # The label is a string of format 'vXX'. We extract the 'XX' part, convert it to an integer,
                # subtract 1 (because Python uses zero-based indexing), and then set the corresponding element in the
                # binary list to 1.
                labels[int(v[2:]) - 1] = 1

        # Return the binary representation of labels
        return labels

    def read_json(self, input_file):
        X = []
        y = []
        with open(input_file, encoding='utf-8') as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                for one in line:
                    X.append(one['sentence'])
                    labels = self.get_labels(one['labels'])
                    y.append(labels)
        return X, y

    def read_tag_names(self, fin: str):
        lines = []
        with open(fin, encoding='utf8') as f:
            for line in f.readlines():
                lines.append(line.strip().replace('|', '或'))
        return lines


def get_tag_positions(tokens):
    """
    This function finds the positions of certain tags in a list of tokens.

    :param tokens: A list of tokens.
    :return: Two lists, `head` and `tail`, representing the positions of '(' and ')' respectively in the list of tokens.
    """

    # Flag to indicate the start of search area for tags
    start_tokens_b = False
    # Initialize empty lists to hold positions of '(' and ')'
    head, tail = [], []

    # Loop over the tokens
    for idx, token in enumerate(tokens):
        # If we encounter "<sep>", we start looking for tags
        if token == "<sep>":
            start_tokens_b = True

        # If we are in the search area
        if start_tokens_b:
            # If we find '(', add its position to `head`
            if token == "(":
                head.append(idx)
            # If we find ')', add its position to `tail`
            if token == ')':
                tail.append(idx)

    # Assert that we have found 20 '(' and 20 ')' tags
    assert len(head) == 20 == len(tail)

    # Assert that the tokens at the positions in `head` are all '('
    for idx in head:
        assert tokens[idx] == '('
    # Assert that the tokens at the positions in `tail` are all ')'
    for idx in tail:
        assert tokens[idx] == ')'

    # Return the positions of '(' and ')' tags
    return head, tail


def get_from_vec(tag_names, min_weight: float = 0):
    if tag_names[0] == '婚后有子女':
        res = [1 - 2 * min_weight, min_weight, min_weight]
    if tag_names[0] == '解除劳动关系':
        res = [min_weight, 1 - 2 * min_weight, min_weight]
    if tag_names[0] == '债权人转让债权':
        res = [min_weight, min_weight, 1 - 2 * min_weight]
    assert sum(res) == 1
    return res


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
            This function converts examples to features that can be directly given as input to a BERT model.

        :param examples: A list of InputExample instances.
        :param max_seq_length: Maximum length of the sequences.
        :param tokenizer: Tokenizer instance to tokenize the inputs.
        :param cls_token_at_end: Whether to append [CLS] token at the end.
        :param cls_token: [CLS] token.
        :param cls_token_segment_id: Segment id for [CLS] token.
        :param sep_token: [SEP] token.
        :param sep_token_extra: Whether to add an extra [SEP] token.
        :param pad_on_left: Whether to pad on left side or right side.
        :param pad_token: Padding token.
        :param pad_token_segment_id: Segment id for padding token.
        :param sequence_a_segment_id: Segment id for first sequence.
        :param sequence_b_segment_id: Segment id for second sequence.
        :param mask_padding_with_zero: Whether to mask padding with zero.
        :return: A list of InputFeatures instances.
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize text_a and text_b
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = ''.join('('+tag+')' for tag in example.tag_names)
        tokens_b = tokenizer.tokenize(tokens_b)

        # Truncate if needed
        if len(tokens_b) > max_seq_length:
            tokens_b = tokens_b[:max_seq_length]
        if len(tokens_a) > max_seq_length - len(tokens_b) - 3:
            tokens_a = tokens_a[:max_seq_length - len(tokens_b) - 3]
        # Prepare tokens and segment_ids
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        # Add [CLS] token
        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        # Convert tokens to input_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Prepare input_mask
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Add padding
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        # Ensure everything is of correct length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Additional features
        label_id = example.label
        head_positions, tail_positions = get_tag_positions(tokens)
        mfeats = example.tfidf_feat
        from_vec = get_from_vec(example.tag_names, 0)

        # Logging for the first 5 examples
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in example.label]))
            logger.info("head position: %s" % " ".join([str(x) for x in head_positions]))
            logger.info("tail position: %s" % " ".join([str(x) for x in tail_positions]))
            logger.info('from vec: %s' % ' '.join([str(x) for x in from_vec]))

        # Add the features for this example to the list
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          head_positions=head_positions,
                          tail_positions=tail_positions,
                          mfeats=mfeats,
                          from_vec=from_vec))
    # Return the list of features
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def compute_metrics(logits, labels):
    assert logits.shape == labels.shape
    dv_end_idx = 3711
    lb_end_idx = 6348
    ln_end_idx = 8462
    probs = torch.sigmoid(logits)
    # print(probs.shape)
    assert probs.shape == labels.shape
    y_pred = torch.zeros(labels.shape)
    y_pred[probs > 0.5] = 1

    y_pred = y_pred.numpy()
    labels = labels.numpy()
    dv_pred = y_pred[0:dv_end_idx]
    dv_labels = labels[0:dv_end_idx]

    lb_pred = y_pred[dv_end_idx:lb_end_idx]
    lb_labels = labels[dv_end_idx:lb_end_idx]

    ln_pred = y_pred[lb_end_idx:]
    ln_labels = labels[lb_end_idx:]

    macro_dv = f1_score(dv_labels, dv_pred, average='macro')
    micro_dv = f1_score(dv_labels, dv_pred, average='micro')
    macro_lb = f1_score(lb_labels, lb_pred, average='macro')
    micro_lb = f1_score(lb_labels, lb_pred, average='micro')
    macro_ln = f1_score(ln_labels, ln_pred, average='macro')
    micro_ln = f1_score(ln_labels, ln_pred, average='micro')
    dv_score = 0.5 * macro_dv + 0.5 * micro_dv
    lb_score = 0.5 * macro_lb + 0.5 * micro_lb
    ln_score = 0.5 * macro_ln + 0.5 * micro_ln
    final = (dv_score + lb_score + ln_score) / 3
    logger.info('DV: {:.3f}, LB: {:.3f}, LN: {:.3f}, final: {:.3f}'.format(dv_score, lb_score, ln_score, final))
    # logger.info(classification_report(dv_labels, dv_pred))
    # logger.info(classification_report(lb_labels, lb_pred))
    # logger.info(classification_report(ln_labels, ln_pred))
    return {'dv_score': dv_score, 'lb_score': lb_score, 'ln_score': ln_score, 'final': final}
