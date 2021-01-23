# -*- coding: utf-8 -*-
import pickle as pkl
from collections import Counter, defaultdict
import numpy as np
import data_apis.data_process as data_process


class LoadPoem(object):
    # 即便是做align实验，vocab也不能变，必须和原先完全一样，因此build vocab所使用的的数据集还是之前的train数据集
    def __init__(self, corpus_path, vocab_path, test_path,  max_vocab_cnt, with_sentiment=False,
                 word2vec=None, word2vec_dim=None):
        """
        the folder that contains the demo data
        """
        self._path = corpus_path
        self._path_test = test_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.unk_id = None

        Data = data_process.read_data(self._path, type=2+int(with_sentiment))
        data = data_process.prepare_poem(Data, 68000, 2000, 2000, type=1+int(with_sentiment))  # train_len, val_len, test_len

        self.train_corpus = defaultdict(list)
        self.train_corpus['pos'] = self.add_start_end_label(data["train_pos"], type=1 + int(with_sentiment))
        self.train_corpus['neu'] = self.add_start_end_label(data["train_neu"], type=1 + int(with_sentiment))
        self.train_corpus['neg'] = self.add_start_end_label(data["train_neg"], type=1 + int(with_sentiment))

        self.valid_corpus = defaultdict(list)
        self.valid_corpus['pos'] = self.add_start_end_label(data["valid_pos"], type=1 + int(with_sentiment))
        self.valid_corpus['neu'] = self.add_start_end_label(data["valid_neu"], type=1 + int(with_sentiment))
        self.valid_corpus['neg'] = self.add_start_end_label(data["valid_neg"], type=1 + int(with_sentiment))

        self.load_vocab(vocab_path)
        test_titles = data_process.read_data(self._path_test, type=1)
        data["test"] = data_process.prepare_test_data(test_titles)
        self.test_corpus = self.process_test(data["test"])

    def load_vocab(self, vocab_path):
        with open(vocab_path) as vocab_file:
            self.vocab = vocab_file.read().strip().split('\n')
            self.rev_vocab = {self.vocab[idx]: idx for idx in range(len(self.vocab))}
            self.unk_id = self.rev_vocab['<unk>']

    # 根据输入的训练对情感的不同，分布五个类别，返回一个dict
    def process_sentiment(self, sentiment_data):
        """
        prepare sentiment validating data with starting and ending label
        transfer sentiment to integer
        """
        new_utts_dict = {'1':[], '2':[], '3':[], '4':[], '5':[]}
        for l in sentiment_data:
            title = ["<s>"] + l[0] + ["</s>"]
            context = ["<s>"] + l[1] + ["</s>"]
            target = ["<s>"] + l[2] + ["</s>"]
            sentiment = l[3][0]
            new_utts_dict[sentiment].append([title, context, target, sentiment])
        return new_utts_dict

    def add_start_end_label(self, data, type=1):
        """
        prepare training  and  validating data with starting and ending label 
        """
        new_utts = []
        if type == 1:
            for line in data:
                title = ["<s>"] + line[0] + ["</s>"]
                context = ["<s>"] + line[1] + ["</s>"]
                target = ["<s>"] + line[2] + ["</s>"]
                new_utts.append([title, context, target])

        elif type == 2:
            for line in data:
                title = ["<s>"] + line[0] + ["</s>"]
                context = ["<s>"] + line[1] + ["</s>"]
                target = ["<s>"] + line[2] + ["</s>"]
                sentiment = line[3]
                new_utts.append([title, context, target, sentiment])

        elif type == 3:
            for line in data:
                new_utts.append([["<s>"] + list(line) + ["/s"]])

        else:
            print("Invalid type in process function")
            return

        return new_utts        

    def process_test(self, data):
        """
        prepare testing data with starting and ending label
        """
        new_utts = []
        for l in data:
            tem = []
            for sent in l:
                tem.append(["<s>"] + sent + ["</s>"])
            new_utts.append(tem)
        return new_utts  # 以输入的测试标题为topic，四句空诗

    def get_tokenized_test_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                tem = []
                for sent in line:
                    tem.append([self.rev_vocab.get(t, self.unk_id) for t in sent])
                results.append(tem)
            return results
        id_test = _to_id_corpus(self.test_corpus)
        return {'test': id_test}

    def get_tokenized_poem_corpus(self, train_corpus, valid_corpus):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([[self.rev_vocab.get(t, self.unk_id) for t in line[0]],   # 题目
                                [self.rev_vocab.get(t, self.unk_id) for t in line[1]],   # last sentence
                                [self.rev_vocab.get(t, self.unk_id) for t in line[2]],   # current sentence (target)
                                int(line[3][0])])  # current sentence (target)           # sentiment
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(train_corpus)
        id_valid = _to_id_corpus(valid_corpus)
        return {'train': id_train, 'valid': id_valid}


class LoadPretrainPoem(object):

    def __init__(self, corpus_path, vocab_path=None, word2vec_dim=None):
        """
        the folder that contains the demo data
        """
        self.corpus_path = corpus_path
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.unk_id = None

        self.data = self.read_specific_sentiment_data(self.corpus_path)
        self.vocab, self.rev_vocab = self.load_vocab(vocab_path)

    def load_vocab(self, vocab_path):
        with open(vocab_path) as vocab_file:
            vocab = vocab_file.read().strip().split('\n')
            rev_vocab = {vocab[idx]: idx for idx in range(len(vocab))}
            self.unk_id = rev_vocab['<unk>']
            return vocab, rev_vocab

    def read_specific_sentiment_data(self, root):
        with open('{}/positive.txt'.format(root)) as pos, \
                open('{}/neutral.txt'.format(root)) as neu, \
                open('{}/negative.txt'.format(root)) as neg:
            pos_data = pos.read().strip().split('\n')
            neu_data = neu.read().strip().split('\n')
            neg_data = neg.read().strip().split('\n')
            data = {'pos': [["<s>"] + list(line) + ["</s>"] for line in pos_data],
                    'neu': [["<s>"] + list(line) + ["</s>"] for line in neu_data],
                    'neg': [["<s>"] + list(line) + ["</s>"] for line in neg_data]}
            return data

    def get_tokenized_poem_corpus(self, train_corpus, valid_corpus):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])  # current sentence (target)
            return results
        # convert the corpus into ID
        id_train = _to_id_corpus(train_corpus)
        id_valid = _to_id_corpus(valid_corpus)
        return {'train': id_train, 'valid': id_valid}