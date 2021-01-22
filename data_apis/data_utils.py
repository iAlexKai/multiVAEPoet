# -*- coding: utf-8 -*-
import numpy as np
import random

# Data feed
class LongDataLoader(object):
    """
    A special efficient data loader for TBPTT
    """
    batch_size = 0
    ptr = 0
    num_batch = None
    data_size = None
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch)

    # def _prepare_batch(self, cur_grid, prev_grid):
    def _prepare_batch(self, cur_grid, type=1):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, shuffle=False, sample_rate=1):
        """
        prepare shuffled batches
        """
        self.ptr = 0
        self.batch_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.all_batch = []
        for i in range(temp_num_batch):
            self.all_batch.append(self.data[i * self.batch_size:(i + 1) * self.batch_size])
        # 只采样sample_rate分之一的batch，总数量为len // sample_rate个batch
        if sample_rate == 1 and shuffle is False:
            self.batch = self.all_batch
        else:
            self.batch = random.sample(self.all_batch, int(len(self.all_batch) // sample_rate))
        # import pdb
        # pdb.set_trace()
        # left_over = self.data_size - temp_num_batch * batch_size

        # shuffle batch indexes
        # if shuffle:
        #     self._shuffle_batch_indexes()

        self.num_batch = len(self.batch)

        # if sample_rate != 1:
        #     print("%s begins with 1/%d epoch with %d batches" % (self.name, sample_rate, self.num_batch))
        # else:
        #     # print("%s begins with %d batches with %d left over samples" % (self.name, self.num_batch, left_over))
        #     print("%s begins with a epoch with %d batches" % (self.name, self.num_batch))

    def next_batch(self, type=1):
        """
        output the id of the next batch
        """
        if self.ptr < self.num_batch:
            current_grid = self.batch[self.ptr]  # 里面有batch_size条训练样本[title, last_sentence, current_sentence]
            # if self.ptr > 0:
            #     prev_grid = self.batch[self.ptr-1]
            # else:
            #     prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, type=type)
        else:
            return None

    def next_batch_test(self):
        """
        prepare testing data for feeding in the CVAE-D model
        """
        if self.ptr < self.num_batch:
            # print("当前： ", self.ptr, " ", self.num_batch)
            current_grid = self.batch[self.ptr]
            if self.ptr > 0:
                prev_grid = self.batch[self.ptr-1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_test_batch(cur_grid=current_grid)
        else:
            return None


class SWDADataLoader(LongDataLoader):
    def __init__(self, name, data, config):
        self.name = name
        self.data = data
        self.data_size = len(data)
        self.max_utt_size = config.maxlen
        self.max_de_len = config.maxlen

    def pad_to(self, tokens, size, do_pad=True):
        """
        pad the input data to a fixed length
        """
        # 截取前size-1个字，加上</s>
        if len(tokens) >= size:
            return tokens[0: size-1] + [tokens[-1]]
        # 将0pad到后面
        elif do_pad:
            return tokens + [0] * (size - len(tokens))
        else:
            return tokens

    def sample_one_batch(self, sentiment=False):
        sample_ptr = random.randint(0, self.num_batch-1)
        current_grid = self.batch[sample_ptr]  # 里面有batch_size条训练样本[title, last_sentence, current_sentence]
        if sentiment:
            return self._prepare_sentiment_batch(cur_grid=current_grid)
        else:
            return self._prepare_batch(cur_grid=current_grid)

    def _prepare_batch(self, cur_grid, type=1):
        rows = cur_grid
        lines, line_lens = [], []
        for row in rows:
            lines.append(self.pad_to(row, size=self.max_utt_size))
            line_lens.append(len(row))

        vec_line = np.zeros((self.batch_size, self.max_utt_size), dtype=np.int64)
        vec_line_lens = np.array(line_lens)
        for b_id in range(self.batch_size):
            vec_line[b_id, 0: vec_line_lens[b_id]] = lines[b_id][0: vec_line_lens[b_id]]

        return vec_line, vec_line_lens

    def _prepare_test_batch(self, cur_grid):
        """
        prepare every batch of testing data
        """
        rows = cur_grid
        titles = []
        # import pdb
        # pdb.set_trace()
        for row in rows:
            titles.append(self.pad_to(row[0], size=self.title_size))

        # 将4个list转为4个np的vector
        vec_title = np.zeros((self.batch_size, self.title_size), dtype=np.int64)

        for b_id in range(self.batch_size):
            vec_title[b_id, :] = np.array(titles[b_id])

        return vec_title  # 将4个vector返回给模型训练