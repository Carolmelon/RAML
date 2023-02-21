import json
import pickle
from argparse import Namespace

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import random
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import transformers

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class AmazonReview(Dataset):
    def __init__(self, args, mode, KG, max_len=128, bert_model='albert-base-v1'):
        super(AmazonReview, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.KG = KG
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.max_len = max_len
        self.mode = mode

        self.data = pickle.load(open('xxx/amazon.pkl', 'rb'))

        self.key_map = dict({eachkey:idx for idx, eachkey in enumerate(self.data.keys())})

        if self.mode == 'train':
            self.classes_idx = np.array([2, 3, 4, 7, 11, 12, 13, 18, 19, 20])
        elif self.mode =='val':
            self.classes_idx = np.array([1, 22, 23, 6, 9])
        elif self.mode == 'test':
            self.classes_idx = np.array([0, 5, 14, 15, 8, 10, 16, 17, 21])

    def to_graphs(self, seq):
        graph_lst = []
        for raw_x in seq:
            x, edge_index, edge_attr, batch, num_nodes, num_edges, entities_id, edge_type = \
                self.KG.reduce_connected([raw_x], raw=False)
            graph_lst.append((x, edge_index, edge_attr, num_nodes, num_edges, entities_id, edge_type))
        return graph_lst

    def get_part_data(self, sel_category, sel_sample):
        seq = list(self.data[sel_category][sel_sample])

        graph_lst = []

        if self.args.use_kg:
            graph_lst = self.to_graphs(seq)

        encoded_seq = self.tokenizer(seq,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=self.max_len,
                                     return_tensors='pt')

        token_ids = encoded_seq['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_seq['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_seq['token_type_ids'].squeeze(0)

        return token_ids, attn_masks, token_type_ids, graph_lst

    def __getitem__(self, index):
        inputa=[]
        inputb = []

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            token_ids_a, token_ids_b = [], []
            attn_masks_a, attn_masks_b = [], []
            token_type_ids_a, token_type_ids_b = [], []
            graph_lst_a, graph_lst_b = [], []

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(len(self.data[self.choose_classes[j]]))
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                process_data = self.get_part_data(self.choose_classes[j], choose_samples)
                token_ids_a.append(process_data[0][:self.k_shot])
                token_ids_b.append(process_data[0][self.k_shot:])

                attn_masks_a.append(process_data[1][:self.k_shot])
                attn_masks_b.append(process_data[1][self.k_shot:])

                token_type_ids_a.append(process_data[2][:self.k_shot])
                token_type_ids_b.append(process_data[2][self.k_shot:])

                graph_lst_a.extend(process_data[3][:self.k_shot])
                graph_lst_b.extend(process_data[3][self.k_shot:])

                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            token_ids_a = torch.cat(token_ids_a, dim=0)
            token_ids_b = torch.cat(token_ids_b, dim=0)

            attn_masks_a = torch.cat(attn_masks_a, dim=0)
            attn_masks_b = torch.cat(attn_masks_b, dim=0)

            token_type_ids_a = torch.cat(token_type_ids_a, dim=0)
            token_type_ids_b = torch.cat(token_type_ids_b, dim=0)

            inputa.append([token_ids_a, attn_masks_a, token_type_ids_a, graph_lst_a])
            inputb.append([token_ids_b, attn_masks_b, token_type_ids_b, graph_lst_b])

        return inputa, torch.LongTensor(support_y), inputb, torch.LongTensor(
            query_y)

# num_classes: n-way
# update_batch_size: k-shot
# update_batch_size_eval: k-query
args = Namespace(datadir='xxx', datasource='huffpost', knn=1, log=1, logdir='xxx', meta_batch_size=1, meta_lr=2e-05, metatrain_iterations=10000, num_classes=5, num_filters=64, num_test_task=600, num_updates=5, num_updates_test=10, ratio=1.0, resume=0, select_data=-1, test_dataset=-1, test_epoch=-1, test_set=1, trail=0, train=1, update_batch_size=1, update_batch_size_eval=5, update_lr=0.001, use_kg=0, warm_epoch=0, weight_decay=0.0)

class Huffpost(Dataset):
    def __init__(self, args, mode, text_maxlength, tokenizer, n_context, answer_maxlength=20, max_len=128, bert_model='t5-base', support_tokenizer='albert-base-v1', data=None, support_n_context=1, opt=None):
        super(Huffpost, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes  # 5个类
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval    # 1+5=6
        self.n_way = args.num_classes  # n-way      5
        self.query_tokenizer = transformers.T5Tokenizer.from_pretrained(bert_model)
        self.k_shot = args.update_batch_size  # k-shot      1
        self.k_query = args.update_batch_size_eval  # for evaluation    5(query)
        self.set_size = self.n_way * self.k_shot  # num of samples per set      5*1
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation     5*5
        self.max_len = max_len
        self.mode = mode

        self.n_context = n_context
        self.support_n_context = support_n_context
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.support_tokenizer = AutoTokenizer.from_pretrained(support_tokenizer)
        self.tokenizer = self.query_tokenizer

        if opt.datasource == 'huffpost':
            if self.mode == 'train':
                self.classes_idx = np.array([30, 25, 36, 4, 14, 22, 28, 7, 20, 8, 9, 3, 15, 34, 24, 29, 1,
                                             6, 0, 16, 37, 5, 33, 35, 27])
            elif self.mode =='val':
                self.classes_idx = np.array([17, 13, 18,  2, 40, 39])
            elif self.mode == 'test':
                self.classes_idx = np.array([32, 11, 23, 19, 10, 26, 12, 31, 21, 38])
        elif opt.datasource == 'amazonreview':
            if self.mode == 'train':
                self.classes_idx = np.array([2, 3, 4, 7, 11, 12, 13, 18, 19, 20])
            elif self.mode == 'val':
                self.classes_idx = np.array([1, 22, 23, 6, 9])
            elif self.mode == 'test':
                self.classes_idx = np.array([0, 5, 14, 15, 8, 10, 16, 17, 21])
        else:
            print("invalid opt.datasource, please check!")
            exit(1)

        if data == None:
            self.data = json.load(open('data_fewshot/dict_data.json'))
        else:
            self.data = data
        # 加上类标签
        self.idx_to_cls = json.load(open('data_fewshot/idx_to_cls.json'))
        self.idx_to_token = {}
        for idx, cls in self.idx_to_cls.items():
            encoded_seq = self.tokenizer(cls,
                                         padding='max_length',
                                         truncation=True,
                                         max_length=self.max_len,
                                         return_tensors='pt')
            self.idx_to_token[idx] = encoded_seq

    # 获取文字结构
    def get_part_data(self, sel_category, sel_sample):
        seq = []
        for i in range(len(sel_sample)):    # 一项一项地添加
            seq.append(self.data[str(sel_category)][sel_sample[i]])

        return_results = []
        for i, example in enumerate(seq):
            index = sel_sample[i]
            question = example['headline']
            target = example['category']

            if 'ctxs' in example and self.n_context is not None:
                f = "title:" + " {} " + 'context:' + " {}"
                contexts = example['ctxs'][:self.n_context]
                passages = [f.format(c['title'], c['text']) for c in contexts]
                scores = [float(c['score']) for c in contexts]
                scores = torch.tensor(scores)
                # TODO(egrave): do we want to keep this?
                if len(contexts) == 0:
                    contexts = [question]
            else:
                passages, scores = None, None

            return_result = {
                'index': index,
                'headline': question,
                'target': target,
                'passages': passages,
                'scores': scores
            }
            return_results.append(return_result)

        return return_results

        # encoded_seq = self.tokenizer(seq,
        #                              padding='max_length',
        #                              truncation=True,
        #                              max_length=self.max_len,
        #                              return_tensors='pt')
        #
        # token_ids = encoded_seq['input_ids'].squeeze(0)  # tensor of token ids
        # attn_masks = encoded_seq['attention_mask'].squeeze(
        #     0)  # binary tensor with "0" for padded values and "1" for the other values
        # token_type_ids = encoded_seq['token_type_ids'].squeeze(0)
        #
        # return token_ids, attn_masks, token_type_ids

    # 转换成tensor
    def process_batch_data(self, batch, support=False):
        assert (batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        # 都不要附加passages
        # support = True

        def append_question(example):
            if support or example['passages'] is None:     # support不需要辅助信息
                return [example['headline']]
            elif support:
                # support只编码self.support_n_context条信息
                return [example['headline'] + " " + example['passages'][idx] for idx in range(self.support_n_context)]
            return [example['headline'] + " " + t for t in example['passages']]

        text_passages = [append_question(example) for example in batch]
        tokenizer = self.support_tokenizer
        # tokenizer，分词，support是Albert，query是t5(x，也是Albert)
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.support_tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

    def __getitem__(self, index):
        inputa=[] # support
        inputb = [] # query
        support_x = []
        query_x = []

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])    #   [1,5]
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])    #   [1,25]
        support_y_true_cls = []   #   [1,5]
        query_y_true_cls = []    #   [1,25]

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            support_x_batch = []
            query_x_batch = []
            support_x_batch_token = []
            query_x_batch_token = []

            support_y_true_cls_ids = [[], [], [], []]
            query_y_true_cls_ids = [[], [], [], []]

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(len(self.data[str(self.choose_classes[j])]))    # 属于当前选中类的 句子的索引集合
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]   # 每个类选_个samples
                process_data = self.get_part_data(self.choose_classes[j], choose_samples)   # (6,maxlen)
                support_x_batch.append(process_data[:self.k_shot])
                query_x_batch.append(process_data[self.k_shot:])
                # token_ids_a.append(process_data[0][:self.k_shot])
                # token_ids_b.append(process_data[0][self.k_shot:])

                # attn_masks_a.append(process_data[1][:self.k_shot])
                # attn_masks_b.append(process_data[1][self.k_shot:])
                #
                # token_type_ids_a.append(process_data[2][:self.k_shot])
                # token_type_ids_b.append(process_data[2][self.k_shot:])

                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j
                # 用类标签的文本来增强，存储真实的类
                support_y_true_cls_ids[0].append(self.idx_to_token[str(self.choose_classes[j])]['input_ids'])
                query_y_true_cls_ids[0].append(
                    torch.cat(
                        [self.idx_to_token[str(self.choose_classes[j])]['input_ids'] for i in range(self.k_query)],
                        dim=0)
                )

                support_y_true_cls_ids[1].append(self.idx_to_token[str(self.choose_classes[j])]['attention_mask'])
                query_y_true_cls_ids[1].append(
                    torch.cat(
                        [self.idx_to_token[str(self.choose_classes[j])]['attention_mask'] for i in range(self.k_query)],
                        dim=0)
                )
                # 处理成token      support不编码passages
                support_x_batch_token.append(self.process_batch_data(process_data[:self.k_shot], support=True))
                # support_x_batch_token.append(self.process_batch_data(process_data[:self.k_shot]))
                query_x_batch_token.append(self.process_batch_data(process_data[self.k_shot:]))

            support_x.append([
                torch.cat(dim=0, tensors=[item[0] for item in support_x_batch_token]),  # index
                torch.cat(dim=0, tensors=[item[1] for item in support_x_batch_token]),  # label
                torch.cat(dim=0, tensors=[item[2] for item in support_x_batch_token]),  # label_mask
                torch.cat(dim=0, tensors=[item[3] for item in support_x_batch_token]),  # passages
                torch.cat(dim=0, tensors=[item[4] for item in support_x_batch_token]),  # passages_mask
            ])
            query_x.append([
                torch.cat(dim=0, tensors=[item[0] for item in query_x_batch_token]),  # index
                torch.cat(dim=0, tensors=[item[1] for item in query_x_batch_token]),  # label
                torch.cat(dim=0, tensors=[item[2] for item in query_x_batch_token]),  # label_mask
                torch.cat(dim=0, tensors=[item[3] for item in query_x_batch_token]),  # passages
                torch.cat(dim=0, tensors=[item[4] for item in query_x_batch_token]),  # passages_mask
            ])

            support_y_true_cls_ids[0] = torch.cat(support_y_true_cls_ids[0], dim=0)
            query_y_true_cls_ids[0] = torch.cat(query_y_true_cls_ids[0], dim=0)

            support_y_true_cls_ids[1] = torch.cat(support_y_true_cls_ids[1], dim=0)
            query_y_true_cls_ids[1] = torch.cat(query_y_true_cls_ids[1], dim=0)

            support_y_true_cls.append(support_y_true_cls_ids)
            query_y_true_cls.append(query_y_true_cls_ids)
        # 这里面support_y_true_cls是小写，support_x[0][1]里面的label是大写
        return support_x, torch.LongTensor(support_y), support_y_true_cls, query_x, torch.LongTensor(
            query_y), query_y_true_cls
