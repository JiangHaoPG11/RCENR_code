import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator # operator模块输出一系列对应Python内部操作符的函数
from tqdm import tqdm

entities2id = {}
relations2id = {}
relation_tph = {}
relation_hpt = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))

def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))

class E(nn.Module):
    def __init__(self, entity_num, relation_num, dim, margin, norm, C):
        super(E, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num, embedding_dim=self.dim).to(device)
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num, embedding_dim=self.dim).to(device)
        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean").to(device)
        self.__data_init()

    def __data_init(self):
        # embedding.weight (Tensor) -形状为(num_embeddings, embedding_dim)的嵌入中可学习的权值
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.normalization_rel_embedding()
        self.normalization_ent_embedding()


    def normalization_ent_embedding(self):
        norm = self.ent_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.ent_embedding.weight.data.copy_(torch.from_numpy(norm))

    def normalization_rel_embedding(self):
        norm = self.rel_embedding.weight.detach().cpu().numpy()
        norm = norm / np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        self.rel_embedding.weight.data.copy_(torch.from_numpy(norm))

    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.from_numpy(np.array(ent_vector[i]))
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.from_numpy(np.array(rel_vector[i]))

    def distance(self, h, r, t):
        # 在 tensor 的指定维度操作就是对指定维度包含的元素进行操作，如果想要保持结果的维度不变，设置参数keepdim=True即可
        # 如 下面sum中 r_norm * h 结果是一个1024 *50的矩阵（2维张量） sum在dim的结果就变成了 1024的向量（1位张量） 如果想和r_norm对应元素两两相乘
        # 就需要sum的结果也是2维张量 因此需要使用keepdim= True报纸张量的维度不变
        # 另外关于 dim 等于几表示张量的第几个维度，从0开始计数，可以理解为张量的最开始的第几个左括号，具体可以参考这个https://www.cnblogs.com/flix/p/11262606.html
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        tail = self.ent_embedding(t)
        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3
        score = torch.norm(distance, p = self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):
        head = self.ent_embedding(h.to(device))
        rel = self.rel_embedding(r.to(device))
        tail = self.ent_embedding(t.to(device))
        distance = head + rel - tail
        # dim = -1表示的是维度的最后一维 比如如果一个张量有3维 那么 dim = 2 = -1， dim = 0 = -3
        score = torch.norm(distance, p=self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.sum(torch.max(torch.sum(embedding ** 2, dim=1, keepdim=True)- torch.autograd.Variable(torch.FloatTensor([1.0]).to(device)), torch.autograd.Variable(torch.FloatTensor([0.0]).to(device))))

    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)
        h = torch.squeeze(h, dim=1).to(device)
        r = torch.squeeze(r, dim=1).to(device)
        t = torch.squeeze(t, dim=1).to(device)
        h_c = torch.squeeze(h_c, dim=1).to(device)
        r_c = torch.squeeze(r_c, dim=1).to(device)
        t_c = torch.squeeze(t_c, dim=1).to(device)
        # torch.nn.embedding类的forward只接受longTensor类型的张量
        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)
        entity_embedding = self.ent_embedding(torch.cat([h, t, h_c, t_c]).to(device))
        relation_embedding = self.rel_embedding(torch.cat([r, r_c]).to(device))

        y = Variable(torch.Tensor([-1])).to(device)
        loss = self.loss_F(pos, neg, y)
        ent_scale_loss = self.scale_loss(entity_embedding)
        rel_scale_loss = self.scale_loss(relation_embedding)
        return loss + self.C * (ent_scale_loss/len(entity_embedding) + rel_scale_loss/len(relation_embedding))

class TransE:
    def __init__(self, entity, relation, triple_list, epochs, batch_size, embedding_dim=50, lr=0.01, margin=1.0, norm=1, C = 1.0, valid_triple_list = None):
        self.entities = entity
        self.relations = relation
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.valid_loss = 0.0
        self.valid_triples = valid_triple_list
        self.train_loss = []
        self.validation_loss = []
        self.test_triples = []
        self.C = C

        self.epochs = epochs
        self.batch_size = batch_size

    def data_initialise(self):
        print(len(self.entities))
        self.model = E(len(self.entities), len(self.relations), self.dimension, self.margin, self.norm, self.C)
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optim = optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def insert_pre_data(self, file1, file2):
        entity_dic = {}
        relation = {}
        with codecs.open(file1, 'r') as f1, codecs.open(file2, 'r') as f2:
          lines1 = f1.readlines()
          lines2 = f2.readlines()
          for line in lines1:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              entity_dic[int(line[0])] = json.loads(line[1])

          for line in lines2:
              line = line.strip().split('\t')
              if len(line) != 2:
                  continue
              relation[int(line[0])] = json.loads(line[1])
        self.model.input_pre_transe(entity_dic, relation)

    def insert_test_data(self, file1, file2, file3):
        self.insert_pre_data(file1, file2)
        triple_list = []
        with codecs.open(file3, 'r') as f4:
          content = f4.readlines()
          for line in content:
              triple = line.strip().split("\t")
              if len(triple) != 3:
                  continue
              head = int(entities2id[triple[0]])
              relation = int(relations2id[triple[1]])
              tail = int(entities2id[triple[2]])
              triple_list.append([head, relation, tail])
        self.test_triples = triple_list

    def insert_traning_data(self, file1, file2, file3):
        self.insert_pre_data(file1, file2)
        with codecs.open(file3, 'r') as f5:
            lines = f5.readlines()
            for line in lines:
                line = line.strip().split('\t')
                self.train_loss = json.loads(line[0])
                self.validation_loss = json.loads(line[1])
        print(self.train_loss, self.validation_loss)

    def training_run(self, out_file_title = ''):
        n_batches = int(len(self.triples) / self.batch_size)
        # valid_batch = int(len(self.valid_triples) / batch_size) + 1
        print("batch size: ", n_batches)
        for epoch in range(self.epochs):
            loop = tqdm(total=len(self.triples))
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            for batch in range(n_batches):
                batch_samples = random.sample(self.triples, self.batch_size)
                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    '''
                    这里关于p的说明 tph 表示每一个头结对应的平均尾节点数 hpt 表示每一个尾节点对应的平均头结点数
                    当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体
                    举例说明 
                    在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，
                    那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，
                    则此时我们更倾向于替换头实体，
                    因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
                    '''
                    if pr < 0.5:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]
                    current.append(sample)
                    corrupted.append(corrupted_sample)
                current = torch.from_numpy(np.array(current)).long()
                corrupted = torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)
                loop.update(self.batch_size)
            loop.close()
            # for batch in range(valid_batch):
            #     batch_samples = random.sample(self.valid_triples, batch_size)
            #     current = []
            #     corrupted = []
            #     for sample in batch_samples:
            #         corrupted_sample = copy.deepcopy(sample)
            #         pr = np.random.random(1)[0]
            #         p = relation_tph[int(corrupted_sample[1])] / (relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])
            #         if pr > p:
            #             # change the head entity
            #             corrupted_sample[0] = random.sample(self.entities, 1)[0]
            #             while corrupted_sample[0] == sample[0]:
            #                 corrupted_sample[0] = random.sample(self.entities, 1)[0]
            #         else:
            #             # change the tail entity
            #             corrupted_sample[2] = random.sample(self.entities, 1)[0]
            #             while corrupted_sample[2] == sample[2]:
            #                 corrupted_sample[2] = random.sample(self.entities, 1)[0]
            #         current.append(sample)
            #         corrupted.append(corrupted_sample)
            #     current = torch.from_numpy(np.array(current)).long()
            #     corrupted = torch.from_numpy(np.array(corrupted)).long()
            #     self.calculate_valid_loss(current, corrupted)
            end = time.time()
            mean_train_loss = self.loss / n_batches
            # mean_valid_loss = self.valid_loss / valid_batch
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("Train loss: ", mean_train_loss.cpu().detach().item())
            self.train_loss.append(float(mean_train_loss))
            # self.validation_loss.append(float(mean_valid_loss))

        entity_embedding = []
        relation_embedding = []
        for i, e in enumerate(self.model.ent_embedding.weight):
            entity_embedding.append(e.cpu().detach().numpy().tolist())
        for i, r in enumerate(self.model.rel_embedding.weight):
            relation_embedding.append(r.cpu().detach().numpy().tolist())
        return entity_embedding, relation_embedding

    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        self.loss += loss
        loss.backward()
        self.optim.step()

    def calculate_valid_loss(self, correct_sample, corrupted_sample):
        loss = self.model(correct_sample, corrupted_sample)
        self.valid_loss += loss

    def test_run(self, filter=False):
        self.filter = filter
        hits = 0
        rank_sum = 0
        num = 0
        for triple in self.test_triples:
            start = time.time()
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            #
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            head_filter = []
            tail_filter = []
            if self.filter:

                for tr in self.triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.test_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.valid_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)

            for i, entity in enumerate(self.entities):
                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(head_triple[0])
                norm_relation.append(head_triple[1])
                tail_embedding.append(head_triple[2])
                tamp.append(tuple(head_triple))
            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()
            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []
            for i, tail in enumerate(self.entities):
                tail_triple = [triple[0], triple[1], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(tail_triple[0])
                norm_relation.append(tail_triple[1])
                tail_embedding.append(tail_triple[2])
                tamp.append(tuple(tail_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()

            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]
            # itemgetter 返回一个可调用对象，该对象可以使用操作__getitem__()方法从自身的操作中捕获item
            # 使用itemgetter()从元组记录中取回特定的字段 搭配sorted可以将dictionary根据value进行排序
            # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
            '''
            sorted(iterable, cmp=None, key=None, reverse=False)
            参数说明：
            iterable -- 可迭代对象。
            cmp -- 比较的函数，这个具有两个参数，参数的值都是从可迭代对象中取出，此函数必须遵守的规则为，大于则返回1，小于则返回-1，等于则返回0。
            key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
            reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
            '''
            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)
            # calculate the mean_rank and hit_10
            # head data
            i = 0
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            # tail rank
            i = 0
            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0][2]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)), str(hits / (2 * num)),
                  str(rank_sum / (2 * num)))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))
        return self.hit_10, self.mean_rank


