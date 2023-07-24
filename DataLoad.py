import random
import collections
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import csv
import argparse
# from main import parse_args
path = os.path.dirname(os.getcwd())
print(path)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='MRNNRL')
    parser.add_argument('--epoch', type=int, default= 20)
    parser.add_argument('--user_size', type = int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--user_clicked_num', type=int, default=10)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--title_size', type=int, default=400, help='新闻初始向量维数')
    parser.add_argument('--embedding_size', type=int, default=100, help='新闻和用户向量')
    parser.add_argument('--depth', type=list, default=[5,3,2], help='K-跳深度')
    parser.add_argument('--checkpoint_dir', type=str, default='./AnchorKG_out/save_model/', help='模型保留位置')
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')
    parser.add_argument('--category_num', type=int, default=15, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=170, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=11969, help='单词总数')
    parser.add_argument('--title_num', type=int, default=4139, help='标题新闻总数')
    parser.add_argument('--user_clicked_news_num', type=int, default=50, help='单个用户点击的新闻个数')
    parser.add_argument('--total_word_size', type=int, default=11969, help='词袋中总的单词数量')
    parser.add_argument('--title_word_size', type=int, default=23, help='每个title中的单词数量')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--total_entity_size', type=int, default=3803, help='总实体特征个数')
    parser.add_argument('--news_entity_size', type=int, default=10, help='单个新闻最大实体个数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--sample_num', type=int, default=5, help='采样点数')
    parser.add_argument('--neigh_num', type=int, default=5, help='邻居节点个数')
    parser.add_argument('--category_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_dim', type=int, default=100, help='自主题总数')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    return parser.parse_args()

# 生成候选新闻和点击新闻相关元信息
def metadata_generator(path):
    # 嵌入数据
    news_title_embedding = np.load(path + '/R&C4ERec_code/Data/metadata/news_title_embedding.npy')
    # 新闻实体
    news_entity_index = np.load(path + '/R&C4ERec_code/Data/metadata/news_entity_index.npy')
    # 用户点击新闻
    user_clicked_newsindex = np.load(path + '/R&C4ERec_code/Data/metadata/user_clicked_newsindex.npy')
    # 训练集
    candidate_newsindex_train = np.load(path + '/R&C4ERec_code/Data/metadata/candidate_newsindex.npy')
    user_index_train = np.load(path + '/R&C4ERec_code/Data/metadata/user_index.npy')
    label_train = np.load(path + '/R&C4ERec_code/Data/metadata/label.npy')
    # 测试集
    candidate_newsindex_test = np.load(path + '/R&C4ERec_code/Data/test/test_candidate_newsindex.npy')
    user_index_test = np.load(path + '/R&C4ERec_code/Data/test/test_user_index.npy')
    label_test = np.load(path + '/R&C4ERec_code/Data/test/test_label.npy')
    Bound_test = np.load(path + '/R&C4ERec_code/Data/test/test_bound.npy')
    # 选择bound
    candidate_newsindex_select = []
    user_index_select = []
    label_select = []
    bound_select = []
    test_size = int(len(Bound_test) * 0.3)
    train_size = int(len(Bound_test) * 0.7)

    candidate_newsindex_train = candidate_newsindex_train[:train_size]
    user_index_train = user_index_train[:train_size]
    label_train = label_train[:train_size]

    candidate_newsindex_vaild = candidate_newsindex_train[:]
    user_index_vaild = user_index_train[:]
    label_vaild = label_train[:]

    bound_test = Bound_test[train_size+1:]
    index = 0
    for i in range(len(bound_test)):
        start = bound_test[i][0]
        end = bound_test[i][1]
        temp1 = candidate_newsindex_test[start:end]
        temp2 = user_index_test[start:end]
        temp3 = label_test[start:end]
        start_news = index
        end_news = index + len(temp1)
        index = index + len(temp1)
        bound_select.append([start_news,end_news])
        candidate_newsindex_select.extend(temp1)
        user_index_select.extend(temp2)
        label_select.extend(temp3)
    candidate_newsindex_test = np.array(candidate_newsindex_select)
    user_index_test = np.array(user_index_select)
    label_test = np.array(label_select)
    bound_test = np.array(bound_select)
    print('训练集印象数{}'.format(len(label_train)))
    print('测试集印象数{}'.format(len(bound_test)))
    print('训练集点击数{}'.format(len(candidate_newsindex_train)))
    print('测试集点击数{}'.format(len(candidate_newsindex_test)))
    return news_title_embedding,  news_entity_index, user_clicked_newsindex, \
           candidate_newsindex_train, user_index_train, label_train, \
           candidate_newsindex_vaild, user_index_vaild, label_vaild, \
           candidate_newsindex_test, user_index_test, label_test, bound_test

# 生成实体字典和关系字典
def load_entity_relation_dict(path):
    print('constructing entity_dict relation_dict ...')
    entityid2index = pd.read_csv( path + '/R&C4ERec_code/Data/KG/entityid2index.csv')
    relationid2index = pd.read_csv(path + '/R&C4ERec_code/Data/KG/relationid2index.csv')
    entity_dict = {}
    relation_dict = {}
    entity_id_list = entityid2index['entity_id'].tolist()
    entity_index_list = entityid2index['index'].tolist()
    for i in range(len(entity_id_list)):
        entity_dict[entity_id_list[i]] = entity_index_list[i]
    relation_id_list = relationid2index['relation_id'].tolist()
    relation_index_list = relationid2index['index'].tolist()
    for i in range(len(relation_id_list)):
        relation_dict[relation_id_list[i]] = relation_index_list[i]
    return entity_dict, relation_dict

# 生成新闻实体字典和实体新闻字典
def load_news_entity(news_num, entity_dict, news_entity_index):
    print('constructing news_entity_dict entity_news_dict ...')
    news_entity_index_list = news_entity_index.tolist()
    news_entity_dict = {}
    entity_news_dict = {}
    for i in range(len(news_entity_index_list)):
        news_entity_dict[i] = []
        for entity_index in news_entity_index_list[i][:20]:
            news_entity_dict[i].append(entity_index)
            if entity_index not in entity_news_dict:
                entity_news_dict[entity_index] = []
            entity_news_dict[entity_index].append(i)
    # 实体新闻字典去重
    for item in entity_news_dict:
        entity_news_dict[item] = list(set(entity_news_dict[item]))
    for item in entity_news_dict:
        if len(entity_news_dict[item]) > 20:
            entity_news_dict[item] = entity_news_dict[item][:20]
    # 填充
    for entity in entity_dict.values():
        if entity not in entity_news_dict.keys():
            entity_news_dict[entity] = [news_num - 1]
    return news_entity_dict, entity_news_dict

# 生成用户点击新闻字典
def load_user_clicked(user_clicked_newsindex):
    print('constructing user_click_dict...')
    user_clicked_newsindex = user_clicked_newsindex.tolist()
    # user_click_dict = {}
    # for i in range(len(user_clicked_newsindex)):
    #     user_click_dict[i] = user_clicked_newsindex[i]
    user_click_dict = []
    for i in range(len(user_clicked_newsindex)):
        user_click_dict.append(user_clicked_newsindex[i])
    return user_click_dict

def load_word_index():
    print('constructing word_index subcategory_index')
    news_title_word_index = np.load('./Data/metadata/news_title_word_index.npy')
    return news_title_word_index

def load_category_subcategory_index():
    print('constructing category_index subcategory_index')
    news_category_index = np.load('./Data/metadata/news_category_index.npy')
    news_subcategory_index = np.load('./Data/metadata/news_subcategory_index.npy')

    category_news_dict = {}
    for i in range(len(news_category_index)):
        category = news_category_index[i]
        if category not in category_news_dict.keys():
            category_news_dict[category] = []
        category_news_dict[category].append(i)


    subcategory_news_dict = {}
    for i in range(len(news_subcategory_index)):
        subcategory = news_subcategory_index[i]
        if subcategory not in subcategory_news_dict.keys():
            subcategory_news_dict[subcategory] = []
        subcategory_news_dict[subcategory].append(i)

    return news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict

def build_KG_network(path, news_entity_dict):
    print('constructing adjacency matrix ...')
    # 将新闻实体加入KG
    network = nx.DiGraph()
    # index = 1
    for key, value in news_entity_dict.items():
        # index  += 1
        # if index > 10:
        #     break
        newsid = 'news' + str(key)
        for entity in value:
            if entity != 0:
                network.add_edge(newsid, entity, label="innews", weight = 0)
                network.add_edge(entity, newsid, label="outnews", weight = 0)
    # plt.plot()
    # nx.draw_networkx(network,  with_labels=True)
    # plt.show()
    # network.add_edge(newsid, 0, label="innews", weight=0)

    graph = pd.read_csv(path + '/R&C4ERec_code/Data/KG/graph_index.csv')
    head_entity_list = graph['h_index'].tolist()
    relation_list = graph['r_index'].tolist()
    tail_entity_list = graph['t_idnex'].tolist()
    adj = {}

    for i in range(graph.shape[0]):
        head_entity_index = head_entity_list[i]
        relation_index = relation_list[i]
        tail_entity_index = tail_entity_list[i]
        if head_entity_index not in adj:
            adj[head_entity_index] = []
        adj[head_entity_index].append((tail_entity_index, relation_index))
        network.add_edge(head_entity_index, tail_entity_index, label=relation_index, weight=relation_index)
        network.add_edge(tail_entity_index, head_entity_index, label=relation_index, weight=relation_index)

    for key, value in adj.items():
        if len(value) < 20:
            need_len = 20 - len(value)
            for i in range(need_len):
                value.append((0, 0))
        elif len(value) > 20:
            adj[key] = value[:20]
    adj_entity = {}
    adj_entity[0] = list(map(lambda x: int(x), np.zeros(20)))
    for item in adj:
        adj_entity[item] = list(map(lambda x: x[0], adj[item]))
    adj_relation = {}
    adj_relation[0] = list(map(lambda x: int(x), np.zeros(20)))
    for item in adj:
        adj_relation[item] = list(map(lambda x: x[1], adj[item]))
    return adj_entity, adj_relation, network

# 实体邻居新闻的平均嵌入
def build_neighbor_word_index(entity_news_dict, news_title_embedding):
    print('build neiborhood embedding ...')
    entity_neibor_embedding_list = []
    entity_neibor_num_list = []
    for i in range(len(entity_news_dict)):
        entity_neibor_embedding_list.append(np.zeros(400))
        entity_neibor_num_list.append(1)
    for key, value in entity_news_dict.items():
        entity_news_embedding_list = []
        for news_index in value:
            entity_news_embedding_list.append(news_title_embedding[news_index])
        entity_neibor_embedding_list[key] = np.sum(entity_news_embedding_list, axis=0)
        if len(value) >= 2:
            entity_neibor_num_list[key] = len(value) - 1
    return torch.FloatTensor(np.array(entity_neibor_embedding_list)), torch.FloatTensor(np.array(entity_neibor_num_list))

def build_entity_relation_embedding(path):
    print('constructing entity and relation embedding ...')
    TransE_entity_embedding = np.load(path+'/R&C4ERec_code/Data/KG/TransE_entity_embedding.npy')
    TransE_relation_embedding = np.load(path + '/R&C4ERec_code/Data/KG/TransE_relation_embedding.npy')
    TransE_relation_embedding = np.delete(TransE_relation_embedding, 0, axis = 0)
    TransE_relation_embedding = np.concatenate(([np.random.normal(-0.1, 0.1, 100)], TransE_relation_embedding), axis=0)
    return torch.FloatTensor(TransE_entity_embedding), torch.FloatTensor(TransE_relation_embedding)

def build_word_embedding(path):
    print('constructing word embedding ...')
    word_embedding = np.load(path+'/R&C4ERec_code/Data/metadata/news_title_word_embedding.npy')
    return torch.FloatTensor(word_embedding)

class Train_Dataset(Dataset):
    def __init__(self, candidate_newsindex, user_clicked_newsindex, user, label):
        self.user_clicked_newsindex = user_clicked_newsindex
        self.candidate_newsindex = candidate_newsindex
        self.user = user
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        candidate_newsindex = self.candidate_newsindex[index]
        user_index = self.user[index]
        label_index = self.label[index]
        user_clicked_newsindex = self.user_clicked_newsindex[user_index]
        return candidate_newsindex, user_index, user_clicked_newsindex, torch.Tensor(label_index)

class Vaild_Dataset(Dataset):
    def __init__(self, candidate_newsindex, user_clicked_newsindex, user, label):
        self.user_clicked_newsindex = user_clicked_newsindex
        self.candidate_newsindex = candidate_newsindex
        self.user = user
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        candidate_newsindex = self.candidate_newsindex[index]
        user_index = self.user[index]
        label_index = self.label[index]
        user_clicked_newsindex = self.user_clicked_newsindex[user_index]
        return candidate_newsindex, user_index, user_clicked_newsindex, torch.Tensor(label_index)

class Test_Dataset(Dataset):
    def __init__(self, candidate_newsindex, user_clicked_newsindex, user):
        self.user_clicked_newsindex = user_clicked_newsindex
        self.candidate_newsindex = candidate_newsindex
        self.user = user
    def __len__(self):
        return len(self.candidate_newsindex)
    def __getitem__(self, index):
        candidate_newsindex = self.candidate_newsindex[index]
        user_index = self.user[index]
        user_clicked_newsindex = self.user_clicked_newsindex[user_index]
        return candidate_newsindex, user_index, user_clicked_newsindex

def load_data(args, path):
    news_title_embedding, news_entity_index, user_clicked_newsindex, \
    candidate_newsindex_train, user_index_train, label_train, \
    candidate_newsindex_vaild, user_index_vaild, label_vaild, \
    candidate_newsindex_test, user_index_test, label_test, bound_test = metadata_generator(path)
    entity_dict, relation_dict = load_entity_relation_dict(path)
    news_entity_dict, entity_news_dict = load_news_entity(args.title_num, entity_dict, news_entity_index)
    user_click_dict = load_user_clicked(user_clicked_newsindex)
    news_title_word_index = load_word_index()
    news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict = load_category_subcategory_index()
    entity_adj, relation_adj, kg_env = build_KG_network(path, news_entity_dict)
    neibor_embedding, neibor_num = build_neighbor_word_index(entity_news_dict, news_title_embedding)
    entity_embedding, relation_embedding = build_entity_relation_embedding(path)
    # hit_dict = build_hit_dict(user_clicked_newsindex)
    word_embedding = build_word_embedding(path)
    train_data = Train_Dataset(candidate_newsindex_train, user_clicked_newsindex, user_index_train, label_train)
    train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, drop_last = True)
    vaild_data = Vaild_Dataset(candidate_newsindex_vaild, user_clicked_newsindex,  user_index_vaild, label_vaild)
    vaild_dataloader = DataLoader(vaild_data, batch_size=50, shuffle=True, drop_last=True)
    test_data = Test_Dataset(candidate_newsindex_test, user_clicked_newsindex, user_index_test)
    test_dataloader = DataLoader(test_data, batch_size=50, shuffle=False, drop_last = True)

    return train_dataloader, test_dataloader, vaild_dataloader,\
           news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, \
           news_entity_dict, entity_news_dict, user_click_dict, \
           news_title_word_index, news_category_index, news_subcategory_index, \
           category_news_dict, subcategory_news_dict, word_embedding, \
           neibor_embedding, neibor_num, entity_embedding, relation_embedding, \
           len(vaild_data), len(train_data), len(test_data), label_test, bound_test

if __name__ == "__main__":
    args = parse_args()
    load_data(args, path)
