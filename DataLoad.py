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
    # RippleNet
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,help='whether using outputs of all hops or just the last hop when making prediction')
    # KPRN
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')
    # ADAC
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')
    # MRNN
    parser.add_argument('--category_num', type=int, default=15, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=170, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=11969, help='单词总数')
    parser.add_argument('--title_num', type=int, default=4139, help='标题新闻总数')
    parser.add_argument('--user_clicked_new_num', type=int, default=50, help='单个用户点击的新闻个数')
    parser.add_argument('--total_word_size', type=int, default=11969, help='词袋中总的单词数量')
    parser.add_argument('--title_word_size', type=int, default=23, help='每个title中的单词数量')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--total_entity_size', type=int, default=3803, help='总实体特征个数')
    parser.add_argument('--new_entity_size', type=int, default=10, help='单个新闻最大实体个数')
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
    new_title_embedding = np.load(path + '/R&C4ERec_code/Data/metadata/new_title_embedding.npy')
    # # 单词嵌入
    # new_word_embedding = np.load(path + '/R&C4ERec_code/Data/metadata/new_title_word_embedding.npy')
    # new_word_index = np.load(path + '/R&C4ERec_code/Data/metadata/news_title_word_dict.npy')
    new_entity_index = np.load(path + '/R&C4ERec_code/Data/metadata/new_entity_index.npy')
    # 用户点击新闻
    user_clicked_newindex = np.load(path + '/R&C4ERec_code/Data/metadata/user_clicked_newindex.npy')
    # 训练集
    candidate_newindex_train = np.load(path + '/R&C4ERec_code/Data/metadata/candidate_newindex.npy')
    user_index_train = np.load(path + '/R&C4ERec_code/Data/metadata/user_index.npy')
    label_train = np.load(path + '/R&C4ERec_code/Data/metadata/label.npy')
    # 测试集
    candidate_newindex_test = np.load(path + '/R&C4ERec_code/Data/test/test_candidate_newindex.npy')
    user_index_test = np.load(path + '/R&C4ERec_code/Data/test/test_user_index.npy')
    label_test = np.load(path + '/R&C4ERec_code/Data/test/test_label.npy')
    Bound_test = np.load(path + '/R&C4ERec_code/Data/test/test_bound.npy')
    # 选择bound
    candidate_newindex_select = []
    user_index_select = []
    label_select = []
    bound_select = []
    test_size = int(len(Bound_test) * 0.3)
    train_size = int(len(Bound_test) * 0.7)

    candidate_newindex_train = candidate_newindex_train[:train_size]
    user_index_train = user_index_train[:train_size]
    label_train = label_train[:train_size]

    candidate_newindex_vaild = candidate_newindex_train[:]
    user_index_vaild = user_index_train[:]
    label_vaild = label_train[:]

    bound_test = Bound_test[train_size+1:]
    index = 0
    for i in range(len(bound_test)):
        start = bound_test[i][0]
        end = bound_test[i][1]
        temp1 = candidate_newindex_test[start:end]
        temp2 = user_index_test[start:end]
        temp3 = label_test[start:end]
        start_new = index
        end_new = index + len(temp1)
        index = index + len(temp1)
        bound_select.append([start_new,end_new])
        candidate_newindex_select.extend(temp1)
        user_index_select.extend(temp2)
        label_select.extend(temp3)
    candidate_newindex_test = np.array(candidate_newindex_select)
    user_index_test = np.array(user_index_select)
    label_test = np.array(label_select)
    bound_test = np.array(bound_select)
    print('训练集印象数{}'.format(len(label_train)))
    print('测试集印象数{}'.format(len(bound_test)))
    print('训练集点击数{}'.format(len(candidate_newindex_train)))
    print('测试集点击数{}'.format(len(candidate_newindex_test)))
    return new_title_embedding,  new_entity_index, user_clicked_newindex, \
           candidate_newindex_train, user_index_train, label_train, \
           candidate_newindex_vaild, user_index_vaild, label_vaild, \
           candidate_newindex_test, user_index_test, label_test, bound_test

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
def load_user_clicked(user_clicked_newindex):
    print('constructing user_click_dict...')
    user_clicked_newindex = user_clicked_newindex.tolist()
    # user_click_dict = {}
    # for i in range(len(user_clicked_newindex)):
    #     user_click_dict[i] = user_clicked_newindex[i]
    user_click_dict = []
    for i in range(len(user_clicked_newindex)):
        user_click_dict.append(user_clicked_newindex[i])
    return user_click_dict

def load_word_index():
    print('constructing word_index subcategory_index')
    new_title_word_index = np.load('./Data/metadata/new_title_word_index.npy')
    return new_title_word_index

def load_category_subcategory_index():
    print('constructing category_index subcategory_index')
    new_category_index = np.load('./Data/metadata/new_category_index.npy')
    new_subcategory_index = np.load('./Data/metadata/new_subcategory_index.npy')

    category_news_dict = {}
    for i in range(len(new_category_index)):
        category = new_category_index[i]
        if category not in category_news_dict.keys():
            category_news_dict[category] = []
        category_news_dict[category].append(i)


    subcategory_news_dict = {}
    for i in range(len(new_subcategory_index)):
        subcategory = new_subcategory_index[i]
        if subcategory not in subcategory_news_dict.keys():
            subcategory_news_dict[subcategory] = []
        subcategory_news_dict[subcategory].append(i)

    return new_category_index, new_subcategory_index, category_news_dict, subcategory_news_dict

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

def get_ripple_set(args, entity_adj, relation_adj, news_entity_dict, entity_news_dict, user_clicked_newindex):
    print('constructing ripple set ...')
    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = []
    for i in range(len(user_clicked_newindex)):
        ripple_set.append([])
        for h in range(args.ripplenet_n_hop + 1):
            memories_h = []
            memories_r = []
            memories_t = []
            if h == 0:
                tails_of_last_hop = user_clicked_newindex[i]
                for news in tails_of_last_hop:
                    for news_entity in news_entity_dict[news]:
                        memories_h.append(news)
                        memories_t.append(news_entity)
                        memories_r.append(0)
            elif h == args.ripplenet_n_hop:
                tails_of_last_hop = ripple_set[-1][-1][2]
                for entity in tails_of_last_hop:
                    for entity_news in entity_news_dict[entity]:
                        memories_h.append(entity)
                        memories_t.append(entity_news)
                        memories_r.append(0)
            else:
                tails_of_last_hop = ripple_set[-1][-1][2]
                for entity in tails_of_last_hop:
                    if entity in entity_adj.keys():
                        for i in range(len(entity_adj[entity])):
                            memories_h.append(entity)
                            memories_t.append(entity_adj[entity][i])
                            memories_r.append(relation_adj[entity][i])
                    else:
                        memories_h.append(entity)
                        memories_t.append(0)
                        memories_r.append(0)
            if len(memories_h) == 0:
                ripple_set[-1].append(ripple_set[i][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.ripplenet_n_memory
                indices = np.random.choice(len(memories_h), size=args.ripplenet_n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[-1].append((memories_h, memories_r, memories_t))
    return torch.IntTensor(ripple_set)

# 实体邻居新闻的平均嵌入
def build_neighbor_word_index(entity_news_dict, new_title_embedding):
    print('build neiborhood embedding ...')
    entity_neibor_embedding_list = []
    entity_neibor_num_list = []
    for i in range(len(entity_news_dict)):
        entity_neibor_embedding_list.append(np.zeros(400))
        entity_neibor_num_list.append(1)
    for key, value in entity_news_dict.items():
        entity_news_embedding_list = []
        for news_index in value:
            entity_news_embedding_list.append(new_title_embedding[news_index])
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


# def build_hit_dict(user_clicked_newindex):
#     print('build hit dict ....')
#     hit_dict = {}
#     for i in range(user_clicked_newindex.shape[0]):
#         hit_dict[i] = []
#         for clicked_news_index in user_clicked_newindex[i]:
#             hit_dict[i].append(clicked_news_index)
#     return hit_dict

def build_word_embedding(path):
    print('constructing word embedding ...')
    word_embedding = np.load(path+'/R&C4ERec_code/Data/metadata/new_title_word_embedding.npy')
    return torch.FloatTensor(word_embedding)


class Train_Dataset(Dataset):
    def __init__(self, candidate_newindex, user_clicked_newindex, user, label):
        self.user_clicked_newindex = user_clicked_newindex
        self.candidate_newindex = candidate_newindex
        self.user = user
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        candidate_newindex = self.candidate_newindex[index]
        user_index = self.user[index]
        label_index = self.label[index]
        user_clicked_newindex = self.user_clicked_newindex[user_index]
        return candidate_newindex, user_index, user_clicked_newindex, torch.Tensor(label_index)

class Vaild_Dataset(Dataset):
    def __init__(self, candidate_newindex, user_clicked_newindex, user, label):
        self.user_clicked_newindex = user_clicked_newindex
        self.candidate_newindex = candidate_newindex
        self.user = user
        self.label = label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        candidate_newindex = self.candidate_newindex[index]
        user_index = self.user[index]
        label_index = self.label[index]
        user_clicked_newindex = self.user_clicked_newindex[user_index]
        return candidate_newindex, user_index, user_clicked_newindex, torch.Tensor(label_index)

class Test_Dataset(Dataset):
    def __init__(self, candidate_newindex, user_clicked_newindex, user):
        self.user_clicked_newindex = user_clicked_newindex
        self.candidate_newindex = candidate_newindex
        self.user = user
    def __len__(self):
        return len(self.candidate_newindex)
    def __getitem__(self, index):
        candidate_newindex = self.candidate_newindex[index]
        user_index = self.user[index]
        user_clicked_newindex = self.user_clicked_newindex[user_index]
        return candidate_newindex, user_index, user_clicked_newindex

def load_data(args, path):
    new_title_embedding, news_entity_index, user_clicked_newindex, \
    candidate_newindex_train, user_index_train, label_train, \
    candidate_newindex_vaild, user_index_vaild, label_vaild, \
    candidate_newindex_test, user_index_test, label_test, bound_test = metadata_generator(path)
    entity_dict, relation_dict = load_entity_relation_dict(path)
    news_entity_dict, entity_news_dict = load_news_entity(args.title_num, entity_dict, news_entity_index)
    user_click_dict = load_user_clicked(user_clicked_newindex)
    new_title_word_index = load_word_index()
    new_category_index, new_subcategory_index, category_news_dict, subcategory_news_dict = load_category_subcategory_index()
    entity_adj, relation_adj, kg_env = build_KG_network(path, news_entity_dict)
    ripple_set = get_ripple_set(args, entity_adj, relation_adj, news_entity_dict, entity_news_dict, user_clicked_newindex)
    neibor_embedding, neibor_num = build_neighbor_word_index(entity_news_dict, new_title_embedding)
    entity_embedding, relation_embedding = build_entity_relation_embedding(path)
    # hit_dict = build_hit_dict(user_clicked_newindex)
    word_embedding = build_word_embedding(path)
    train_data = Train_Dataset(candidate_newindex_train, user_clicked_newindex, user_index_train, label_train)
    train_dataloader = DataLoader(train_data, batch_size=50, shuffle=True, drop_last = True)
    vaild_data = Vaild_Dataset(candidate_newindex_vaild, user_clicked_newindex,  user_index_vaild, label_vaild)
    vaild_dataloader = DataLoader(vaild_data, batch_size=50, shuffle=True, drop_last=True)
    test_data = Test_Dataset(candidate_newindex_test, user_clicked_newindex, user_index_test)
    test_dataloader = DataLoader(test_data, batch_size=50, shuffle=False, drop_last = True)

    return train_dataloader, test_dataloader, vaild_dataloader,\
           new_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
           new_title_word_index, new_category_index, new_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
           neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
           len(vaild_data), len(train_data), len(test_data), label_test, bound_test

if __name__ == "__main__":
    args = parse_args()
    load_data(args, path)
