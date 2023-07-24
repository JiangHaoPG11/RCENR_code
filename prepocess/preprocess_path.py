import networkx as nx
import torch
import numpy as np
import pandas as pd
import argparse

# 构建新闻——>实体字典
def load_news_entity(news_entity_index):
    print('constructing news_entity_dict entity_news_dict ...')
    news_entity_index_list = news_entity_index.tolist()
    news_entity_dict = {}
    for i in range(len(news_entity_index_list)):
        news_entity_dict[i] = []
        for entity_index in news_entity_index_list[i]:
            news_entity_dict[i].append(entity_index)
    return news_entity_dict

# 构建KG的环境（加入了新闻节点和新闻实体）
def build_KG_network(news_entity_dict, user_clicked_newsindex):
    print('constructing kg env ...')
    print('adding news to KG')
    network = nx.DiGraph()
    for key, value in news_entity_dict.items():
        newsid = 'news' + str(key)
        for entity in value:
            if entity != 0:
                network.add_edge(newsid, entity, label = "innews", weight = 0)
                network.add_edge(entity, newsid, label = "outnews", weight = 0)

    print('adding users to KG')
    for i in range(user_clicked_newsindex.shape[0]):
        single_user_clicked = user_clicked_newsindex[i, :]
        for clicked_index in single_user_clicked:
            network.add_edge('user' + str(i), 'news' + str(clicked_index.item()), label = "inuser", weight = 1)
            network.add_edge('news' + str(clicked_index.item()), 'user' + str(i), label = "outuser", weight = 1)

    print('adding entity to KG')
    network.add_edge(0, 0, weight=0)
    graph = pd.read_csv('../Data/KG/graph_index.csv')
    head_entity_list = graph['h_index'].tolist()
    relation_list = graph['r_index'].tolist()
    tail_entity_list = graph['t_idnex'].tolist()
    for i in range(graph.shape[0]):
        head_entity_index = head_entity_list[i]
        relation_index = relation_list[i]
        tail_entity_index = tail_entity_list[i]
        network.add_edge(head_entity_index, tail_entity_index, label = relation_index, weight = relation_index)
        network.add_edge(tail_entity_index, head_entity_index, label = relation_index, weight = relation_index)
    return network

def bulid_news_network(network):
    print('bulid news network')
    news_category_index = np.load('../Data/metadata/news_category_index.npy')
    news_subcategory_index = np.load('../Data/metadata/news_subcategory_index.npy')
    news_category_index = news_category_index.tolist()
    news_subcategory_index = news_subcategory_index.tolist()
    for i in range(len(news_category_index)):
        newsid = 'news' + str(i)
        network.add_edge(newsid, 'category' + str(news_category_index[i]), label="mention", weight = 0)
        network.add_edge(newsid, 'subcategory' + str(news_subcategory_index[i]), label="mention", weight=0)
    return network


# 获取KPRN模型当中的用户——>点击新闻——>候选新闻的路径
def kprn_get_paths(args, kg_env, user_index, candidate_news):
    print('constructing KPRN paths ...')
    total_paths = []
    total_relations = []
    total_paths_index = []
    total_relations_index = []
    total_type_index = []

    for m in range(candidate_news.shape[1]):
        paths_index = []
        relations_index = []
        type_index = []
        for j in range(candidate_news.shape[0]):
            total_paths.append([])
            total_relations.append([])
            index = 0
            for path in nx.all_simple_paths(kg_env, source='user' + str(user_index[j].item()), target='news' + str(candidate_news[j, m].item()), cutoff= args.kprn_max_path):
                index += 1
                if index > args.kprn_max_path:
                    break
                total_paths[-1].append(path)
                total_relations[-1].append([0])
                for i in range(len(path) - 1):
                    total_relations[-1][-1].append(int(kg_env[path[i]][path[i + 1]]['weight']))
                if len(path) < args.kprn_path_long:
                    for i in range(args.kprn_path_long - len(path)):
                        total_paths[-1][-1].insert(-2, 0)
                        total_relations[-1][-1].insert(-2, 0)

            if len(total_paths[-1]) < args.kprn_max_path:
                for i in range(args.kprn_max_path - len(total_paths[-1])):
                    total_paths[-1].append(
                        ['user' + str(user_index[j].item()), 0, 0, 0, 0, 'news' + str(candidate_news[j, m].item())])
                    total_relations[-1].append([0, 0, 0, 0, 0, 0])

            paths_index.append([])
            relations_index.append([])
            type_index.append([])
            for a in range(len(total_paths[-1])):
                paths_index[-1].append([])
                relations_index[-1].append(total_relations[-1][a])
                type_index[-1].append([])
                single_total_paths = total_paths[-1][a]
                for path_node in single_total_paths:
                    if type(path_node) == str:
                        path_node_type = path_node[:4]
                        if path_node_type == 'user':
                            path_node_type_index = 0
                        else:
                            path_node_type_index = 1
                        path_node_index = int(path_node[4:])
                    if type(path_node) == int:
                        path_node_type_index = 2
                        path_node_index = int(path_node)
                    paths_index[-1][-1].append(path_node_index)
                    type_index[-1][-1].append(path_node_type_index)
        total_paths_index.append(paths_index)
        total_relations_index.append(relations_index)
        total_type_index.append(type_index)
    np.save('../Data/pathdata/kprn_path_index.npy', np.array(total_paths_index))
    np.save('../Data/pathdata/kprn_relations_index.npy', np.array(total_relations_index))
    np.save('../Data/pathdata/kprn_type_index.npy', np.array(total_type_index))
    # total_paths_index = torch.transpose(torch.IntTensor(total_paths_index), 1, 0)
    # total_relations_index = torch.transpose(torch.IntTensor(total_relations_index), 1, 0)
    # total_type_index = torch.transpose(torch.IntTensor(total_type_index), 1, 0)


# 获取ADAC模型当中的用户——>候选新闻的路径
def adac_get_paths(args, kg_env, user_index, ):
    print('constructing ADAC paths ...')
    total_paths = []
    total_relations = []
    total_paths_index = []
    total_relations_index = []
    total_type_index = []

    for m in range(user_clicked_newsindex.shape[1]):
        paths_index = []
        relations_index = []
        type_index = []
        for j in range(user_clicked_newsindex.shape[0]):
            total_paths.append([])
            total_relations.append([])
            index = 0
            for path in nx.all_simple_paths(kg_env, source='user' + str(user_index[j].item()), target='news' + str(user_clicked_newsindex[j, m].item()), cutoff= args.ADAC_path_long):
                index += 1
                if index > args.ADAC_max_path:
                    break
                total_paths[-1].append(path)
                total_relations[-1].append([0])
                for i in range(len(path) - 1):
                    total_relations[-1][-1].append(int(kg_env[path[i]][path[i + 1]]['weight']))
                if len(path) < args.ADAC_path_long + 1:
                    for i in range(args.ADAC_path_long + 1 - len(path)):
                        total_paths[-1][-1].insert(-1, 0)
                        total_relations[-1][-1].insert(-1, 0)

            if len(total_paths[-1]) < args.ADAC_max_path:
                for i in range(args.ADAC_max_path - len(total_paths[-1])):
                    total_paths[-1].append(
                        ['user' + str(user_index[j].item()), 0, 0, 0, 0, 'news' + str(user_clicked_newsindex[j, m].item())])
                    total_relations[-1].append([0, 0, 0, 0, 0, 0])

            paths_index.append([])
            relations_index.append([])
            type_index.append([])
            for a in range(len(total_paths[-1])):
                paths_index[-1].append([])
                relations_index[-1].append(total_relations[-1][a])
                type_index[-1].append([])
                single_total_paths = total_paths[-1][a]
                for path_node in single_total_paths:
                    if type(path_node) == str:
                        path_node_type = path_node[:4]
                        if path_node_type == 'user':
                            path_node_type_index = 0
                        else:
                            path_node_type_index = 1
                        path_node_index = int(path_node[4:])
                    if type(path_node) == int:
                        path_node_type_index = 2
                        path_node_index = int(path_node)
                    paths_index[-1][-1].append(path_node_index)
                    type_index[-1][-1].append(path_node_type_index)
        total_paths_index.append(paths_index)
        total_relations_index.append(relations_index)
        total_type_index.append(type_index)
    np.save('../Data/pathdata/adac_path_index.npy', np.array(total_paths_index))
    np.save('../Data/pathdata/adac_relations_index.npy', np.array(total_relations_index))
    np.save('../Data/pathdata/adac_type_index.npy', np.array(total_type_index))

def parse_args():
    parser = argparse.ArgumentParser()
    # KPRN
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')
    # ADAC
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')
    parser.add_argument('--ADAC_max_path', type=int, default=1, help='每个用户项目对的最大个数')
    return parser.parse_args()


if __name__ == "__main__":
    news_entity_index = np.load('../Data/metadata/news_entity_index.npy')
    user_index = np.load('../Data/metadata/user_index.npy')
    candidate_news = np.load('../Data/metadata/candidate_newsindex.npy')
    user_clicked_newsindex = np.load('../Data/metadata/user_clicked_newsindex.npy')
    args = parse_args()
    news_entity_dict = load_news_entity(news_entity_index)
    kg_env = build_KG_network(news_entity_dict, user_clicked_newsindex)
    news_network = bulid_news_network(kg_env)
    kprn_get_paths(args, kg_env, user_index, candidate_news)
    adac_get_paths(args, kg_env, user_index)
