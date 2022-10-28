import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import groupby

class Reasoner(torch.nn.Module):
    def __init__(self, args, kg_env_all, entity_embedding, relation_embedding, news_title_embedding, device):
        super(Reasoner, self).__init__()
        self.args = args
        self.device = device
        self.kg_env_all = kg_env_all
        self.tanh = nn.Tanh()
        self.gru = torch.nn.GRU(self.args.embedding_size, self.args.embedding_size)

        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size)
        self.type_embedding = nn.Embedding(5, self.args.embedding_size)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        # new_embedding = news_title_embedding.tolist()
        # new_embedding.append(np.array([0 for i in range(768)]))
        self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(news_title_embedding)))

        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.gru_output_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.gru_output_layer2 = nn.Linear(self.args.embedding_size, 1)
        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        trans_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_graph_list(self, graph_layers, graph_type, batch_size):
        graph_list_flat = []
        graph_list = []
        graph_type_list_flat = []
        graph_type_list = []
        for i in range(batch_size):
            graph_list_flat.append([])
            graph_list.append([[],[],[]])
            graph_type_list_flat.append([])
            graph_type_list.append([[], [], []])

        for i in range(len(graph_layers)):
            for j in range(len(graph_layers[i])):
                for k in range(len(graph_layers[i][j])):
                    graph_list[j][i].append(int(graph_layers[i][j][k].data.cpu().numpy()))
                    graph_list_flat[j].append(int(graph_layers[i][j][k].data.cpu().numpy()))

                    graph_type_list[j][i].append(int(graph_type[i][j][k].data.cpu().numpy()))
                    graph_type_list_flat[j].append(int(graph_type[i][j][k].data.cpu().numpy()))
        return graph_list_flat, graph_list, graph_type_list_flat, graph_type_list

    def get_overlap_entities(self, news_graph, news_graph_type, user_graph, user_graph_type):
        overlap_entity = []
        overlap_entity_num = []
        news_graph_num = []
        user_graph_num = []
        for i in range(len(news_graph)):
            news_graph_set = set()
            user_graph_set = set()
            for m in range(len(news_graph[i])):
                if news_graph[i][m] != 0 :
                    news_graph_set.add((int(news_graph_type[i][m]) ,int(news_graph[i][m])))
            for j in range(1, len(user_graph[i])):
                if user_graph[i][j] != 0:
                    node = user_graph[i][j]
                    user_graph_set.add((int(user_graph_type[i][j]) ,int(node)))
            if len(news_graph_set & user_graph_set) == 0:
                overlap_entity.append([0])
            else:
                overlap_entity.append(list(news_graph_set & user_graph_set))
            overlap_entity_num.append(len(news_graph_set & user_graph_set))
            news_graph_num.append(len(news_graph_set))
            user_graph_num.append(len(user_graph_set))
        overlap_entity_num_cpu = overlap_entity_num
        return torch.tensor(overlap_entity_num).to(self.device), torch.tensor(news_graph_num).to(self.device),\
               torch.tensor(user_graph_num).to(self.device), overlap_entity_num_cpu, overlap_entity

    def get_reasoning_paths(self, candidate_news, user_index, news_graph, user_graph,
                            news_graph_relation,  user_graph_relation,
                            news_graph_type_list, user_graph_type_list,
                            overlap_entity_num_cpu):
        reasoning_paths = []
        reasoning_edges = []
        path_num = 0
        # print('--------')
        for i in range(len(candidate_news)):
            reasoning_paths.append([])
            reasoning_edges.append([])
            #print('候选新闻：{}, 用户：{}'.format('news' + str(candidate_news[i]), 'user' + str(user_index[i])))
            # print(str(user_index[i].item()))
            if overlap_entity_num_cpu[i] > 0:
                # print(overlap_entity_num_cpu[i])
                # print(overlap_entity[i])
                subgraph = nx.Graph()
                subgraph.add_node('news' + str(candidate_news[i].item()))
                subgraph.add_node('user' + str(user_index[i].item()))

                for index1 in range(self.args.depth[0]):
                    if news_graph[i][0][index1] != 0:
                        if news_graph_type_list[i][0][index1] == 0:
                            type_news_1 = 'user'
                        elif news_graph_type_list[i][0][index1] == 1:
                            type_news_1 = 'news'
                        elif news_graph_type_list[i][0][index1] == 2:
                            type_news_1 = 'entity'
                        elif news_graph_type_list[i][0][index1] == 3:
                            type_news_1 = 'category'
                        elif news_graph_type_list[i][0][index1] == 4:
                            type_news_1 = 'subcategory'
                        subgraph.add_edge('news' + str(candidate_news[i].item()),
                                          type_news_1 + str(news_graph[i][0][index1]),
                                          weight=0)

                        for index2 in range(self.args.depth[1]):
                            drump_1 = index1 * self.args.depth[1]
                            if news_graph[i][1][drump_1 + index2] != 0:
                                if news_graph_type_list[i][0][index1] == 0:
                                    type_news_2 = 'user'
                                elif news_graph_type_list[i][0][index1] == 1:
                                    type_news_2 = 'news'
                                elif news_graph_type_list[i][0][index1] == 2:
                                    type_news_2 = 'entity'
                                elif news_graph_type_list[i][0][index1] == 3:
                                    type_news_2 = 'category'
                                elif news_graph_type_list[i][0][index1] == 4:
                                    type_news_2 = 'subcategory'
                                subgraph.add_edge(type_news_1 + str(news_graph[i][0][index1]),
                                                  type_news_2 + str(news_graph[i][1][drump_1 + index2]),
                                                  weight=news_graph_relation[1][i][drump_1 + index2])

                                for index3 in range(self.args.depth[2]):
                                    drump_2 = index1 * self.args.depth[1] * self.args.depth[2] + index2 * \
                                              self.args.depth[2]
                                    if news_graph[i][2][drump_2 + index3] != 0:
                                        if news_graph_type_list[i][2][drump_2 + index3] == 0:
                                            type_news_3 = 'user'
                                        elif news_graph_type_list[i][2][drump_2 + index3] == 1:
                                            type_news_3 = 'news'
                                        elif news_graph_type_list[i][2][drump_2 + index3] == 2:
                                            type_news_3 = 'entity'
                                        elif news_graph_type_list[i][2][drump_2 + index3] == 3:
                                            type_news_3 = 'category'
                                        elif news_graph_type_list[i][2][drump_2 + index3] == 4:
                                            type_news_3 = 'subcategory'
                                        subgraph.add_edge(type_news_2 + str(news_graph[i][1][drump_1 + index2]),
                                                          type_news_3 + str(news_graph[i][2][drump_2 + index3]),
                                                          weight=news_graph_relation[2][i][drump_2 + index3])
                for index1 in range(self.args.depth[0]):
                    if user_graph[i][0][index1] != 0:
                        subgraph.add_edge('user' + str(user_index[i].item()), 'news' + str(user_graph[i][0][index1]), weight=1)
                        for index2 in range(self.args.depth[1]):
                            drump_1 = index1 * self.args.depth[1]
                            if user_graph[i][1][drump_1 + index2] != 0:
                                if user_graph_type_list[i][1][drump_1 + index2] == 0:
                                    type_user_2 = 'user'
                                elif user_graph_type_list[i][1][drump_1 + index2] == 1:
                                    type_user_2 = 'news'
                                elif user_graph_type_list[i][1][drump_1 + index2] == 2:
                                    type_user_2 = 'entity'
                                elif user_graph_type_list[i][1][drump_1 + index2] == 3:
                                    type_user_2 = 'category'
                                elif user_graph_type_list[i][1][drump_1 + index2] == 4:
                                    type_user_2 = 'subcategory'
                                subgraph.add_edge('news' + str(user_graph[i][0][index1]),
                                                  type_user_2 + str(user_graph[i][1][drump_1 + index2]),
                                                  weight=user_graph_relation[1][i][drump_1 + index2])

                                for index3 in range(self.args.depth[2]):
                                    drump_2 = index1 * self.args.depth[1] * self.args.depth[2] + index2 * self.args.depth[2]

                                    if user_graph[i][2][drump_2 + index3] != 0:
                                        if user_graph_type_list[i][2][drump_2 + index3] == 0:
                                            type_user_3 = 'user'
                                        elif user_graph_type_list[i][2][drump_2 + index3] == 1:
                                            type_user_3 = 'news'
                                        elif user_graph_type_list[i][2][drump_2 + index3] == 2:
                                            type_user_3 = 'entity'
                                        elif user_graph_type_list[i][2][drump_2 + index3] == 3:
                                            type_user_3 = 'category'
                                        elif user_graph_type_list[i][2][drump_2 + index3] == 4:
                                            type_user_3 = 'subcategory'
                                        subgraph.add_edge(type_user_2 + str(user_graph[i][1][drump_1 + index2]),
                                                          type_user_3 + str(user_graph[i][2][drump_2 + index3]),
                                                          weight=user_graph_relation[2][i][drump_2+ index3])
                print(subgraph)
                nx.draw(subgraph, node_size=300, with_labels=True, node_color='r')
                plt.show()
                plt.close()

                for path in nx.all_simple_paths(subgraph, source='user' + str(user_index[i].item()), target='news' + str(candidate_news[i].item()), cutoff=5):
                    # print(len(path))
                    # print(path)
                    # print(path[1:-1])
                    path_num += 1
                    reasoning_paths[-1].append(path)
                    reasoning_edges[-1].append([])
                    for j in range(len(path)- 1):
                        reasoning_edges[-1][-1].append(int(subgraph[path[j]][path[j+1]]['weight']))
                    reasoning_edges[-1][-1].append(int(0))
                if len(reasoning_paths[-1]) == 0:
                    reasoning_paths[-1].append(['user' + str(user_index[i].item()), 'entity' + str(0), 'news' + str(candidate_news[i].item())])
                    reasoning_edges[-1].append([0, 0, 0])
            else:
                reasoning_paths[-1].append(['user' + str(user_index[i].item()), 'entity' + str(0), 'news' + str(candidate_news[i].item())])
                reasoning_edges[-1].append([0, 0, 0])
            # print(reasoning_paths[-1])
            # print(reasoning_edges[-1])
        print('学习到的推理路径数：{}'.format(path_num))
        return reasoning_paths, reasoning_edges


    def get_path_score(self, reasoning_paths):
        predict_scores = []
        for paths in reasoning_paths:
            predict_scores.append([0])
            for path in paths:
                path_node_embeddings = self.entity_embedding(torch.tensor(path).to(self.device))
                if len(path_node_embeddings.shape) == 1:
                    path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)
                path_node_embeddings = torch.unsqueeze(path_node_embeddings, 1)
                output, h_n = self.gru(path_node_embeddings)
                path_score = (self.gru_output_layer2(self.elu(self.gru_output_layer1(torch.squeeze(output[-1])))))
                predict_scores[-1].append(path_score)
            predict_scores[-1] = torch.sum(torch.tensor(predict_scores[-1]).to(self.device)).float()
        return torch.stack(predict_scores).cuda()

    def Split_num_letters(self, astr):
        nums, letters = "", ""
        for i in astr:
            if i.isdigit():
                nums = nums + i
            elif i.isspace():
                pass
            else:
                letters = letters + i
        return nums, letters



    def forward(self, candidate_news, user_index,  news_graph, user_graph,
                news_graph_relation, user_graph_relation,
                news_graph_type, user_graph_type):

        candidate_news = torch.flatten(candidate_news, 0, 1)
        # print(candidate_news)

        pt = user_index.unsqueeze(-1)
        for i in range(5):
            if i == 0:
                new_pt = pt
            else:
                new_pt = torch.cat([new_pt, pt], dim=-1)
        user_index = torch.flatten(new_pt, 0, 1).to(self.device)


        # user_index = torch.flatten(user_index.unsqueeze(1).repeat(1, 5, 1), 0, 1)
        # print(user_index.unsqueeze(1).repeat(1, 5, 1))
        news_graph_list_flat, news_graph_list, \
        news_graph_type_list_flat, news_graph_type_list = self.get_graph_list(news_graph, news_graph_type, len(candidate_news)) # bz,d(1-hop)*d(2-hop)*d(3-hop); # bz, 3, d(1-hop) + d(2-hop) + d(3-hop)
        user_graph_list_flat, user_graph_list, \
        user_graph_type_list_flat, user_graph_type_list= self.get_graph_list(user_graph, user_graph_type,  len(user_index))

        overlap_entity_num, news_graph_num, user_graph_graph_num, \
        overlap_entity_num_cpu, overlap_entity = self.get_overlap_entities(news_graph_list_flat, news_graph_type_list_flat,
                                                                           user_graph_list_flat, user_graph_type_list_flat)
        reasoning_paths, reasoning_edges = self.get_reasoning_paths(candidate_news, user_index, news_graph_list, user_graph_list,
                                                                    news_graph_relation, user_graph_relation,
                                                                    news_graph_type_list, user_graph_type_list,
                                                                    overlap_entity_num_cpu)
        predict_scores = []
        #print('------')
        for i in range(len(reasoning_paths)):
            paths = reasoning_paths[i]
            edges = reasoning_edges[i]
            predict_scores.append([])
            for j in range(len(paths)):
                if len(paths[j]) != 1:
                    path_node_embeddings_list = []
                    path_node_type_embeddings_list = []
                    for m in range(len(paths[j])):
                        index, type_index = self.Split_num_letters(paths[j][m])
                        if type_index == 'user':
                            path_node_embeddings_list.append(self.user_embedding(torch.tensor(int(index)).to(self.device)))
                            path_node_type_embeddings_list.append(self.type_embedding(torch.tensor(int(0)).to(self.device)))
                        elif type_index == 'news':
                            path_node_embeddings_list.append(self.trans_news_embedding(torch.tensor(int(index)).to(self.device)))
                            path_node_type_embeddings_list.append(self.type_embedding(torch.tensor(int(1)).to(self.device)))
                        elif type_index == 'entity':
                            path_node_embeddings_list.append(self.entity_embedding(torch.tensor(int(index)).to(self.device)))
                            path_node_type_embeddings_list.append(self.type_embedding(torch.tensor(int(2)).to(self.device)))
                        elif type_index == 'category':
                            path_node_embeddings_list.append(self.category_embedding(torch.tensor(int(index)).to(self.device)))
                            path_node_type_embeddings_list.append(self.type_embedding(torch.tensor(int(3)).to(self.device)))
                        elif type_index == 'subcategory':
                            path_node_embeddings_list.append(self.subcategory_embedding(torch.tensor(int(index)).to(self.device)))
                            path_node_type_embeddings_list.append(self.type_embedding(torch.tensor(int(4)).to(self.device)))
                    path_node_embeddings = torch.stack(path_node_embeddings_list).to(self.device)
                    path_node_type_embeddings = torch.stack(path_node_embeddings_list).to(self.device)
                    # print(path_node_embeddings.shape)
                    # print(path_node_type_embeddings.shape)
                else:

                    index, type_index = self.Split_num_letters(paths[j][0])
                    if type_index == 'user':
                        path_node_embeddings = self.user_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                        path_node_type_embeddings = self.type_embedding(torch.tensor(int(0)).to(self.device)).unsqueeze(0)
                    elif type_index == 'news':
                        path_node_embeddings = self.trans_news_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                        path_node_type_embeddings = self.type_embedding(torch.tensor(int(1)).to(self.device)).unsqueeze(0)
                    elif type_index == 'entity':
                        path_node_embeddings = self.entity_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                        path_node_type_embeddings = self.type_embedding(torch.tensor(int(2)).to(self.device)).unsqueeze(0)
                    elif type_index == 'category':
                        path_node_embeddings = self.category_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                        path_node_type_embeddings = self.type_embedding(torch.tensor(int(3)).to(self.device)).unsqueeze(0)
                    elif type_index == 'subcategory':
                        path_node_embeddings = self.subcategory_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                        path_node_type_embeddings = self.type_embedding(torch.tensor(int(4)).to(self.device)).unsqueeze(0)

                    # print(path_node_embeddings.shape)
                    # print(path_node_type_embeddings.shape)
                path_edge_embeddings = self.relation_embedding(torch.tensor(edges[j]).to(self.device)).to(self.device) # dim

                if len(path_node_embeddings.shape) == 1:
                    path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)
                    path_node_type_embeddings = torch.unsqueeze(path_node_type_embeddings, 0)
                    path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 0)
                path_node_embeddings = torch.unsqueeze(path_node_embeddings, 1)  # 1, 1, dim
                path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 1)  # 1, 1, dim
                path_node_type_embeddings = torch.unsqueeze(path_node_type_embeddings, 0)

                output, _ = self.gru(path_node_embeddings + path_edge_embeddings)
                path_score = self.sigmoid(self.gru_output_layer2(self.elu(self.gru_output_layer1(torch.squeeze(output[-1])))))
                predict_scores[-1].append(path_score.float())
            predict_scores[-1] = torch.sum(torch.tensor(predict_scores[-1]).to(self.device)).float()

        # 计算路径得分 和 路径重叠实体
        paths_predict_socre = torch.stack(predict_scores).to(self.device)
        reason_qua = self.tanh(torch.div(paths_predict_socre, (torch.log((np.e + news_graph_num+user_graph_graph_num).float()))))
        reason_num = self.tanh(torch.div(overlap_entity_num, (torch.log((np.e + news_graph_num+user_graph_graph_num).float()))))
        predicts = 0.5 * reason_qua + 0.5 * reason_num
        predicts = predicts.reshape(self.args.batch_size, self.args.sample_size)

        return predicts, reason_num.reshape(self.args.batch_size, self.args.sample_size), reasoning_paths, reasoning_edges, predict_scores
