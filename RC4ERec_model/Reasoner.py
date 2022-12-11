import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import groupby

class Reasoner(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_title_embedding, device):
        super(Reasoner, self).__init__()
        self.args = args
        self.device = device
        self.entity_num = entity_embedding.shape[0]
        self.relation_num = relation_embedding.shape[0]

        # no_embedding
        self.news_embedding = torch.FloatTensor(np.array(news_title_embedding))
        self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(news_title_embedding))).to(self.device)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding).to(self.device)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding).to(self.device)

        # embedding
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size).to(self.device)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size).to(self.device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size).to(self.device)

        # net
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.gru = torch.nn.GRU(self.args.embedding_size, self.args.embedding_size)
        self.gru_output_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.gru_output_layer2 = nn.Linear(self.args.embedding_size, 1)
        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

    def _reconstruct_node_embedding(self):
        self.node_embedding = torch.cat([self.entity_embedding.weight,
                                         self.category_embedding.weight,
                                         self.subcategory_embedding.weight], dim=0).to(self.device)
        return self.node_embedding

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        trans_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_graph_list(self, graph_layers, batch_size):
        graph_list_flat = []
        graph_list = []
        for i in range(batch_size):
            graph_list_flat.append([])
            graph_list.append([[], [], []])
        for i in range(len(graph_layers)):
            for j in range(len(graph_layers[i])): # j代表每个batch有多少用户
                for k in range(len(graph_layers[i][j])):
                    graph_list[j][i].append(int(graph_layers[i][j][k].data.cpu().numpy()))
                    graph_list_flat[j].append(int(graph_layers[i][j][k].data.cpu().numpy()))
        return graph_list_flat, graph_list

    def get_overlap_entities(self, news_graph, user_graph):
        overlap_entity = []
        overlap_entity_num = []
        news_graph_num = []
        user_graph_num = []
        for i in range(len(news_graph)):
            news_graph_set = set()
            user_graph_set = set()
            for m in range(len(news_graph[i])):
                if news_graph[i][m] != 0:
                    news_graph_set.add(int(news_graph[i][m]))
            for j in range(1, len(user_graph[i])):
                if user_graph[i][j] != 0:
                    node = user_graph[i][j]
                    user_graph_set.add(int(node))
            if len(news_graph_set & user_graph_set) == 0:
                overlap_entity.append([0])
            else:
                overlap_entity.append(list(news_graph_set & user_graph_set))
            overlap_entity_num.append(len(news_graph_set & user_graph_set))
            news_graph_num.append(len(news_graph_set))
            user_graph_num.append(len(user_graph_set))

        def _cal_total_overlap_num(overlap_entity_num):
            total_overlap_num = 0
            for entity_num in overlap_entity_num:
                total_overlap_num += entity_num
            return total_overlap_num

        total_overlap_num = _cal_total_overlap_num(overlap_entity_num)
        overlap_entity_num_cpu = overlap_entity_num
        return torch.tensor(overlap_entity_num).to(self.device), overlap_entity_num_cpu, \
               torch.tensor(news_graph_num).to(self.device), torch.tensor(user_graph_num).to(self.device), overlap_entity, total_overlap_num

    def get_reasoning_paths(self, candidate_news, user_index, news_graph, user_graph,
                            news_graph_relation,  user_graph_relation,
                            overlap_entity_num_cpu):
        reasoning_paths = []
        reasoning_edges = []
        path_num_list = []
        # print('--------')
        for i in range(len(candidate_news)):
            path_num = 0
            reasoning_paths.append([])
            reasoning_edges.append([])
            # print('候选新闻：{}, 用户：{}'.format('news' + str(candidate_news[i]), 'user' + str(user_index[i])))
            # print(str(user_index[i].item()))

            # 新闻子图
            if overlap_entity_num_cpu[i] > 0:
                # print(overlap_entity_num_cpu[i])
                # print(overlap_entity[i])
                subgraph = nx.Graph()
                subgraph.add_node('news' + str(candidate_news[i].item()))
                subgraph.add_node('user' + str(user_index[i].item()))

                for index1 in range(self.args.depth[0]):
                    if news_graph[i][0][index1] != 0:
                        subgraph.add_edge('news' + str(candidate_news[i].item()),
                                          str(news_graph[i][0][index1]),
                                          weight = 0)
                        for index2 in range(self.args.depth[1]):
                            drump_1 = index1 * self.args.depth[1]
                            if news_graph[i][1][drump_1 + index2] != 0:
                                subgraph.add_edge(str(news_graph[i][0][index1]),
                                                  str(news_graph[i][1][drump_1 + index2]),
                                                  weight = news_graph_relation[1][i][drump_1 + index2])
                                for index3 in range(self.args.depth[2]):
                                    drump_2 = index1 * self.args.depth[1] * self.args.depth[2] + index2 * \
                                              self.args.depth[2]
                                    if news_graph[i][2][drump_2 + index3] != 0:
                                        subgraph.add_edge(str(news_graph[i][1][drump_1 + index2]),
                                                          str(news_graph[i][2][drump_2 + index3]),
                                                          weight = news_graph_relation[2][i][drump_2 + index3])

                # 用户子图
                for index1 in range(self.args.depth[0]):
                    if user_graph[i][0][index1] != 0:
                        subgraph.add_edge('user' + str(user_index[i].item()),
                                          'news' + str(user_graph[i][0][index1]),
                                          weight = 1)
                        for index2 in range(self.args.depth[1]):
                            drump_1 = index1 * self.args.depth[1]
                            if user_graph[i][1][drump_1 + index2] != 0:
                                subgraph.add_edge('news' + str(user_graph[i][0][index1]),
                                                  str(user_graph[i][1][drump_1 + index2]),
                                                  weight = user_graph_relation[1][i][drump_1 + index2])

                                for index3 in range(self.args.depth[2]):
                                    drump_2 = index1 * self.args.depth[1] * self.args.depth[2] + index2 * self.args.depth[2]
                                    if user_graph[i][2][drump_2 + index3] != 0:
                                        subgraph.add_edge(str(user_graph[i][1][drump_1 + index2]),
                                                          str(user_graph[i][2][drump_2 + index3]),
                                                          weight = user_graph_relation[2][i][drump_2+ index3])
                # 画出重叠图
                # print(subgraph)
                # nx.draw(subgraph, node_size=300, with_labels=True, node_color='r')
                # plt.show()
                # plt.close()

                for path in nx.all_simple_paths(subgraph,
                                                source='user' + str(user_index[i].item()),
                                                target='news' + str(candidate_news[i].item()),
                                                cutoff=5):
                    #print(path)
                    path_num += 1
                    reasoning_paths[-1].append(path)
                    reasoning_edges[-1].append([])
                    for j in range(len(path) - 1):
                        reasoning_edges[-1][-1].append(int(subgraph[path[j]][path[j+1]]['weight']))
                    reasoning_edges[-1][-1].append(int(0))
                if len(reasoning_paths[-1]) == 0:
                    reasoning_paths[-1].append(['user' + str(user_index[i].item()),
                                                str(0),
                                                'news' + str(candidate_news[i].item())])
                    reasoning_edges[-1].append([self.relation_num - 1, self.relation_num- 1, self.relation_num- 1])
            else:
                reasoning_paths[-1].append(['user' + str(user_index[i].item()),
                                            str(0),
                                            'news' + str(candidate_news[i].item())])
                reasoning_edges[-1].append([self.relation_num- 1, self.relation_num- 1, self.relation_num- 1])
            path_num_list.append(path_num)

        def _cal_total_path_num(path_num_list):
            total_path_num = 0
            for path_num in path_num_list:
                total_path_num += path_num
            return total_path_num
        total_path_num = _cal_total_path_num(path_num_list)
            # print(reasoning_paths[-1])
            # print(reasoning_edges[-1])

        return reasoning_paths, reasoning_edges, total_path_num, path_num_list

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


    def cal_nodes_num(self, graphs_flat):
        graph_num = []
        for graph in graphs_flat:
            num = 0
            for node in graph:
                if node != 0:
                   num += 1
            graph_num.append(num)
        return  graph_num


    def forward(self, candidate_news, user_index,
                news_graph, user_graph,
                news_graph_relation, user_graph_relation):
        self.node_embedding = self._reconstruct_node_embedding()
        candidate_news = torch.flatten(candidate_news, 0, 1)
        user_index = user_index.unsqueeze(1)
        user_index = user_index.expand(user_index.shape[0], 5)
        user_index = torch.flatten(user_index, 0, 1).to(self.device)

        news_graph_list_flat, news_graph_list = self.get_graph_list(news_graph, len(candidate_news)) # bz,d(1-hop)*d(2-hop)*d(3-hop); # bz, 3, d(1-hop) + d(2-hop) + d(3-hop)
        user_graph_list_flat, user_graph_list = self.get_graph_list(user_graph, len(user_index))

        news_graph_nodes_num = self.cal_nodes_num(news_graph_list_flat)
        user_graph_nodes_num = self.cal_nodes_num(user_graph_list_flat)

        overlap_entity_num, overlap_entity_num_list, \
        news_graph_num, user_graph_graph_num, overlap_entity,total_overlap_num = self.get_overlap_entities(news_graph_list_flat,
                                                                                                             user_graph_list_flat)
        reasoning_paths, reasoning_edges, path_num, path_num_list = self.get_reasoning_paths(candidate_news, user_index,
                                                                                              news_graph_list, user_graph_list,
                                                                                              news_graph_relation, user_graph_relation,
                                                                                              overlap_entity_num_list)
        predict_scores = None
        path_node_embeddings = None
        path_scores = None
        for i in range(len(reasoning_paths)):
            paths = reasoning_paths[i]
            edges = reasoning_edges[i]
            for j in range(len(paths)):
                if len(paths[j]) > 2:
                    path_node_embeddings_list = []
                    for m in range(1, len(paths[j]) - 1):
                        index, type_index = self.Split_num_letters(paths[j][m])
                        if type_index == 'user':
                            path_node_embeddings_list.append(self.user_embedding(torch.tensor(int(index)).to(self.device)))
                        elif type_index == 'news':
                            path_node_embeddings_list.append(self.trans_news_embedding(torch.tensor(int(index)).to(self.device)))
                        else:
                            path_node_embeddings_list.append(self.node_embedding[torch.tensor(int(index)).to(self.device)])
                    path_node_embeddings = torch.stack(path_node_embeddings_list)
                    path_edge_embeddings = self.relation_embedding(torch.LongTensor(edges[j][1:-1]).to(self.device))# dim
                elif len(paths[j]) ==  2:
                    path_node_embeddings_list = []
                    index, type_index = self.Split_num_letters(paths[j][1])
                    if type_index == 'user':
                        path_node_embeddings_list.append(self.user_embedding(torch.tensor(int(index)).to(self.device)))
                    elif type_index == 'news':
                        path_node_embeddings_list.append(self.trans_news_embedding(torch.tensor(int(index)).to(self.device)))
                    else:
                        path_node_embeddings_list.append(self.node_embedding[torch.tensor(int(index)).to(self.device)])
                    path_node_embeddings = torch.stack(path_node_embeddings_list)
                    path_edge_embeddings = self.relation_embedding(torch.tensor(edges[j][0]).to(self.device)).unsqueeze(0)# dim
                else:
                    index, type_index = self.Split_num_letters(paths[j][0])
                    if type_index == 'user':
                        path_node_embeddings = self.user_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                    elif type_index == 'news':
                        path_node_embeddings = self.trans_news_embedding(torch.tensor(int(index)).to(self.device)).unsqueeze(0)
                    else:
                        path_node_embeddings = self.node_embedding[torch.tensor(int(index)).to(self.device)].unsqueeze(0)
                    path_edge_embeddings = self.relation_embedding(torch.tensor(edges[j][0]).to(self.device)).unsqueeze(0)# dim
                if len(path_node_embeddings.shape) == 1:
                    path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)
                    path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 0)
                path_node_embeddings = torch.unsqueeze(path_node_embeddings, 1)  # 1, 1, dim
                path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 1)  # 1, 1, dim
                output, _ = self.gru(path_node_embeddings + path_edge_embeddings)
                path_score = self.gru_output_layer2(self.elu(self.gru_output_layer1(torch.squeeze(output[-1]))))

                if j == 0:
                    path_scores = path_score
                else:
                    path_scores = path_scores + path_score
            if i == 0:
                path_input_total = path_scores
            else:
                path_input_total = torch.cat([path_input_total, path_scores], dim=0)
            #     output, _ = self.gru(path_node_embeddings + path_edge_embeddings)
            #     if j == 0:
            #         input = output[-1]
            #     else:
            #         input = output[-1] + input
            # if i == 0:
            #     path_input = input
            # else:
            #     path_input = torch.cat([path_input, input], dim=0)

        # path_score = self.gru_output_layer2(self.elu(self.gru_output_layer1(torch.squeeze(path_input))))
        # path_score = path_score.squeeze()
        # 计算路径得分 和 路径重叠实体
        reason_qua = torch.div(path_input_total, (torch.log((np.e + news_graph_num + user_graph_graph_num).float())))
        reason_num = torch.div(overlap_entity_num, (torch.log((np.e + news_graph_num + user_graph_graph_num).float())))
        predicts = 0.5 * reason_qua + 0.5 * reason_num
        predicts = predicts.reshape(self.args.batch_size, self.args.sample_size)
        print('学习到的推理路径数：{}'.format(path_num))
        print('学习到的重叠点数：{}'.format(total_overlap_num))
        return predicts, reason_num.reshape(self.args.batch_size, self.args.sample_size), path_num, path_num_list, total_overlap_num, overlap_entity_num_list, \
               reasoning_paths, reasoning_edges, \
               news_graph_list, user_graph_list, news_graph_nodes_num, user_graph_nodes_num
