import torch
import torch.nn as nn
import numpy as np
import networkx as nx


class Reasoner(torch.nn.Module):
    def __init__(self, args, kg_env_all, entity_embedding, relation_embedding, device):
        super(Reasoner, self).__init__()
        self.args = args
        self.device = device
        self.kg_env_all = kg_env_all
        self.tanh = nn.Tanh()
        self.gru = torch.nn.GRU(self.args.embedding_size, self.args.embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.gru_output_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.gru_output_layer2 = nn.Linear(self.args.embedding_size, 1)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)

    def get_anchor_graph_list(self, anchor_graph_layers, batch_size):
        anchor_graph_list_flat = []
        anchor_graph_list = []
        for i in range(batch_size):
            anchor_graph_list_flat.append([])
            anchor_graph_list.append([[],[],[]])
        for i in range(len(anchor_graph_layers)):
            for j in range(len(anchor_graph_layers[i])):
                for k in range(len(anchor_graph_layers[i][j])):
                    anchor_graph_list[j][i].append(int(anchor_graph_layers[i][j][k].data.cpu().numpy()))
                    anchor_graph_list_flat[j].append(int(anchor_graph_layers[i][j][k].data.cpu().numpy()))
        return anchor_graph_list_flat, anchor_graph_list

    def get_overlap_entities(self, anchor_graph1, anchor_graph2):
        overlap_entity_num = []
        anchor_graph1_num = []
        anchor_graph2_num = []
        for i in range(len(anchor_graph1)):
            anchor_graph1_set = set()
            anchor_graph2_set = set()
            for anchor_node in anchor_graph1[i]:
                anchor_graph1_set.add(int(anchor_node))
            for anchor_node in anchor_graph2[i]:
                anchor_graph2_set.add(int(anchor_node))
            anchor_graph1_set.discard(0)
            anchor_graph2_set.discard(0)
            overlap_entity_num.append(len(anchor_graph1_set & anchor_graph2_set))
            anchor_graph1_num.append(len(anchor_graph1_set))
            anchor_graph2_num.append(len(anchor_graph2_set))
        overlap_entity_num_cpu = overlap_entity_num
        return torch.tensor(overlap_entity_num).to(self.device), torch.tensor(anchor_graph1_num).to(self.device), torch.tensor(anchor_graph2_num).to(self.device), overlap_entity_num_cpu

    def get_reasoning_paths(self, candidate_news, clicked_news, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2, overlap_entity_num_cpu):
        reasoning_paths = []
        reasoning_edges = []
        for i in range(len(candidate_news)):
            if overlap_entity_num_cpu[i]>0:
                subgraph = nx.MultiGraph()
                for index1 in range(self.args.depth[0]):
                    if anchor_graph1[i][0][index1] != 0:
                        subgraph.add_edge('news' + str(candidate_news[i]), anchor_graph1[i][0][index1], weight=0)
                    if anchor_graph2[i][0][index1] != 0:
                        subgraph.add_edge('news' + str(clicked_news[i]), anchor_graph2[i][0][index1], weight=0)
                    for index2 in range(self.args.depth[1]):
                        if anchor_graph1[i][1][index1*self.args.depth[1]+index2] != 0:
                            subgraph.add_edge(anchor_graph1[i][0][index1], anchor_graph1[i][1][index1*self.args.depth[1]+index2], weight=anchor_relation1[1][i][index1*self.args.depth[1]+index2])
                        if anchor_graph2[i][1][index1*self.args.depth[1]+index2] != 0:
                            subgraph.add_edge(anchor_graph2[i][0][index1], anchor_graph2[i][1][index1*self.args.depth[1]+index2], weight=anchor_relation2[1][i][index1*self.args.depth[1]+index2])
                        for index3 in range(self.args.depth[2]):
                            if anchor_graph1[i][2][index1*self.args.depth[1]*self.args.depth[2]+index2*self.args.depth[2]+index3] != 0:
                                subgraph.add_edge(anchor_graph1[i][1][index1*self.args.depth[1]+index2], anchor_graph1[i][2][index1*self.args.depth[1]*self.args.depth[2]+index2*self.args.args.depth[2]+index3], weight=anchor_relation1[2][i][index1*self.args.depth[1]*self.args.depth[2]+index2*self.args.depth[2]+index3])
                            if anchor_graph2[i][2][index1*self.args.depth[1]*self.args.depth[2]+index2*self.args.depth[2]+index3] != 0:
                                subgraph.add_edge(anchor_graph2[i][1][index1*self.args.depth[1]+index2], anchor_graph2[i][2][index1*self.args.depth[1]*self.args.depth[2]+index2*self.args.args.depth[2]+index3], weight=anchor_relation2[2][i][index1*self.args.depth[1]*self.args.depth[2]+index2*self.args.depth[2]+index3])
                reasoning_paths.append([])
                reasoning_edges.append([])
                for path in nx.all_simple_paths(subgraph, source='news' + str(candidate_news[i]), target='news' + str(clicked_news[i]), cutoff=5):
                    # print(len(path))
                    reasoning_paths[-1].append(path[1:-1])
                    reasoning_edges[-1].append([])
                    for i in range(len(path)-2):
                        reasoning_edges[-1][-1].append(int(subgraph[path[i]][path[i+1]][0]['weight']))
                    # print(reasoning_paths[-1])
                    # print(reasoning_edges[-1])
            else:
                reasoning_paths.append([0])
                reasoning_edges.append([0])
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

    def reshape_candidate_news(self, candidate_news, candidate_anchor_graph):
        candidate_news = candidate_news.unsqueeze(-1).repeat(1, 1, self.args.user_clicked_num)
        candidate_news = torch.flatten(candidate_news, 0, 2)
        candidate_anchor_graph_reshape = []
        for anchor_graph in candidate_anchor_graph:
            anchor_graph = anchor_graph.reshape(self.args.batch_size, self.args.sample_size, -1)
            anchor_graph = anchor_graph.unsqueeze(2)
            anchor_graph = anchor_graph.repeat(1, 1, self.args.user_clicked_num, 1)
            anchor_graph = torch.flatten(anchor_graph, 0, 2)
            candidate_anchor_graph_reshape.append(anchor_graph)
        return candidate_news, candidate_anchor_graph_reshape

    def reshape_clicked_news(self, clicked_news, clicked_anchor_graph):
        clicked_news = clicked_news.unsqueeze(1).repeat(1, self.args.sample_size, self.args.user_clicked_num)
        clicked_news = torch.flatten(clicked_news, 0, 2)
        clicked_anchor_graph_reshape = []
        for anchor_graph in clicked_anchor_graph:
            anchor_graph = anchor_graph.reshape(self.args.batch_size, self.args.user_clicked_num, -1)
            anchor_graph = anchor_graph.unsqueeze(1)
            anchor_graph = anchor_graph.repeat(1, self.args.sample_size, 1, 1)
            anchor_graph = torch.flatten(anchor_graph, 0, 2)
            clicked_anchor_graph_reshape.append(anchor_graph)
        return clicked_news, clicked_anchor_graph_reshape

    def forward(self, candidate_news, clicked_news, candidate_anchor_graph, clicked_anchor_graph2, anchor_relation1, anchor_relation2):
        candidate_news, candidate_anchor_graph = self.reshape_candidate_news(candidate_news, candidate_anchor_graph)
        clicked_news, clicked_anchor_graph2 = self.reshape_clicked_news(clicked_news, clicked_anchor_graph2)
        candidate_anchor_graph_list_flat, candidate_anchor_graph_list = self.get_anchor_graph_list(candidate_anchor_graph, len(candidate_news)) # bz,d(1-hop)*d(2-hop)*d(3-hop); # bz, 3, d(1-hop) + d(2-hop) + d(3-hop)
        clicked_anchor_graph_list_flat, clicked_anchor_graph_list = self.get_anchor_graph_list(clicked_anchor_graph2, len(clicked_news))
        overlap_entity_num, candidate_anchor_graph_num, clicked_anchor_graph_num, overlap_entity_num_cpu = self.get_overlap_entities(candidate_anchor_graph_list_flat, clicked_anchor_graph_list_flat)
        reasoning_paths, reasoning_edges = self.get_reasoning_paths(candidate_news, clicked_news, candidate_anchor_graph_list, clicked_anchor_graph_list,
                                                                    anchor_relation1, anchor_relation2, overlap_entity_num_cpu)
        predict_scores = []
        for i in range(len(reasoning_paths)):
            paths = reasoning_paths[i]
            edges = reasoning_edges[i]
            predict_scores.append([])
            paths_node_embeddings = None
            paths_edge_embeddings = None
            for j in range(len(paths)):
                path_node_embeddings = self.entity_embedding(torch.tensor(paths[j]).to(self.device)) # dim
                path_edge_embeddings = self.relation_embedding(torch.tensor(edges[j]).to(self.device)) # dim
                if len(path_node_embeddings.shape) == 1:
                    path_node_embeddings = torch.unsqueeze(path_node_embeddings, 0)
                    path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 0)
                path_node_embeddings = torch.unsqueeze(path_node_embeddings, 1)  # 1, 1, dim
                path_edge_embeddings = torch.unsqueeze(path_edge_embeddings, 1)  # 1, 1, dim
                if j == 0:
                    paths_node_embeddings = path_node_embeddings
                    paths_edge_embeddings = path_edge_embeddings
                else:
                    paths_node_embeddings = torch.cat([paths_node_embeddings, path_node_embeddings], dim=0)
                    paths_edge_embeddings = torch.cat([paths_edge_embeddings, path_edge_embeddings], dim=0)

            output, _ = self.gru(paths_node_embeddings + paths_edge_embeddings)
            path_score = self.sigmoid(self.gru_output_layer2(self.elu(self.gru_output_layer1(torch.squeeze(output[-1])))))
            predict_scores[-1].append(path_score.float())
            # predict_scores[-1] = torch.sum(torch.tensor(predict_scores[-1]).cuda()).float()
        paths_predict_socre = torch.tensor(predict_scores).to(self.device).squeeze()
        predicts_qua = self.tanh(torch.div(paths_predict_socre, (torch.log((np.e+candidate_anchor_graph_num+clicked_anchor_graph_num).float()))))
        predicts_num = self.tanh(torch.div(overlap_entity_num, (torch.log((np.e+candidate_anchor_graph_num+clicked_anchor_graph_num).float()))))
        predicts = 0.8 * predicts_qua + 0.2 * predicts_num
        predicts = predicts.reshape(self.args.batch_size, self.args.sample_size, self.args.user_clicked_num)
        predicts = torch.sum(predicts, dim = -1)
        return predicts, reasoning_paths, reasoning_edges, predict_scores
