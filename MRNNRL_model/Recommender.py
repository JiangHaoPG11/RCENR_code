import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class new_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(new_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        # 主题级表征网络
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.fc1 = nn.Linear(category_dim, self.multi_dim, bias=True)
        self.batchnorm2 = nn.BatchNorm1d(self.multi_dim)
        # 副主题级表征网络
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc2 = nn.Linear(subcategory_dim, self.multi_dim, bias=True)
        self.batchnorm1 = nn.BatchNorm1d(self.multi_dim)
        # 单词级表征网络
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, attention_dim * attention_heads, attention_heads)
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        # 实体级表征网络
        self.GCN = gcn(entity_size, entity_embedding_dim, self.multi_dim)
        self.norm3 = nn.LayerNorm(self.multi_dim)
        self.entity_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm4 = nn.LayerNorm(self.multi_dim)
        self.new_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm5 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 主题级表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.fc1(category_embedding)
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)
        category_rep = self.batchnorm1(category_rep)
        # 副主题级表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        subcategory_rep = self.batchnorm2(subcategory_rep)
        # 单词级新闻表征
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm1(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.word_attention(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        word_rep = self.norm2(word_rep)
        # 实体级新闻表征
        entity_embedding = F.dropout(entity_embedding, p=self.dropout_prob, training=self.training)
        entity_inter = self.GCN(entity_embedding)
        entity_inter = self.norm3(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_rep = self.entity_attention(entity_inter)
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)
        entity_rep = self.norm4(entity_rep)
        # 新闻附加注意力
        new_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        new_rep = self.new_attention(new_rep)
        new_rep = self.norm5(new_rep)
        return new_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                       entity_embedding_dim, category_dim, subcategory_dim, category_size, subcategory_size)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads, attention_heads)
        self.multi_dim = attention_dim * attention_heads
        self.norm1 = nn.LayerNorm(self.multi_dim)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.norm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 点击新闻表征
        new_rep = self.new_encoder(word_embedding, entity_embedding, category_index, subcategory_index).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.multiheadatt(new_rep)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.norm1(new_rep)
        # 用户表征
        user_rep = self.user_attention(new_rep)
        user_rep = self.norm2(user_rep)
        return user_rep

class Recommender(torch.nn.Module):

    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 new_title_word_index, word_embedding, new_category_index, new_subcategory_index,
                 device):
        super(Recommender, self).__init__()
        self.args = args
        self.device = device

        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size)
        # new_embedding = news_title_embedding.tolist()
        # new_embedding.append(np.array([0 for i in range(768)]))
        self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(news_title_embedding)))
        self.type_embedding = nn.Embedding(5, self.args.embedding_size)
        self.entity_embedding_pre = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding_pre = nn.Embedding.from_pretrained(relation_embedding)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size)


        self.new_encoder = new_encoder(self.args.word_embedding_dim, self.args.attention_dim, self.args.attention_heads, self.args.query_vector_dim,
                                       self.args.new_entity_size, self.args.entity_embedding_dim, self.args.category_dim, self.args.subcategory_dim,
                                       self.args.category_num, self.args.subcategory_num)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim, self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.new_entity_size, self.args.entity_embedding_dim,self.args.category_dim, self.args.subcategory_dim,
                                         self.args.category_num, self.args.subcategory_num)
        self.new_title_word_index = new_title_word_index
        self.new_category_index = new_category_index
        self.new_subcategory_index = new_subcategory_index
        self.word_embedding = word_embedding
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

        self.elu = nn.ELU(inplace=False)
        self.mlp_layer1 = nn.Linear(self.args.embedding_size + 400, self.args.embedding_size)
        self.mlp_layer2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.embedding_layer = nn.Linear(self.args.embedding_size * 2, self.args.embedding_size)
        self.weights_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.weights_layer2 = nn.Linear(self.args.embedding_size, 1)
        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        trans_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.news_title_embedding[newsid]).to(self.device))
        return torch.stack(news_embeddings)

    def get_news_entities_batch(self, newsids):
        news_entities = []
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                news_entities[-1].append([])
                news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.new_entity_size])
        return np.array(news_entities)

    def get_entity_neighbors(self, entities):
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities: # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if entity not in self.entity_adj.keys():
                    neighbor_entities[-1].append(self.entity_adj[0])
                    neighbor_relations[-1].append(self.relation_adj[0])
                else:
                    if type(entity) == int:
                        neighbor_entities[-1].append(self.entity_adj[entity])
                        neighbor_relations[-1].append(self.relation_adj[entity])
                    else:
                        neighbor_entities[-1].append([])
                        neighbor_relations[-1].append([])
                        for entity_i in entity:
                            neighbor_entities[-1][-1].append(self.entity_adj[entity_i])
                            neighbor_relations[-1][-1].append(self.relation_adj[entity_i])
        return torch.tensor(neighbor_entities).to(self.device), torch.tensor(neighbor_relations).to(self.device)

    def get_news_neighbors(self, news):
        neighbor_entities = []
        neighbor_relations = []
        for news_batch in news: # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
            neighbor_entities.append([])
            neighbor_relations.append([])
            for new in news_batch:
                if new not in self.news_entity_dict.keys():
                    neighbor_entities[-1].append(self.news_entity_dict[0])
                    neighbor_relations[-1].append([0 for i in range(len(self.news_entity_dict[0]))])
                else:
                    if type(new) == int:
                        neighbor_entities[-1].append(self.news_entity_dict[new])
                        neighbor_relations[-1].append([0 for i in range(len(self.news_entity_dict[0]))])
                    else:
                        neighbor_entities[-1].append([])
                        neighbor_relations[-1].append([])
                        for new_i in new:
                            neighbor_entities[-1][-1].append(self.news_entity_dict[new_i])
                            neighbor_relations[-1][-1].append([0 for i in range(len(self.news_entity_dict[0]))])
        return torch.tensor(neighbor_entities).to(self.device), torch.tensor(neighbor_relations).to(self.device)

    def get_user_graph_embedding(self, user_graph): # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        user_graph_nodes = []
        for i in range(len(user_graph[1])):
            user_graph_nodes.append([])
            user_graph_nodes[-1].extend(user_graph[0][i].tolist())
        user_graph_nodes_embedding1 = self.trans_news_embedding((torch.tensor(user_graph_nodes).to(self.device)).squeeze().to(torch.int64))  # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), dim
        neibor_entities, neibor_relations = self.get_news_neighbors(user_graph_nodes)  # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20
        neibor_entities_embedding1 = self.entity_embedding_pre(neibor_entities).repeat(1, 1, 2, 1)  # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim
        neibor_relations_embedding1 = self.relation_embedding_pre(neibor_relations).repeat(1, 1, 2, 1)   # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim

        user_graph_nodes = []
        for i in range(len(user_graph[1])): # bz
            user_graph_nodes.append([])
            for j in range(1, len(user_graph)): # k-hop
                user_graph_nodes[-1].extend(user_graph[j][i].tolist()) # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
        user_graph_nodes_embedding2 = self.entity_embedding_pre(torch.tensor(user_graph_nodes).to(self.device)) # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), dim

        neibor_entities, neibor_relations = self.get_entity_neighbors(user_graph_nodes) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20
        neibor_entities_embedding2 = self.entity_embedding_pre(neibor_entities.to(self.device)) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim
        neibor_relations_embedding2 = self.relation_embedding_pre(neibor_relations.to(self.device)) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim

        user_graph_nodes_embedding = torch.cat([user_graph_nodes_embedding1, user_graph_nodes_embedding2], dim = 1)
        neibor_entities_embedding = torch.cat([neibor_entities_embedding1, neibor_entities_embedding2], dim=1)
        neibor_relations_embedding = torch.cat([neibor_relations_embedding1, neibor_relations_embedding2], dim=1)

        user_graph_embedding = torch.cat([user_graph_nodes_embedding, torch.sum(neibor_entities_embedding + neibor_relations_embedding, dim=-2)], dim=-1) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 2*dim
        user_graph_embedding = self.tanh(self.embedding_layer(user_graph_embedding))
        user_graph_embedding_weight = F.softmax(self.weights_layer2(self.elu(self.weights_layer1(user_graph_embedding))), dim = -2)
        user_graph_embedding = torch.sum(user_graph_embedding * user_graph_embedding_weight, dim=-2) # bz, dim (ut in equation)
        return user_graph_embedding

        # 把所有的实体嵌入和邻居实体嵌入提取

    def get_news_graph_embedding(self, news_graph):  # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        news_graph_nodes = []
        for i in range(len(news_graph[1])):  # bz
            news_graph_nodes.append([])
            for j in range(len(news_graph)):  # k-hop
                news_graph_nodes[-1].extend(news_graph[j][i].tolist())  # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
        news_graph_nodes_embedding = self.entity_embedding_pre(torch.tensor(news_graph_nodes).to(self.device)) # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), dim
        neibor_entities, neibor_relations = self.get_entity_neighbors(news_graph_nodes) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20
        neibor_entities_embedding = self.entity_embedding_pre(neibor_entities.to(self.device)) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim
        neibor_relations_embedding = self.relation_embedding_pre(neibor_relations.to(self.device)) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim
        news_graph_embedding = torch.cat([news_graph_nodes_embedding, torch.sum(neibor_entities_embedding + neibor_relations_embedding,dim=-2)], dim=-1)  # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 2*dim
        news_graph_embedding = self.tanh(self.embedding_layer(news_graph_embedding))
        news_graph_embedding_weight = F.softmax(self.weights_layer2(self.elu(self.weights_layer1(news_graph_embedding))), dim=-2)
        news_graph_embedding = torch.sum(news_graph_embedding * news_graph_embedding_weight, dim=-2)  # bz, dim (ut in equation)
        return news_graph_embedding


    # 把所有的实体嵌入和邻居实体嵌入提取
    def get_graph_embedding(self, graph, graph_type): # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        graph_nodes = []
        graph_nodes_type = []
        for i in range(len(graph[1])): # bz
            graph_nodes.append([])
            graph_nodes_type.append([])
            for j in range(len(graph)): # k-hop
                graph_nodes[-1].extend(graph[j][i].tolist()) # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
                graph_nodes_type[-1].extend(graph_type[j][i].tolist())  # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)

        graph_nodes_embedding_list = []
        graph_type_embedding_list = []
        for i in range(len(graph_nodes)):
            graph_nodes_embedding_list.append([])
            graph_type_embedding_list.append([])
            for j in range(len(graph_nodes[i])):
                type = graph_nodes_type[i][j]
                entity = graph_nodes[i][j]
                if type == 0:
                    graph_nodes_embedding_list[-1].append(self.user_embedding(torch.tensor(entity).to(self.device)))
                    graph_type_embedding_list[-1].append(self.type_embedding(torch.tensor(type).to(self.device)))
                elif type == 1:
                    graph_nodes_embedding_list[-1].append(self.trans_news_embedding(torch.tensor(entity).to(self.device)))
                    graph_type_embedding_list[-1].append(self.type_embedding(torch.tensor(type).to(self.device)))
                elif type == 2:
                    graph_nodes_embedding_list[-1].append(self.entity_embedding_pre(torch.tensor(entity).to(self.device)))
                    graph_type_embedding_list[-1].append(self.type_embedding(torch.tensor(type).to(self.device)))
                elif type == 3:
                    graph_nodes_embedding_list[-1].append(self.category_embedding(torch.tensor(entity).to(self.device)))
                    graph_type_embedding_list[-1].append(self.type_embedding(torch.tensor(type).to(self.device)))
                elif type == 4:
                    graph_nodes_embedding_list[-1].append(self.subcategory_embedding(torch.tensor(entity).to(self.device)))
                    graph_type_embedding_list[-1].append(self.type_embedding(torch.tensor(type).to(self.device)))
                else:
                    print("实体类型错误")
            graph_nodes_embedding_list[-1] = torch.stack(graph_nodes_embedding_list[-1])
            graph_type_embedding_list[-1] = torch.stack(graph_type_embedding_list[-1])
        graph_nodes_embedding = torch.stack(graph_nodes_embedding_list)
        graph_type_embedding = torch.stack(graph_type_embedding_list)
        graph_embedding = torch.cat([graph_nodes_embedding, graph_type_embedding], dim=-1) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 2*dim
        graph_embedding = self.tanh(self.embedding_layer(graph_embedding))
        graph_embedding_weight = F.softmax(self.weights_layer2(self.elu(self.weights_layer1(graph_embedding))), dim = -2)
        graph_embedding = torch.sum(graph_embedding * graph_embedding_weight, dim=-2) # bz, dim (ut in equation)
        return graph_embedding

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        candidate_new_word_embedding = self.word_embedding[self.new_title_word_index[candidate_news_index]]
        user_clicked_new_word_embedding = self.word_embedding[self.new_title_word_index[user_clicked_news_index]]
        candidate_new_entity_embedding = self.entity_embedding[self.get_news_entities_batch(candidate_news_index)].squeeze().to(device)
        user_clicked_new_entity_embedding = self.entity_embedding[self.get_news_entities_batch(user_clicked_news_index)].squeeze().to(device)
        candidate_new_category_index = torch.IntTensor(self.new_category_index[np.array(candidate_news_index)])
        user_clicked_new_category_index = torch.IntTensor(self.new_category_index[np.array(user_clicked_news_index)]).to(device)
        candidate_new_subcategory_index = torch.IntTensor(self.new_subcategory_index[np.array(candidate_news_index)]).to(device)
        user_clicked_new_subcategory_index = torch.IntTensor(self.new_subcategory_index[np.array(user_clicked_news_index)]).to(device)
        ## 新闻编码器
        new_rep= self.new_encoder(candidate_new_word_embedding.squeeze(), candidate_new_entity_embedding,
                                  candidate_new_category_index.squeeze(), candidate_new_subcategory_index.squeeze())
        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size * self.args.sample_num):
            # 点击新闻单词嵌入
            clicked_new_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
            clicked_new_word_embedding_one = clicked_new_word_embedding_one.squeeze()
            # 点击新闻实体嵌入
            clicked_new_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
            clicked_new_entity_embedding_one = clicked_new_entity_embedding_one.squeeze()
            # 点击新闻主题index
            clicked_new_category_index = user_clicked_new_category_index[i, :]
            # 点击新闻副主题index
            clicked_new_subcategory_index = user_clicked_new_subcategory_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_entity_embedding_one,
                                             clicked_new_category_index, clicked_new_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, new_rep

    def forward(self, cand_news, user_index, user_clicked_news_index, news_graph, user_graph,  news_graph_type, user_graph_type):  # bz, 1; bz, 1; [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        cand_news = torch.flatten(cand_news, 0, 1).unsqueeze(-1).to(self.device)
        # user_index = torch.flatten(user_index.unsqueeze(1).repeat(1, 5, 1), 0, 1).squeeze().to(self.device)
        # pt = user_index.unsqueeze(-1)
        # for i in range(5):
        #     if i == 0:
        #         new_pt = pt
        #     else:
        #         new_pt = torch.cat([new_pt, pt], dim=-1)
        # user_index = torch.flatten(new_pt, 0, 1).to(self.device)

        user_clicked_news_index = torch.flatten(user_clicked_news_index.unsqueeze(1).repeat(1, 5, 1), 0, 1).squeeze().to(self.device)

        user_rep, news_rep = self.get_user_news_rep(cand_news, user_clicked_news_index)
        user_rep = user_rep.squeeze()

        news_graph_embedding = self.get_graph_embedding(news_graph, news_graph_type) # bz, news_dim
        user_graph_embedding = self.get_graph_embedding(user_graph, user_graph_type) # bz, news_dim

        news_embedding = torch.cat([user_rep, news_graph_embedding], dim=-1) # bz, 2 * news_dim
        user_embedding = torch.cat([news_rep, user_graph_embedding], dim=-1) # bz, 2 * news_dim

        news_embedding = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(news_embedding))))
        user_embedding = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(user_embedding))))

        score = self.sigmod(self.cos(news_embedding, user_embedding).view(self.args.batch_size, -1))
        return score
