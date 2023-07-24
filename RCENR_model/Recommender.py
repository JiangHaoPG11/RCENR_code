import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim):
        super(news_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.tanh = nn.Tanh()
        self.dropout_prob = 0.2
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, self.multi_dim, attention_heads)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)

    def forward(self, word_embedding):
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.tanh(self.word_attention(word_embedding))
        news_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim):
        super(user_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.tanh = nn.Tanh()
        self.news_encoder = news_encoder(word_dim, attention_dim, attention_heads, query_vector_dim)
        self.multiheadatt = MultiHeadSelfAttention_2(self.multi_dim,
                                                     self.multi_dim,
                                                     attention_heads)
        self.multiheadatt2 = MultiHeadSelfAttention_2(self.multi_dim,
                                                      self.multi_dim,
                                                      attention_heads)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.layernorm1 = nn.LayerNorm(self.multi_dim)
        self.layernorm2 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding):
        # 点击新闻表征
        news_rep = self.news_encoder(word_embedding).unsqueeze(0)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        news_rep = self.multiheadatt(news_rep)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        news_rep = self.layernorm1(news_rep)
        user_rep = self.user_attention(news_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.layernorm2(user_rep)
        return user_rep



class Recommender(torch.nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index,
                 device):
        super(Recommender, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding

        # embedding
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size).to(self.device)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size).to(self.device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size).to(self.device)
        self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(news_title_embedding))).to(self.device)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)

        # encoder
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim)

        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

        # net
        self.elu = nn.ELU(inplace=False)
        self.mlp_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.mlp_layer2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.embedding_layer = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.weights_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.weights_layer2 = nn.Linear(self.args.embedding_size, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)

    def _reconstruct_node_embedding(self):
        self.node_embedding = torch.cat([self.entity_embedding.weight,
                                         self.category_embedding.weight,
                                         self.subcategory_embedding.weight], dim=0).to(self.device)
        return self.node_embedding

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        trans_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    # 把所有的实体嵌入和邻居实体嵌入提取
    def get_graph_embedding(self, graph, mode): # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        graph_nodes = []
        for i in range(len(graph[1])): # bz
            graph_nodes.append([])
            for j in range(len(graph)): # k-hop
                graph_nodes[-1].extend(graph[j][i].tolist()) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
        if mode == "news":
            graph_nodes_embedding = self.node_embedding[torch.LongTensor(graph_nodes).to(self.device)]
        if mode == "user":
            graph_nodes_embedding_1 = self.trans_news_embedding(graph[0])
            graph_nodes_embedding_2 = self.node_embedding[graph[1]]
            graph_nodes_embedding_3 = self.node_embedding[graph[2]]
            graph_nodes_embedding = torch.cat([graph_nodes_embedding_1, graph_nodes_embedding_2, graph_nodes_embedding_3], dim = 1)

        graph_embedding = self.tanh(self.embedding_layer(graph_nodes_embedding))
        graph_embedding_weight = F.softmax(self.weights_layer2(self.elu(self.weights_layer1(graph_embedding))), dim = -2)
        graph_embedding = torch.sum(graph_embedding * graph_embedding_weight, dim=-2) # bz, dim (ut in equation)
        return graph_embedding


    def get_news_entities_batch(self, newsids):
        news_entities = []
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                news_entities[-1].append([])
                news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:])
        return np.array(news_entities)

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index.cpu()]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index.cpu()]].to(self.device)
        
        ## 新闻编码器
        news_rep = None
        for i in range(self.args.sample_num):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :]
            news_rep_one = self.news_encoder(news_word_embedding_one).unsqueeze(1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim = 1)
        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            # 点击新闻单词嵌入
            clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_word_embedding_one).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep

    def forward(self, candidate_news, user_clicked_news_index, news_graph, user_graph):
        weight = 0.8
        self.node_embedding = self._reconstruct_node_embedding()
        candidate_news = candidate_news.to(self.device)
        user_clicked_news_index = user_clicked_news_index.to(self.device)
        
        # 新闻用户表征
        user_rep, news_rep, score = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        score = torch.sum(news_rep * user_rep, dim = -1)

        # 子图嵌入
        news_graph_embedding = self.get_graph_embedding(news_graph, mode="news") # bz, news_dim
        user_graph_embedding = self.get_graph_embedding(user_graph, mode="user") # bz, news_dim
        graph_score = torch.sum(news_graph_embedding * user_graph_embedding, dim=-1).view(self.args.batch_size, -1) 
        
        # 得分
        score = weight * score + (1 - weight) * graph_score
        return score
