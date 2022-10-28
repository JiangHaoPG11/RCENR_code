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
        self.tanh = nn.Tanh()
        self.dropout_prob = 0.2

        # 主题级表征网络
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.fc1 = nn.Linear(category_dim, self.multi_dim, bias=True)

        # 副主题节点学习网络
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc2 = nn.Linear(subcategory_dim, self.multi_dim, bias=True)

        # 标题节点学习网络
        self.layernorm_word = nn.LayerNorm(word_dim)
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, attention_dim * attention_heads, attention_heads)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.layernorm1 = nn.LayerNorm(self.multi_dim)
        #self.norm2 = nn.LayerNorm(self.multi_dim)

        # 实体节点学习网络
        self.layernorm_entity = nn.LayerNorm(entity_embedding_dim)
        self.fc4 = nn.Linear(entity_embedding_dim, 100, bias=True)
        self.layernorm2 = nn.LayerNorm(self.multi_dim)
        self.GCN = gcn(20, 100, self.multi_dim)
        self.KGAT = KGAT(self.multi_dim, entity_embedding_dim)
        self.entity_attention = Additive_Attention(query_vector_dim, self.multi_dim)

        # 新闻融合
        self.new_attention = Additive_Attention(query_vector_dim, self.multi_dim)

        # 新闻显式关系构建网络
        self.fc5 = nn.Linear(entity_embedding_dim, self.multi_dim, bias = True)
        self.relation_attention = SimilarityAttention()
        self.layernorm3 = nn.LayerNorm(self.multi_dim)

        # 最终归一化
        self.layernorm4 = nn.LayerNorm(self.multi_dim)
        self.layernorm5 = nn.LayerNorm(self.multi_dim)

        self.batchnorm4 = nn.BatchNorm1d(self.multi_dim)
        self.batchnorm5 = nn.BatchNorm1d(self.multi_dim)

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 主题级表征
        #category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        #category_rep = self.tanh(self.fc1(category_embedding))
        #category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 副主题级表征
        #subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        #subcategory_rep = self.tanh(self.fc2(subcategory_embedding))
        #subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        # 单词级新闻表征
        # word_embedding = self.layernorm_word(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        # word_embedding = self.layernorm1(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = self.tanh(self.word_attention(word_embedding))
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        # print('----------单词----------')
        # print(word_rep[0])
        new_rep = word_rep
        # 实体级新闻表征
        # entity_embedding = F.dropout(entity_embedding, p=self.dropout_prob, training=self.training)
        # entity_agg = self.KGAT(entity_embedding, neigh_entity_embedding, neigh_relation_embedding)
        #entity_agg = entity_embedding[:, :20, :]
        #entity_inter = self.tanh(self.fc4(entity_agg))
        #entity_inter = self.GCN(entity_inter)
        #entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        #entity_rep = self.tanh(self.entity_attention(entity_inter))
        #entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)

        # 语义级-新闻表征
        #new_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        #new_rep = self.new_attention(new_rep)
        # new_rep = self.layernorm4(new_rep)

        # 通道级-新闻表征
        # relation = self.tanh(self.fc5(entity_agg))
        # relation = F.dropout(relation, p=self.dropout_prob, training=self.training)
        # relation = self.relation_attention(new_rep, relation)
        # relation_rep = F.dropout(relation, p=self.dropout_prob, training=self.training)
        # relation_rep = self.layernorm5(relation_rep)
        # 融合
        # new_rep = torch.stack([new_rep, relation_rep], dim=1)
        return new_rep


class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.type_encoder = 'userencoder'
        self.new_encoder = new_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                       entity_embedding_dim, category_dim, subcategory_dim, category_size, subcategory_size)
        self.multiheadatt = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads,
                                                     attention_heads)
        self.multiheadatt2 = MultiHeadSelfAttention_2(attention_dim * attention_heads, attention_dim * attention_heads,
                                                     attention_heads)
        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.layernorm1 = nn.LayerNorm(self.multi_dim)
        self.layernorm2 = nn.LayerNorm(self.multi_dim)
        self.layernorm3 = nn.LayerNorm(self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding,
                category_index, subcategory_index):
        # 点击新闻表征
        new_rep = self.new_encoder(word_embedding, entity_embedding,
                                   category_index, subcategory_index).unsqueeze(0)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)

        # # 点击新闻语义
        # new_embedding = new_rep[:, :, 0, :]
        # # 点击关系语义
        # relation_embedding = new_rep[:, :, 1, :]

        # 语义级交互
        new_rep = self.multiheadatt(new_rep)
        new_rep = F.dropout(new_rep, p=self.dropout_prob, training=self.training)
        new_rep = self.layernorm1(new_rep)

        # # 关系级交互
        # relation_rep = self.multiheadatt2(relation_embedding)
        # relation_rep = F.dropout(relation_rep, p=self.dropout_prob, training=self.training)
        # relation_rep = self.layernorm2(relation_rep)

        # 用户表征
        # new_rep = torch.cat([new_rep, relation_rep], dim=0)
        user_rep = self.user_attention(new_rep)
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.layernorm3(user_rep)
        #print(user_rep.shape)
        return user_rep



class Recommender_simple_v5(torch.nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 new_title_word_index, word_embedding, new_category_index, new_subcategory_index,
                 device):
        super(Recommender_simple_v5, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)
        self.news_embedding = torch.FloatTensor(np.array(news_title_embedding))
        self.news_embedding = self.news_embedding.to(device)

        # embedding
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size).to(self.device)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size).to(self.device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size).to(self.device)

        # MRNN
        self.new_encoder = new_encoder(self.args.word_embedding_dim, self.args.attention_dim, self.args.attention_heads, self.args.query_vector_dim,
                                       self.args.new_entity_size, self.args.entity_embedding_dim, self.args.category_dim, self.args.subcategory_dim,
                                       self.args.category_num, self.args.subcategory_num)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim, self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.new_entity_size, self.args.entity_embedding_dim, self.args.category_dim, self.args.subcategory_dim,
                                         self.args.category_num, self.args.subcategory_num)

        # dict
        self.news_title_word_dict = new_title_word_index
        self.news_category_dict = new_category_index
        self.news_subcategory_dict = new_subcategory_index
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
        self.sigmod = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)

    def _reconstruct_node_embedding(self):
        self.node_embedding = torch.cat([self.entity_embedding,
                                         self.category_embedding.weight,
                                         self.subcategory_embedding.weight], dim=0).to(self.device)
        return self.node_embedding

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding[news_index]
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
                #news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.new_entity_size])
                news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:])
        return np.array(news_entities)

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        candidate_new_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index.cpu()]].to(self.device)
        user_clicked_new_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index.cpu()]].to(self.device)
        candidate_new_entity_embedding = self.entity_embedding[self.get_news_entities_batch(candidate_news_index.cpu())].to(self.device).squeeze()
        user_clicked_new_entity_embedding = self.entity_embedding[self.get_news_entities_batch(user_clicked_news_index.cpu())].to(self.device).squeeze()
        candidate_new_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_new_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        candidate_new_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_new_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        ## 新闻编码器
        #new_rep = self.new_encoder(candidate_new_word_embedding.squeeze(), candidate_new_entity_embedding,
        #                           candidate_new_category_index.squeeze(), candidate_new_subcategory_index.squeeze())
        # 用户编码器
        #user_rep = None
        #for i in range(self.args.batch_size * self.args.sample_num):
            # 点击新闻单词嵌入
        #    clicked_new_word_embedding_one = user_clicked_new_word_embedding[i, :, :, :]
        #    clicked_new_word_embedding_one = clicked_new_word_embedding_one.squeeze()
            # 点击新闻实体嵌入
        #    clicked_new_entity_embedding_one = user_clicked_new_entity_embedding[i, :, :, :]
        #    clicked_new_entity_embedding_one = clicked_new_entity_embedding_one.squeeze()
            # 点击新闻主题index
        #    clicked_new_category_index = user_clicked_new_category_index[i, :]
            # 点击新闻副主题index
        #    clicked_new_subcategory_index = user_clicked_new_subcategory_index[i, :]
        #    # 用户表征
        #    user_rep_one = self.user_encoder(clicked_new_word_embedding_one, clicked_new_entity_embedding_one,
        #                                    clicked_new_category_index, clicked_new_subcategory_index).unsqueeze(0)
        #    if i == 0:
        #        user_rep = user_rep_one
        #    else:
        #        user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        
        ## 新闻编码器
        new_rep = None
        for i in range(5):
            new_word_embedding_one = candidate_new_word_embedding[:, i, :]
            new_entity_embedding_one = candidate_new_entity_embedding[:, i, :]
            new_category_index = candidate_new_category_index[:, i]
            new_subcategory_index = candidate_new_subcategory_index[:, i]
            new_rep_one = self.new_encoder(new_word_embedding_one, new_entity_embedding_one,
                                           new_category_index, new_subcategory_index
                                           ).unsqueeze(1)
            if i == 0:
                new_rep = new_rep_one
            else:
                new_rep = torch.cat([new_rep, new_rep_one], dim = 1)
        # 用户编码器
        user_rep = None
        for i in range(50):
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
                                             clicked_new_category_index, clicked_new_subcategory_index
                                             ).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        score = torch.sum(new_rep * user_rep, dim = -1)
        return user_rep, new_rep, score

    def forward(self, cand_news, user_clicked_news_index, news_graph, user_graph):
        self.node_embedding = self._reconstruct_node_embedding()
        #cand_news = torch.flatten(cand_news, 0, 1).unsqueeze(-1).to(self.device)
        #user_clicked_news_index = user_clicked_news_index.unsqueeze(1)
        #user_clicked_news_index = user_clicked_news_index.expand(user_clicked_news_index.shape[0], 5,
        #                                                         user_clicked_news_index.shape[2])
        #user_clicked_news_index = torch.flatten(user_clicked_news_index, 0, 1).to(self.device)
        weight = 0.8
        cand_news = cand_news.to(self.device)
        user_clicked_news_index = user_clicked_news_index.to(self.device)
        # 新闻用户表征
        user_rep, news_rep, score = self.get_user_news_rep(cand_news, user_clicked_news_index)
        
        # 子图嵌入
        news_graph_embedding = self.get_graph_embedding(news_graph, mode = "news") # bz, news_dim
        user_graph_embedding = self.get_graph_embedding(user_graph, mode = "user") # bz, news_dim
        graph_score = torch.sum(news_graph_embedding * user_graph_embedding, dim=-1).view(self.args.batch_size, -1) 
        
        #print(score)
        #print('------')
        #print(graph_score)
        score =  weight * score + (1 - weight) * graph_score        
        # 新闻用户表征
        # news_rep = torch.cat([news_rep, news_graph_embedding], dim=-1) # bz, 2 * news_dim
        # user_rep = torch.cat([user_rep, user_graph_embedding], dim=-1) # bz, 2 * news_dim
        # 预测得分
        # score = self.cos(news_rep, user_rep).view(self.args.batch_size, -1)
        # score = self.sigmod(torch.sum(news_rep * user_rep, dim = -1)).view(self.args.batch_size, -1)
        # score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score
