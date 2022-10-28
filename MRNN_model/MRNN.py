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

class MRNN(torch.nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 new_title_word_index, word_embedding, new_category_index, new_subcategory_index,
                 device):
        super(MRNN, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding
        self.entity_embedding = self.entity_embedding.to(device)
        self.relation_embedding = self.relation_embedding.to(device)

        # embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size).to(device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size).to(device)

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

    def get_news_entities_batch(self, newsids):
        news_entities = []
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                news_entities[-1].append([])
                news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.new_entity_size])
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
        new_rep = self.new_encoder(candidate_new_word_embedding.squeeze(), candidate_new_entity_embedding,
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
        return user_rep.squeeze(), new_rep

    def forward(self, cand_news, user_clicked_news_index):
        cand_news = torch.flatten(cand_news, 0, 1).unsqueeze(-1).to(self.device)
        user_clicked_news_index = user_clicked_news_index.unsqueeze(1)
        user_clicked_news_index = user_clicked_news_index.expand(user_clicked_news_index.shape[0], 5,
                                                                 user_clicked_news_index.shape[2])
        user_clicked_news_index = torch.flatten(user_clicked_news_index, 0, 1).to(self.device)
        # 新闻用户表征
        user_rep, news_rep = self.get_user_news_rep(cand_news, user_clicked_news_index)
        # 预测得分
        # score = self.cos(news_rep, user_rep).view(self.args.batch_size, -1)
        # score = self.sigmod(torch.sum(news_rep * user_rep, dim = -1)).view(self.args.batch_size, -1)
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score
