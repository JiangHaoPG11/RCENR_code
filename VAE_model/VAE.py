import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils.utils import *

class VAE_model(nn.Module):
    def __init__(self, args, news_title_embedding, word_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj,
                 relation_adj, user_click_dict, new_title_word_index, new_category_index, new_subcategory_index):
        super(VAE_model, self).__init__()
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.n_users = args.user_size
        self.n_items = args.title_num
        self.n_relations = len(relation_embedding)
        self.n_entities = len(entity_embedding)

        # ID嵌入
        self.news_embedding = nn.Embedding(self.n_items, self.args.embedding_size)
        self.user_embedding = nn.Embedding(self.n_users, self.args.embedding_size)

        # Feature嵌入
        self.word_embedding = word_embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size)
        self.news_title_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(news_title_embedding))
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)

        # 特征字典
        self.news_title_word_dict = new_title_word_index
        self.news_category_dict = new_category_index
        self.news_subcategory_dict = new_subcategory_index
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.user_click_dict = user_click_dict

        # 图网络参数
        self.decay = args.l2
        self.context_hops = args.context_hops
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.ind = args.ind
        self.lightgcn_layer = 2
        self.n_item_layer = 2

        # 模型初始化
        self._init_model()

    def _init_model(self):
        self.news_fc1 = nn.Linear(400, self.args.embedding_size)
        self.news_fc2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

        self.vae_user_encoder_fea_mean = nn.Linear(self.args.embedding_size * 6, 50)
        self.vae_user_encoder_fea_var = nn.Linear(self.args.embedding_size *6, 50)
        self.vae_user_encoder_id_mean = nn.Linear(self.args.embedding_size , 50)
        self.vae_user_encoder_id_var = nn.Linear(self.args.embedding_size , 50)

        self.vae_news_encoder_fea_mean = nn.Linear(self.args.embedding_size * 6, 50)
        self.vae_news_encoder_fea_var = nn.Linear(self.args.embedding_size * 6, 50)
        self.vae_news_encoder_id_mean = nn.Linear(self.args.embedding_size, 50)
        self.vae_news_encoder_id_var = nn.Linear(self.args.embedding_size, 50)

        self.vae_user_decoder_fea = nn.Linear(50, self.args.embedding_size * 6)
        self.vae_user_decoder_id = nn.Linear(50, self.args.embedding_size )
        self.vae_news_decoder_fea = nn.Linear(50, self.args.embedding_size * 6)
        self.vae_news_decoder_id = nn.Linear(50, self.args.embedding_size )

        self.predict_layer = nn.Linear(50 * 4, 1)

        self.entity_attention = Additive_Attention(self.args.query_vector_dim, self.args.embedding_size)
        self.word_attention = Additive_Attention(self.args.query_vector_dim, 300)

    def get_news_entities_batch(self, newsids):
        news_entities = []
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                news_entities[-1].append([])
                #news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.new_entity_size])
                news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:])
        return np.array(news_entities)

    def dual_vae_net(self, user_id_embedding, news_id_embedding, user_fea_embedding, news_fea_embedding):
        def vae_net_user_id_encoder(user_id_embedding):
            mean = self.vae_user_encoder_id_mean(user_id_embedding)
            var = torch.sigmoid(self.vae_user_encoder_id_var(user_id_embedding))
            return mean, var
        def vae_net_news_id_encoder(news_id_embedding):
            mean = self.vae_news_encoder_id_mean(news_id_embedding)
            var = torch.sigmoid(self.vae_news_encoder_id_var(news_id_embedding))
            return mean, var
        def vae_net_user_fea_encoder(user_fea_embedding):
            mean = self.vae_user_encoder_fea_mean(user_fea_embedding)
            var = torch.sigmoid(self.vae_user_encoder_fea_var(user_fea_embedding))
            return mean, var
        def vae_net_news_fea_encoder(news_fea_embedding):
            mean = self.vae_news_encoder_fea_mean(news_fea_embedding)
            var = torch.sigmoid(self.vae_news_encoder_fea_var(news_fea_embedding))
            return mean, var
        def vae_net_user_id_decoder(user_id_embedding_sample):
            user_id_decoder_embedding = self.vae_user_decoder_id(user_id_embedding_sample)
            return user_id_decoder_embedding
        def vae_net_news_id_decoder(news_id_embedding_sample):
            news_id_decoder_embedding = self.vae_user_decoder_id(news_id_embedding_sample)
            return news_id_decoder_embedding
        def vae_net_user_fea_decoder(user_fea_embedding_sample):
            user_fea_decoder_embedding = self.vae_user_decoder_fea(user_fea_embedding_sample)
            return user_fea_decoder_embedding
        def vae_net_news_fea_decoder(news_fea_embedding_sample):
            user_id_decoder_embedding = self.vae_user_decoder_fea(news_fea_embedding_sample)
            return user_id_decoder_embedding
        def reparameterize(mean, var):
            eps = torch.randn(var.shape).to(self.device)
            std = torch.exp(var) ** 0.5
            z = mean + std * eps
            return z
        def cal_kl(meanq, varq, meanp, varp):
            kl_div = (0.5 * (varp ** -2) * ((meanq - meanp) ** 2 + varq ** 2)) + torch.log((varp + 1e-16) / (varq + 1e-16))
            kl_div = torch.mean(torch.reshape(torch.mean(kl_div, -1), [-1, 1]))
            return kl_div

        userIdMean, userIdVar = vae_net_user_id_encoder(user_id_embedding)
        newsIdMean, newsIdVar = vae_net_news_id_encoder(news_id_embedding)
        userIdZ = reparameterize(userIdMean, userIdVar)
        newsIdZ = reparameterize(newsIdMean, newsIdVar)
        userIdDecode = vae_net_user_id_decoder(userIdZ)
        newsIdDecode = vae_net_news_id_decoder(newsIdZ)

        userFeaMean, userFeaVar = vae_net_user_fea_encoder(user_fea_embedding)
        newsFeaMean, newsFeaVar = vae_net_news_fea_encoder(news_fea_embedding)
        userFeaZ = reparameterize(userFeaMean, userFeaVar)
        newsFeaZ = reparameterize(newsFeaMean, newsFeaVar)
        userFeaDecode = vae_net_user_fea_decoder(userFeaZ)
        newsFeaDecode = vae_net_news_fea_decoder(newsFeaZ)
        if self.training:
            # 计算kl散度
            user_kl = cal_kl(userIdMean, userIdVar, userFeaMean, userFeaVar)
            news_kl = cal_kl(newsIdMean, newsIdVar, newsFeaMean, newsFeaVar)
            user_kl_rep = cal_kl(userFeaMean, userFeaVar, torch.zeros_like(userFeaMean), torch.ones_like(userFeaVar))
            news_kl_rep = cal_kl(newsFeaMean, newsFeaVar, torch.zeros_like(newsFeaMean), torch.ones_like(newsFeaVar))

            # 计算kl-Loss
            kl_loss = (user_kl + news_kl + user_kl_rep + news_kl_rep)
            final_vector = torch.cat([newsFeaZ, newsIdZ, userFeaZ, userIdZ], dim=-1)
            vae_score = self.predict_layer(final_vector).squeeze()

            # vae_score = (newsFeaZ * userFeaZ).sum(dim=-1)
            return vae_score, kl_loss
        else:
            final_vector = torch.cat([newsFeaZ, newsIdZ, userFeaZ[:, 0, :].unsqueeze(1), userIdZ.unsqueeze(1)],dim=-1)
            vae_score = self.predict_layer(final_vector).squeeze()
            # vae_score = (newsFeaZ * userFeaZ[:, 0, :]).sum(dim=-1)
            return vae_score

    def get_user_news_feature_embedding(self, news_feature_list, user_feature_list):
        news_word_rep = torch.tanh(self.word_attention(news_feature_list[0]))
        news_entity_rep = torch.tanh(self.entity_attention(news_feature_list[1]))
        news_category_rep = news_feature_list[2]
        news_subcategory_rep = news_feature_list[3]
        news_feature_embedding = torch.cat([news_word_rep, news_entity_rep, news_category_rep, news_subcategory_rep], dim=-1)
        user_word_rep = torch.mean(torch.tanh(self.word_attention(torch.flatten(user_feature_list[0], 0, 1))).view(-1, 50, 300), dim=1)
        user_entity_rep = torch.mean(torch.tanh(self.entity_attention(torch.flatten(user_feature_list[1], 0, 1))).view(-1,50, 100), dim =1)
        user_category_rep = torch.mean(user_feature_list[2], dim=1)
        user_subcategory_rep = torch.mean(user_feature_list[3], dim=1)
        user_feature_embedding = torch.cat([user_word_rep, user_entity_rep, user_category_rep, user_subcategory_rep], dim=-1).unsqueeze(1).repeat(1, 5, 1)
        return news_feature_embedding, user_feature_embedding

    def forward(self, user_index, candidate_news_index, user_clicked_news_index, labels):
        candidate_news_index = candidate_news_index.to(self.device)
        user_index = user_index.to(self.device)

        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index.cpu()]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index.cpu()]].to(self.device)
        candidate_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(candidate_news_index.cpu())).to(self.device)).to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(user_clicked_news_index.cpu())).to(self.device)).to(self.device).squeeze()
        candidate_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)
        candidate_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)

        news_feature_list = [candidate_news_word_embedding, candidate_news_entity_embedding, candidate_news_category_embedding, candidate_news_subcategory_embedding]
        user_feature_list = [user_clicked_news_word_embedding, user_clicked_news_entity_embedding, user_clicked_news_category_embedding, user_clicked_news_subcategory_embedding]
        news_feature_embedding, user_feature_embedding = self.get_user_news_feature_embedding(news_feature_list, user_feature_list)
        news_id_embedding = self.news_embedding(candidate_news_index)
        user_id_embedding = self.user_embedding(user_index.unsqueeze(1).repeat(1, 5))
        vae_score, kl_loss = self.dual_vae_net(user_id_embedding, news_id_embedding, user_feature_embedding, news_feature_embedding)

        return self.create_bpr_loss(vae_score, kl_loss, labels)


    def create_bpr_loss(self, vae_score, kl_loss, labels):
        scores = vae_score
        scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(scores, torch.argmax(labels.to(self.device), dim=1))
        return rec_loss + kl_loss, scores.view(-1, 5), rec_loss

    def test(self, user_index, candidate_news_index, user_clicked_news_index):
        candidate_news_index = candidate_news_index[:,0].to(self.device).unsqueeze(-1)
        user_index = user_index.to(self.device)
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index.cpu()]].to(
            self.device)
        user_clicked_news_word_embedding = self.word_embedding[
            self.news_title_word_dict[user_clicked_news_index.cpu()]].to(self.device)
        candidate_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(candidate_news_index.cpu())).to(self.device)).to(self.device).squeeze(1)
        user_clicked_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(user_clicked_news_index.cpu())).to(self.device)).to(self.device).squeeze(1)
        candidate_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)
        candidate_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)

        news_feature_list = [candidate_news_word_embedding, candidate_news_entity_embedding,
                             candidate_news_category_embedding, candidate_news_subcategory_embedding]
        user_feature_list = [user_clicked_news_word_embedding, user_clicked_news_entity_embedding,
                             user_clicked_news_category_embedding, user_clicked_news_subcategory_embedding]

        news_feature_embedding, user_feature_embedding = self.get_user_news_feature_embedding(news_feature_list, user_feature_list)
        news_id_embedding = self.news_embedding(candidate_news_index)
        user_id_embedding = self.user_embedding(user_index)
        vae_score = self.dual_vae_net(user_id_embedding, news_id_embedding, user_feature_embedding, news_feature_embedding)
        scores = vae_score
        # scores = F.softmax(scores.view(-1, 5), dim=-1)
        return scores
