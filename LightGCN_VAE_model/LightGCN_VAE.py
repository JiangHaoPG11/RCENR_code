import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils.utils import *

class LightGCN_VAE_model(nn.Module):
    def __init__(self, args, news_title_embedding, word_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj,
                 relation_adj, user_click_dict, new_title_word_index, new_category_index, new_subcategory_index):
        super(LightGCN_VAE_model, self).__init__()
        # ###########
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.n_users = args.user_size
        self.n_items = args.title_num
        self.n_relations = len(relation_embedding)
        self.n_entities = len(entity_embedding)
        self.Graph = self._convert_sp_mat_to_sp_tensor(user_click_dict)

        # ID嵌入
        self.news_embedding = nn.Embedding(self.n_items, self.args.embedding_size)
        self.user_embedding = nn.Embedding(self.n_users, self.args.embedding_size)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.news_embedding.weight, std=0.1)
        
        # Feature嵌入
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

        # 参数
        self.n_layers = self.args.lgn_layers
        self.keep_prob = self.args.keep_prob
        self.decay = args.l2
        self.A_split = False
        self.sig = nn.Sigmoid()
        self._init_model()

    def _init_model(self):
        self.news_fc1 = nn.Linear(400, self.args.embedding_size)
        self.news_fc2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

        self.vae_user_encoder_fea_mean = nn.Linear(self.args.embedding_size * 3, 50)
        self.vae_user_encoder_fea_var = nn.Linear(self.args.embedding_size * 3, 50)
        self.vae_user_encoder_id_mean = nn.Linear(self.args.embedding_size , 50)
        self.vae_user_encoder_id_var = nn.Linear(self.args.embedding_size , 50)

        self.vae_news_encoder_fea_mean = nn.Linear(self.args.embedding_size * 3, 50)
        self.vae_news_encoder_fea_var = nn.Linear(self.args.embedding_size * 3, 50)
        self.vae_news_encoder_id_mean = nn.Linear(self.args.embedding_size, 50)
        self.vae_news_encoder_id_var = nn.Linear(self.args.embedding_size , 50)

        self.vae_user_decoder_fea = nn.Linear(50, self.args.embedding_size * 3)
        self.vae_user_decoder_id = nn.Linear(50, self.args.embedding_size )
        self.vae_news_decoder_fea = nn.Linear(50, self.args.embedding_size * 3)
        self.vae_news_decoder_id = nn.Linear(50, self.args.embedding_size )

        self.entity_attention = Additive_Attention(self.args.query_vector_dim, self.args.embedding_size)

    def _convert_sp_mat_to_sp_tensor(self, user_click_dict):
        adj = np.zeros([self.n_users + self.n_items, self.n_users + self.n_items])
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.n_items - 1:
                    adj[i][j] = 1
                    adj[j][i] = 1
        X = sp.csr_matrix(adj, dtype=np.float32)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def LightGCN_VAE_forward(self):
        user_emds =  self.user_embedding.weight
        item_emds = self.news_embedding.weight
        all_emds = torch.cat([user_emds, item_emds])
        embs = [all_emds]
        if self.args.dropout:
            if self.training:
                g_dropped = self.__dropout(self.keep_prob)
            else:
                g_dropped = self.Graph
        else:
            g_dropped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_dropped)):
                    temp_emb.append(torch.sparse.mm(g_dropped[f], all_emds))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emds = side_emb
            else:
                all_emds = torch.sparse.mm(g_dropped, all_emds)

            embs.append(all_emds)
        embs = torch.stack(embs, dim=1)
        lgn_out = torch.mean(embs, dim=1)
        users, items = torch.split(lgn_out, [self.n_users, self.n_items])
        return users, items

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
            vae_score = (newsFeaZ * userFeaZ).sum(dim=-1)
            return vae_score, kl_loss
        else:
            vae_score = (newsFeaZ * userFeaZ[:,0,:]).sum(dim=-1)
            return vae_score

    def get_user_news_feature_embedding(self, news_feature_list, user_feature_list):
        news_entity_rep = torch.tanh(self.entity_attention(news_feature_list[0]))
        news_category_rep = news_feature_list[1]
        news_subcategory_rep = news_feature_list[2]
        news_feature_embedding = torch.cat([news_entity_rep, news_category_rep, news_subcategory_rep], dim=-1)

        user_entity_rep = torch.mean(torch.tanh(self.entity_attention(torch.flatten(user_feature_list[0], 0, 1))).view(-1, 50, 100), dim =1)
        user_category_rep = torch.mean(user_feature_list[1], dim =1)
        user_subcategory_rep = torch.mean(user_feature_list[2], dim=1)
        user_feature_embedding = torch.cat([user_entity_rep, user_category_rep, user_subcategory_rep], dim=-1).unsqueeze(1).repeat(1, 5, 1)
        return news_feature_embedding, user_feature_embedding
    
    def create_bpr_loss(self, users, items, vae_score, kl_loss, labels):
        batch_size = users.shape[0]
        scores = (items * users.unsqueeze(1)).sum(dim=-1).view(-1, 5) + vae_score
        # scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(scores, torch.argmax(labels.to(self.device), dim=1))
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        return rec_loss + emb_loss + 0.1 * kl_loss, scores.view(-1, 5), rec_loss, emb_loss

    def forward(self,  user_index, candidate_news_index, user_clicked_news_index, label):
        candidate_news_index = candidate_news_index.to(self.device)
        user_index = user_index.to(self.device)

        candidate_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(candidate_news_index.cpu())).to(self.device)).to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(user_clicked_news_index.cpu())).to(self.device)).to(self.device).squeeze()
        candidate_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)
        candidate_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)
        
        # VAE
        news_feature_list = [candidate_news_entity_embedding, candidate_news_category_embedding, candidate_news_subcategory_embedding]
        user_feature_list = [user_clicked_news_entity_embedding, user_clicked_news_category_embedding, user_clicked_news_subcategory_embedding]
        news_feature_embedding, user_feature_embedding = self.get_user_news_feature_embedding(news_feature_list, user_feature_list)
        news_id_embedding = self.news_embedding(candidate_news_index)
        user_id_embedding = self.user_embedding(user_index.unsqueeze(1).repeat(1, 5))
        vae_score, kl_loss = self.dual_vae_net(user_id_embedding, news_id_embedding, user_feature_embedding, news_feature_embedding)
        
        # lightGCN
        all_users_embedding, all_news_embedding = self.LightGCN_VAE_forward()
        users_ID_embs = all_users_embedding[user_index.long()]
        news_ID_embs = all_news_embedding[candidate_news_index.long()]
        
        return self.create_bpr_loss(users_ID_embs, news_ID_embs, vae_score, kl_loss, label)

    def test(self,user_index, candidate_news_index, user_clicked_news_index):
        candidate_news_index = candidate_news_index[:,0].to(self.device).unsqueeze(-1)
        user_index = user_index.to(self.device)

        candidate_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(candidate_news_index.cpu())).to(self.device)).to(self.device).squeeze(1)
        user_clicked_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(user_clicked_news_index.cpu())).to(self.device)).to(self.device).squeeze(1)
        candidate_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)
        candidate_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)

        news_feature_list = [candidate_news_entity_embedding, candidate_news_category_embedding, candidate_news_subcategory_embedding]
        user_feature_list = [user_clicked_news_entity_embedding, user_clicked_news_category_embedding, user_clicked_news_subcategory_embedding]
        news_feature_embedding, user_feature_embedding = self.get_user_news_feature_embedding(news_feature_list, user_feature_list)
        news_id_embedding = self.news_embedding(candidate_news_index)
        user_id_embedding = self.user_embedding(user_index)
        vae_score = self.dual_vae_net(user_id_embedding, news_id_embedding, user_feature_embedding, news_feature_embedding)

        all_users_embedding, all_news_embedding = self.LightGCN_VAE_forward()
        users_ID_embs = all_users_embedding[user_index.long()]
        items_ID_embs = all_news_embedding[candidate_news_index.long()]
        # scores = (items_ID_embs * users_ID_embs.unsqueeze(1)).sum(dim=-1) + vae_score
        scores = (items_ID_embs * users_ID_embs.unsqueeze(1)).sum(dim=-1)
        return scores

