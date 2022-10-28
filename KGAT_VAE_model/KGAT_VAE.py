import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from utils.utils import *

class Aggregator(nn.Module):
    def __init__(self, device, news_entity_dict, entity_adj, relation_adj, category_adj, subcategory_adj, entity_num, category_num, subcategory_num):
        super(Aggregator, self).__init__()
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.category_adj = category_adj
        self.subcategory_adj = subcategory_adj
        self.entity_num = entity_num
        self.category_num = category_num
        self.subcategory_num = subcategory_num
        self.device = device
        self.news_atten = nn.Linear(100, 1)
        self.entity_atten = nn.Linear(100, 1)

    def get_news_entities(self):
        news_entities = []
        news_relations = []
        news = []
        for key, value in self.news_entity_dict.items():
            news_entities.append([])
            news_relations.append([])
            newsindex = key
            news_entities_list = value
            news.append(newsindex)
            news_entities[-1].extend(news_entities_list)
            news_relations[-1].extend([0 for k in range(len(news_entities_list))])
            news_entities[-1].append(self.category_adj[newsindex] + self.entity_num)
            news_relations[-1].append(0)
            news_entities[-1].append(self.subcategory_adj[newsindex] + self.entity_num + self.category_num)
            news_relations[-1].append(0)
        news = torch.tensor(news).to(self.device)
        news_entities = torch.tensor(news_entities).to(self.device)
        news_relations = torch.tensor(news_relations).to(self.device)# bz, news_entity_num
        return news, news_entities, news_relations

    def get_entities_neigh_batch(self, n_entity):
        neigh_entities = []
        neigh_relations = []
        entities = []
        for i in range(n_entity):
            if i in self.entity_adj.keys():
                entities.append(i)
                neigh_entities.append(self.entity_adj[i])
                neigh_relations.append(self.relation_adj[i])
            else:
                entities.append(i)
                neigh_entities.append([0 for k in range(20)])
                neigh_relations.append([0 for k in range(20)])
        entities = torch.tensor(entities).to(self.device)
        neigh_entities = torch.tensor(neigh_entities).to(self.device)
        neigh_relations = torch.tensor(neigh_relations).to(self.device)# bz, news_entity_num
        return entities, neigh_entities, neigh_relations

    def forward(self, user_emb, all_embedding, entity_emb, relation_emb, interact_mat):

        newsid, news_entities, news_relations = self.get_news_entities()
        news_emb = all_embedding[newsid]
        news_neigh_entities_embedding = entity_emb[news_entities]
        news_neigh_relation_embedding = relation_emb[news_relations]
        news_weight = F.softmax(self.news_atten(news_neigh_entities_embedding + news_neigh_relation_embedding), dim = -1)
        news_agg = torch.matmul(torch.transpose(news_weight, -1, -2), news_neigh_entities_embedding).squeeze()

        entities, neigh_entities, neigh_relations = self.get_entities_neigh_batch(n_entity = len(entity_emb))
        entity_emb = all_embedding[entities]
        neigh_entities_embedding = entity_emb[neigh_entities]
        neigh_relation_embedding = relation_emb[neigh_relations]
        entity_weight = F.softmax(self.entity_atten(neigh_relation_embedding + neigh_entities_embedding), dim = -1)
        entity_agg = torch.matmul(torch.transpose(entity_weight, -1, -2), neigh_entities_embedding).squeeze()

        node_emb = torch.cat([news_agg + news_emb, entity_agg + entity_emb])
        user_agg = torch.sparse.mm(interact_mat, node_emb)
        user_agg = user_emb + user_agg  # [n_users, channel]
        return node_emb, user_agg


class KGAT_VAE(nn.Module):
    def __init__(self, device,  n_hops, n_users, n_relations, n_entities, n_category, n_subcategory, interact_mat,
                 news_entity_dict,  entity_adj, relation_adj, category_adj, subcategory_adj,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(KGAT_VAE, self).__init__()

        self.aggs = nn.ModuleList()
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        self.device = device

        for i in range(n_hops):
            self.aggs.append(Aggregator(self.device, news_entity_dict, entity_adj, relation_adj, category_adj, subcategory_adj, n_entities, n_category, n_subcategory))
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    # def _edge_sampling(self, edge_index, edge_type, rate=0.5):
    #     # edge_index: [2, -1]
    #     # edge_type: [-1]
    #     n_edges = edge_index.shape[1]
    #     random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
    #     return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embedding, all_embedding, entity_embedding, relation_embedding, mess_dropout=True, node_dropout=False):
        # """node dropout"""
        # if node_dropout:
        #     edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        node_res_emb = all_embedding  # [n_node, channel]
        user_res_emb = user_embedding  # [n_users, channel]

        for i in range(len(self.aggs)):
            node_emb, user_emb = self.aggs[i](user_embedding, all_embedding, entity_embedding, relation_embedding, self.interact_mat)
            if mess_dropout:
                node_emb = self.dropout(node_emb)
                user_emb = self.dropout(user_emb)

            node_emb = F.normalize(node_emb)
            user_emb = F.normalize(user_emb)

            node_res_emb = torch.add(node_res_emb, node_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return user_res_emb, node_res_emb

class KGAT_VAE_model(nn.Module):
    def __init__(self, args, news_title_embedding, word_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj,
                 relation_adj, user_click_dict, new_title_word_index, new_category_index, new_subcategory_index):
        super(KGAT_VAE_model, self).__init__()
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

        # 交互矩阵
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.user_click_dict, self.news_entity_dict,
                                                              self.news_category_dict, self.news_subcategory_dict).to(self.device)

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

        self.KGAT_VAE = KGAT_VAE(
                                device=self.device,
                                n_hops=self.context_hops,
                                n_users=self.n_users,
                                n_relations=self.n_relations,
                                n_entities=self.n_entities,
                                n_category=self.args.category_num,
                                n_subcategory=self.args.subcategory_num,
                                interact_mat=self.interact_mat,
                                news_entity_dict=self.news_entity_dict,
                                entity_adj=self.entity_adj,
                                relation_adj=self.relation_adj,
                                category_adj=self.news_category_dict,
                                subcategory_adj=self.news_subcategory_dict,
                                ind=self.ind,
                                node_dropout_rate=self.node_dropout_rate,
                                mess_dropout_rate=self.mess_dropout_rate)

    def _concat_all_embedding(self):
        user_embeddings = self.user_embedding.weight
        news_embeddings = self.news_embedding.weight
        entity_embeddings = torch.cat([self.entity_embedding.weight, self.category_embedding.weight ,self.subcategory_embedding.weight], dim = 0)

        relation_embeddings = self.relation_embedding.weight
        all_embedding = torch.cat([news_embeddings, entity_embeddings], dim=0)
        return user_embeddings, all_embedding, entity_embeddings, relation_embeddings

    # update
    def _convert_sp_mat_to_sp_tensor(self, user_click_dict, news_entity_dict, news_category_dict, news_subcategory_dict):
        adj = np.zeros([self.n_users, self.n_items + self.n_entities + self.args.category_num + self.args.subcategory_num])
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.n_items - 1:
                    adj[i][j] = 1
                    entity_list = news_entity_dict[i]
                    for m in entity_list:
                        adj[i][m + self.n_items] = 1
                    category_index = news_category_dict[i]
                    adj[i][category_index + self.n_items + self.n_entities] = 1
                    subcategory_index = news_subcategory_dict[i]
                    adj[i][subcategory_index + self.n_items + self.n_entities + self.args.category_num] = 1

        X = sp.csr_matrix(adj, dtype=np.float32)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

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
        user_entity_rep = torch.mean(torch.tanh(self.entity_attention(torch.flatten(user_feature_list[0], 0, 1))).view(-1, 10, 100), dim =1)
        user_category_rep = torch.mean(user_feature_list[1], dim =1)
        user_subcategory_rep = torch.mean(user_feature_list[2], dim=1)
        user_feature_embedding = torch.cat([user_entity_rep, user_category_rep, user_subcategory_rep], dim=-1).unsqueeze(1).repeat(1, 5, 1)
        return news_feature_embedding, user_feature_embedding

    def forward(self, user_index, candidate_news_index, user_clicked_news_index, labels):
        candidate_news_index = candidate_news_index.to(self.device)
        user_index = user_index.to(self.device)

        candidate_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(candidate_news_index.cpu())).to(self.device)).to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding(torch.IntTensor(self.get_news_entities_batch(user_clicked_news_index.cpu())).to(self.device)).to(self.device).squeeze()
        candidate_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_category_embedding = self.category_embedding(torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)
        candidate_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)).to(self.device)
        user_clicked_news_subcategory_embedding = self.subcategory_embedding(torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)).to(self.device)

        news_feature_list = [candidate_news_entity_embedding, candidate_news_category_embedding, candidate_news_subcategory_embedding]
        user_feature_list = [user_clicked_news_entity_embedding, user_clicked_news_category_embedding, user_clicked_news_subcategory_embedding]
        news_feature_embedding, user_feature_embedding = self.get_user_news_feature_embedding(news_feature_list, user_feature_list)
        news_id_embedding = self.news_embedding(candidate_news_index)
        user_id_embedding = self.user_embedding(user_index.unsqueeze(1).repeat(1, 5))
        vae_score, kl_loss = self.dual_vae_net(user_id_embedding, news_id_embedding, user_feature_embedding, news_feature_embedding)

        user_index = torch.flatten(user_index.unsqueeze(1).repeat(1, 5), 0, 1)
        candidate_news_index = torch.flatten(candidate_news_index, 0, 1)

        user_embeddings, all_embedding, entity_embeddings, relation_embeddings = self._concat_all_embedding()
        user_KGAT_VAE_emb, node_KGAT_VAE_emb = self.KGAT_VAE(user_embeddings, all_embedding,
                                                                entity_embeddings, relation_embeddings,
                                                                mess_dropout=self.mess_dropout)
        u_e = user_KGAT_VAE_emb[user_index]
        i_e = node_KGAT_VAE_emb[candidate_news_index]
        return self.create_bpr_loss(u_e, i_e, vae_score, kl_loss, labels)


    def create_bpr_loss(self, users, items, vae_score, kl_loss, labels):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1).view(-1, 5) + vae_score

        # scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(scores, torch.argmax(labels.to(self.device), dim=1))

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        return rec_loss + emb_loss + 0.1 * kl_loss, scores.view(-1, 5), rec_loss, emb_loss

    def test(self, user_index, candidate_news_index, user_clicked_news_index):
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

        user_embeddings, all_embedding, entity_embeddings, relation_embeddings = self._concat_all_embedding()
        user_KGAT_VAE_emb, node_KGAT_VAE_emb = self.KGAT_VAE(user_embeddings, all_embedding,
                                                                entity_embeddings, relation_embeddings,
                                                                mess_dropout=self.mess_dropout)
        u_e = user_KGAT_VAE_emb[user_index]
        i_e = node_KGAT_VAE_emb[candidate_news_index].squeeze()
        scores = (i_e * u_e).sum(dim=1) + vae_score
        # scores = F.softmax(scores.view(-1, 5), dim=-1)
        return scores
