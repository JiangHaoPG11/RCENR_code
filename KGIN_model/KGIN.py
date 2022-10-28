import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class Aggregator(nn.Module):
    def __init__(self,device, n_users, n_factors, news_entity_dict, entity_adj, relation_adj):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.device = device

    def get_news_entities_batch(self):
        news_entities = []
        news_relations = []
        news = []
        for key, value in self.news_entity_dict.items():
            news.append(key)
            news_entities.append(value)
            news_relations.append([0 for k in range(len(value))])
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

    def forward(self, user_emb, all_embedding, entity_emb, relation_emb, latent_emb, interact_mat, weight, disen_weight_att):
        newsid, news_entities, news_relations = self.get_news_entities_batch()
        news_emb = all_embedding[newsid]
        news_neigh_entities_embedding = entity_emb[news_entities]
        news_neigh_relation_embedding = relation_emb[news_relations]
        news_agg = torch.mean(news_neigh_entities_embedding + news_neigh_relation_embedding, dim=1)

        entities, neigh_entities, neigh_relations = self.get_entities_neigh_batch(n_entity=len(entity_emb))
        entity_emb = all_embedding[entities]
        neigh_entities_embedding = entity_emb[neigh_entities]
        neigh_relation_embedding = relation_emb[neigh_relations]
        entity_agg = torch.mean(neigh_relation_embedding + neigh_entities_embedding, dim=1)

        node_emb = torch.cat([news_agg + news_emb, entity_agg + entity_emb])

        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        user_agg = torch.sparse.mm(interact_mat, node_emb)
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att), weight).expand(self.n_users, self.n_factors, 100)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]
        return node_emb, user_agg


class GraphConv(nn.Module):
    def __init__(self, device, channel,
                 n_hops, n_users,
                 n_relations, n_factors, interact_mat,
                 news_entity_dict,  entity_adj, relation_adj,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj =relation_adj
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.topk = 10
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        self.device = device
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(self.device, n_users, n_factors, news_entity_dict, entity_adj, relation_adj))
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

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_embedding, all_embedding, entity_embedding, relation_embedding, latent_emb,
                interact_mat, mess_dropout=True, node_dropout=False):
        # """node dropout"""
        # if node_dropout:
        #     edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        node_res_emb = all_embedding  # [n_node, channel]
        user_res_emb = user_embedding  # [n_users, channel]

        cor = self._cul_cor()

        for i in range(len(self.convs)):
            node_emb, user_emb = self.convs[i](user_embedding, all_embedding, entity_embedding, relation_embedding,
                                               latent_emb, interact_mat, self.weight, self.disen_weight_att)

            if mess_dropout:
                node_emb = self.dropout(node_emb)
                user_emb = self.dropout(user_emb)

            node_emb = F.normalize(node_emb)
            user_emb = F.normalize(user_emb)

            node_res_emb = torch.add(node_res_emb, node_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return  user_res_emb, node_res_emb, cor

class KGIN_model(nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict,  entity_adj, relation_adj, user_click_dict):
        super(KGIN_model, self).__init__()

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_users = args.user_size
        self.n_items = args.title_num
        self.n_relations = len(relation_embedding)
        self.n_entities = len(entity_embedding)

        self.user_embedding = nn.Embedding(self.n_users, self.args.embedding_size)
        self.news_title_embedding = torch.FloatTensor(news_title_embedding)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj

        self.interact_mat = self._convert_sp_mat_to_sp_tensor(user_click_dict, news_entity_dict).to(self.device)

        self.decay = args.l2
        self.sim_decay = args.sim_regularity
        self.emb_size = args.embedding_size
        self.context_hops = args.context_hops
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.ind = args.ind
        self.n_factors = args.n_factors

        self._init_weight()
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.gcn = self._init_model()
        self.lightgcn_layer = 2
        self.n_item_layer = 2
        self.alpha = 0.2
        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )

        self.news_fc1 = nn.Linear(768, self.args.embedding_size)
        self.news_fc2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

        # self.user_fc1 = nn.Linear(400, self.args.embedding_size)
        # self.user_fc2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

    def _concat_all_embedding(self):
        # user_embeddings = self.user_embedding(torch.IntTensor(np.linspace(0, self.n_users - 1, self.n_users)).to(self.device))
        # news_embeddings = torch.tanh(self.news_fc2(torch.tanh(self.news_fc1(self.news_title_embedding.to(self.device)))))
        # entity_embeddings = self.entity_embedding(torch.IntTensor(np.linspace(0, self.n_entities - 1, self.n_entities)).to(self.device))
        # relation_embeddings = self.relation_embedding(torch.IntTensor(np.linspace(0, self.n_relations - 1, self.n_relations)).to(self.device))

        user_embeddings = self.user_embedding.weight
        news_embeddings = torch.tanh(self.news_fc2(torch.tanh(self.news_fc1(self.news_title_embedding.to(self.device)))))
        entity_embeddings = self.entity_embedding.weight
        relation_embeddings = self.relation_embedding.weight

        all_embedding = torch.cat([news_embeddings, entity_embeddings], dim=0)
        return user_embeddings, all_embedding, entity_embeddings, relation_embeddings

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

    def _init_model(self):
        return GraphConv(device=self.device,
                         channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         news_entity_dict=self.news_entity_dict,
                         entity_adj=self.entity_adj,
                         relation_adj=self.relation_adj,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    # update
    def _convert_sp_mat_to_sp_tensor(self, user_click_dict, news_entity_dict):
        adj = np.zeros([self.n_users, self.n_items + self.n_entities])
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.n_items - 1:
                    adj[i][j] = 1
                    entity_list = news_entity_dict[i]
                    for m in entity_list:
                        adj[i][m + self.n_items] = 1
        print(adj)
        X = sp.csr_matrix(adj, dtype=np.float32)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, user_index,  candidate_newsindex, labels):
        candidate_newsindex = candidate_newsindex.to(self.device)
        user_index = user_index.unsqueeze(1).repeat(1, 5).to(self.device)
        user_index = torch.flatten(user_index, 0, 1)
        candidate_newsindex = torch.flatten(candidate_newsindex, 0, 1)

        user_embeddings, all_embedding, entity_embeddings, relation_embeddings = self._concat_all_embedding()

        user_gcn_emb, all_gcn_emb, cor  =  self.gcn(user_embeddings, all_embedding,
                                                                   entity_embeddings, relation_embeddings,
                                                                   self.latent_emb, self.interact_mat,
                                                                   mess_dropout=self.mess_dropout,
                                                                   node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user_index]
        i_e = all_gcn_emb[candidate_newsindex]
        return self.create_bpr_loss(u_e, i_e, labels, cor)


    def create_bpr_loss(self, users, items, labels, cor):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        #scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(F.softmax(scores.view(-1, 5), dim = -1), torch.argmax(labels.to(self.device), dim=1))
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor
        return rec_loss + emb_loss + cor_loss, scores.view(-1, 5), rec_loss, emb_loss

    def test(self, user_index,  candidate_newsindex):
        candidate_newsindex = candidate_newsindex[:,0].to(self.device)
        user_index = user_index.to(self.device)

        user_embeddings, all_embedding, entity_embeddings, relation_embeddings = self._concat_all_embedding()

        user_gcn_emb, all_gcn_emb, cor = self.gcn(user_embeddings, all_embedding,
                                                                   entity_embeddings, relation_embeddings,
                                                                   self.latent_emb,
                                                                   self.interact_mat,
                                                                   mess_dropout=self.mess_dropout,
                                                                   node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user_index]
        i_e = all_gcn_emb[candidate_newsindex]

        scores = (i_e * u_e).sum(dim=1)
        #scores = torch.sigmoid(scores)
        return scores
