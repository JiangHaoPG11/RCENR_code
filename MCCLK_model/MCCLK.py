import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class Aggregator(nn.Module):
    def __init__(self,device, n_users, news_entity_dict, entity_adj, relation_adj):
        super(Aggregator, self).__init__()
        self.n_users = n_users
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

    def forward(self, user_emb, news_embeding, entity_emb, relation_emb, interact_mat):
        newsid, news_entities, news_relations = self.get_news_entities_batch()
        news_embeding = news_embeding[newsid]
        news_neigh_entities_embedding = entity_emb[news_entities]
        news_neigh_relation_embedding = relation_emb[news_relations]

        # ------------calculate attention weights ---------------
        news_neigh_weight = self.calculate_sim_hrt(news_embeding, news_neigh_entities_embedding,
                                                   news_neigh_relation_embedding)
        news_neigh_emb_weight = F.softmax(news_neigh_weight, dim=-1)
        news_neigh_emb = torch.mul(news_neigh_emb_weight.unsqueeze(-1), news_neigh_entities_embedding)
        news_agg = torch.sum(news_neigh_emb, dim = 1)

        entities, neigh_entities, neigh_relations = self.get_entities_neigh_batch(n_entity = len(entity_emb))
        entities_embedding = entity_emb[entities]
        neigh_entities_embedding = entity_emb[neigh_entities]
        neigh_relation_embedding = relation_emb[neigh_relations]

        # ------------calculate attention weights ---------------
        neigh_weight = self.calculate_sim_hrt(entities_embedding, neigh_entities_embedding, neigh_relation_embedding)
        neigh_emb_weight = F.softmax(neigh_weight, dim=-1)
        neigh_emb = torch.mul(neigh_emb_weight.unsqueeze(-1), neigh_entities_embedding)
        entity_agg = torch.sum(neigh_emb, dim=1)

        user_agg = torch.sparse.mm(interact_mat, news_agg)
        score = torch.mm(user_emb, news_agg.t())
        score = torch.softmax(score, dim=-1)
        user_agg = user_agg + (torch.mm(score, news_agg)) * user_agg
        return news_agg, entity_agg, user_agg

    def calculate_sim_hrt(self, emb_head, emb_tail, relation_emb):
        tail_relation_emb = emb_tail * relation_emb
        #tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = emb_head.unsqueeze(1).repeat(1, emb_tail.shape[1], 1) * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(tail_relation_emb, torch.transpose(head_relation_emb, -2, -1)).squeeze()
        att_weights = att_weights ** 2
        return att_weights


class GraphConv(nn.Module):
    def __init__(self, device, channel,
                 n_hops, n_users,
                 n_relations, interact_mat,
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

        for i in range(n_hops):
            self.convs.append(Aggregator(self.device, n_users, news_entity_dict, entity_adj, relation_adj))
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

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

    def forward(self, user_embedding, news_embedding, entity_embedding, relation_embedding,
                interact_mat, mess_dropout=True, node_dropout=False):
        # """node dropout"""
        # if node_dropout:
        #     edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)

        interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        # ----------------build item-item graph-------------------
        origin_item_adj = self.build_adj(news_embedding, self.topk)

        news_res_emb  = news_embedding
        entity_res_emb = entity_embedding  # [n_entity, channel]
        user_res_emb = user_embedding  # [n_users, channel]
        for i in range(len(self.convs)):
            news_agg, entity_emb, user_emb = self.convs[i](user_embedding, news_embedding,
                                                           entity_embedding, relation_embedding,
                                                           interact_mat)
            if mess_dropout:
                news_emb = self.dropout(news_agg)
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)

            news_emb = F.normalize(news_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            news_res_emb = torch.add(news_res_emb, news_emb)

        # update item-item graph
        item_adj = (1 - self.lambda_coeff) * self.build_adj(news_res_emb, self.topk) + self.lambda_coeff * origin_item_adj
        return entity_res_emb, user_res_emb, news_res_emb, item_adj

    def build_adj(self, context, topk):
        # construct similarity adj matrix
        n_entity = context.shape[0]
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)

        # adj_matrix = (torch.zeros_like(sim)).scatter_(-1, knn_ind, knn_val)
        knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)

        y = knn_ind.reshape(-1)
        x = torch.arange(0, n_entity).unsqueeze(dim=-1).to(self.device)
        x = x.expand(n_entity, topk).reshape(-1)
        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entity, n_entity])).to(self.device)

        # normalized laplacian adj
        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_entity).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_entity)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entity, n_entity]))
        L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity), d_mat_inv_sqrt)
        return L_norm

class MCCLK_model(nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict,  entity_adj, relation_adj, user_click_dict):
        super(MCCLK_model, self).__init__()

        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_users = args.user_size
        self.n_items = args.title_num
        self.n_relations = len(relation_embedding)
        self.n_entities = len(entity_embedding)

        self.user_embedding = nn.Embedding(self.n_users, self.args.embedding_size)
        news_title_embedding = news_title_embedding.tolist()
        news_title_embedding.append(np.random.normal(-0.1, 0.1, 768))
        self.news_title_embedding = torch.FloatTensor(news_title_embedding)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.interact_mat = user_click_dict
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.interact_mat).to(self.device)

        self.decay = args.l2
        self.sim_decay = args.sim_regularity
        self.emb_size = args.embedding_size
        self.context_hops = args.context_hops
        self.node_dropout = args.node_dropout
        self.node_dropout_rate = args.node_dropout_rate
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.ind = args.ind

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

    def _init_model(self):
        return GraphConv(device=self.device,
                         channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         news_entity_dict=self.news_entity_dict,
                         entity_adj=self.entity_adj,
                         relation_adj=self.relation_adj,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        adj = np.zeros([self.n_users, self.n_items+1])
        for i in range(len(X)):
            news_index = X[i]
            for j in news_index:
                if j != self.n_items-1:
                    adj[i][j] = 1
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


        self.user_embeddings = self.user_embedding(torch.IntTensor(np.linspace(0, self.n_users - 1, self.n_users)).to(torch.int32).to(self.device))
        self.news_embeddings = torch.tanh(self.news_fc2(torch.tanh(self.news_fc1(self.news_title_embedding.to(self.device)))))
        self.entity_embeddings = self.entity_embedding(torch.IntTensor(np.linspace(0, self.n_entities - 1, self.n_entities)).to(torch.int32).to(self.device))
        self.relation_embeddings = self.relation_embedding(torch.IntTensor(np.linspace(0, self.n_relations - 1, self.n_relations)).to(torch.int32).to(self.device))
        entity_gcn_emb, user_gcn_emb, news_gcn_emb, item_adj = self.gcn(self.user_embeddings, self.news_embeddings,
                                                                        self.entity_embeddings, self.relation_embeddings,
                                                                        self.interact_mat,
                                                                        mess_dropout=self.mess_dropout,
                                                                        node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user_index]
        i_e = news_gcn_emb[candidate_newsindex]
        i_h = self.news_embeddings
        for i in range(self.n_item_layer):
            i_h = torch.sparse.mm(item_adj, i_h)
        i_h = F.normalize(i_h, p=2, dim=1)
        i_e_1 = i_h[candidate_newsindex]

        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()
        x = indice_old[0, :]
        y = indice_old[1, :]
        x_A = x
        y_A = y + self.n_users
        x_A_T = y + self.n_users
        y_A_T = x
        x_new = torch.cat((x_A, x_A_T), dim=-1)
        y_new = torch.cat((y_A, y_A_T), dim=-1)
        indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        value_new = torch.cat((value_old, value_old), dim=-1)
        interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
            [self.n_users + self.n_items, self.n_users + self.n_items]))
        user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(self.user_embeddings, self.news_embeddings, interact_graph)
        u_e_2 = user_lightgcn_emb[user_index]
        i_e_2 = item_lightgcn_emb[candidate_newsindex]

        item_1 = self.user_embeddings[user_index]
        user_1 = self.news_embeddings[candidate_newsindex]
        loss_contrast = self.calculate_loss(i_e_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_1(item_1, i_e_2)
        loss_contrast = loss_contrast + self.calculate_loss_2(user_1, u_e_2)

        u_e = torch.cat((u_e, u_e_2, u_e_2), dim=-1)
        i_e = torch.cat((i_e, i_e_1, i_e_2), dim=-1)
        return self.create_bpr_loss(u_e, i_e, labels, loss_contrast)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6  # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = - torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        ret = loss_1
        ret = ret.mean()
        return ret

    def calculate_loss_1(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6  # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def calculate_loss_2(self, A_embedding, B_embedding):
        # first calculate the sim rec
        tau = 0.6  # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc3(A_embedding)
        B_embedding = self.fc3(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def light_gcn(self, user_embedding, item_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, items, labels, loss_contrast):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(scores.view(-1, 5), torch.argmax(labels.to(self.device), dim=1))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor
        return rec_loss + emb_loss + 0.001 * loss_contrast, scores.view(-1, 5), rec_loss, emb_loss

    def test(self, user_index,  candidate_newsindex):
        candidate_newsindex = candidate_newsindex[:,0].to(self.device)
        user_index = user_index.to(self.device)

        self.user_embeddings = self.user_embedding(torch.IntTensor(np.linspace(0, self.n_users - 1, self.n_users)).to(torch.int32).to(self.device))
        self.news_embeddings = torch.tanh(self.news_fc2(torch.tanh(self.news_fc1(self.news_title_embedding.to(self.device)))))
        self.entity_embeddings = self.entity_embedding(torch.IntTensor(np.linspace(0, self.n_entities - 1, self.n_entities)).to(torch.int32).to(self.device))
        self.relation_embeddings = self.relation_embedding(torch.IntTensor(np.linspace(0, self.n_relations - 1, self.n_relations)).to(torch.int32).to(self.device))
        entity_gcn_emb, user_gcn_emb, news_gcn_emb, item_adj = self.gcn(self.user_embeddings, self.news_embeddings,
                                                                        self.entity_embeddings, self.relation_embeddings,
                                                                        self.interact_mat,
                                                                        mess_dropout=self.mess_dropout,
                                                                        node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user_index]
        i_e = news_gcn_emb[candidate_newsindex]
        i_h = self.news_embeddings
        for i in range(self.n_item_layer):
            i_h = torch.sparse.mm(item_adj, i_h)
        i_h = F.normalize(i_h, p=2, dim=1)
        i_e_1 = i_h[candidate_newsindex]

        interact_mat_new = self.interact_mat
        indice_old = interact_mat_new._indices()
        value_old = interact_mat_new._values()
        x = indice_old[0, :]
        y = indice_old[1, :]
        x_A = x
        y_A = y + self.n_users
        x_A_T = y + self.n_users
        y_A_T = x
        x_new = torch.cat((x_A, x_A_T), dim=-1)
        y_new = torch.cat((y_A, y_A_T), dim=-1)
        indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        value_new = torch.cat((value_old, value_old), dim=-1)
        interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
            [self.n_users + self.n_items, self.n_users + self.n_items]))
        user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(self.user_embeddings, self.news_embeddings, interact_graph)
        u_e_2 = user_lightgcn_emb[user_index]
        i_e_2 = item_lightgcn_emb[candidate_newsindex]

        u_e = torch.cat((u_e, u_e_2, u_e_2), dim=-1)
        i_e = torch.cat((i_e, i_e_1, i_e_2), dim=-1)

        scores = (i_e * u_e).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
