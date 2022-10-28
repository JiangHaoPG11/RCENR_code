import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

class LightGCL_model(nn.Module):
    def __init__(self, args, news_title_embedding, user_click_dict):
        super(LightGCL_model, self).__init__()
        self.args = args
        self.news_title_embedding = news_title_embedding
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__init_weight()
        self.UIGraph, self.IUGraph = self._convert_sp_mat_to_sp_tensor(user_click_dict)
        self.UIGraph_svd, self.IUGraph_svd = self.svd(self.UIGraph)

    def __init_weight(self):
        self.num_users = self.args.user_size
        self.num_items = self.args.title_num
        self.latent_dim = self.args.embedding_size
        self.n_layers = self.args.lgn_layers
        self.keep_prob = self.args.keep_prob
        self.A_split = False
        self.user_emds = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.news_title_embedding.shape[1])
        self.item_emds = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.news_title_embedding.shape[1])
        nn.init.normal_(self.user_emds.weight, std=0.1)
        nn.init.normal_(self.item_emds.weight, std=0.1)

        self.sig = nn.Sigmoid()

    def _convert_sp_mat_to_sp_tensor(self, user_click_dict):
        adj = np.zeros([self.num_users, self.num_items])
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.num_items - 1:
                    adj[i][j] = 1
        print(adj)
        X = sp.csr_matrix(adj, dtype=np.float32)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        UIGraph = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)

        adj = np.zeros([self.num_items, self.num_users])
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.num_items - 1:
                    adj[j][i] = 1
        X = sp.csr_matrix(adj, dtype=np.float32)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        IUGraph = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return UIGraph, IUGraph

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

    def __dropout(self, graph, keep_prob):
        if self.A_split:
            graph = []
            for g in graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(graph, keep_prob)
        return graph

    def svd(self, UIGraph):
        u, s, v = torch.svd(UIGraph.to_dense())
        s_topk = torch.topk(s, 5)
        u_topk = torch.index_select(u, 1, s_topk[1].squeeze())
        v_topk = torch.index_select(v, 1, s_topk[1].squeeze())
        UIGraph_svd = torch.matmul(u_topk , v_topk.T).to(self.device)
        IUGraph_svd = torch.matmul(v_topk , u_topk.T).to(self.device)
        return UIGraph_svd, IUGraph_svd


    def compute(self, users, items):
        user_emds = self.user_emds.weight
        item_emds = self.item_emds.weight

        # if self.args.dropout:
        #     if self.training:
        #         UIGraph_dropped = self.__dropout(self.UIGraph, self.keep_prob)
        #         IUGraph_dropped = self.__dropout(self.IUGraph, self.keep_prob)
        #     else:
        #         UIGraph_dropped = self.UIGraph
        #         IUGraph_dropped = self.IUGraph
        # else:
        #     UIGraph_dropped = self.UIGraph
        #     IUGraph_dropped = self.IUGraph

        Gi_list = []
        Gu_list = []
        Zi_list = []
        Zu_list = []
        for layer in range(self.n_layers):
            Zu_emds = torch.sparse.mm(self.UIGraph, item_emds)
            Zi_emds = torch.sparse.mm(self.IUGraph, user_emds)
            Zi_list.append(Zi_emds)
            Zu_list.append(Zu_emds)
            
            Gu_emds = torch.sparse.mm(self.UIGraph_svd, item_emds)
            Gi_emds = torch.sparse.mm(self.IUGraph_svd, user_emds)
            Gi_list.append(Gi_emds)
            Gu_list.append(Gu_emds)

        Zi = torch.stack(Zi_list, dim=1)
        Zi = torch.mean(Zi, dim=1)
        Zu = torch.stack(Zu_list, dim=1)
        Zu = torch.mean(Zu, dim=1)

        Gi = torch.stack(Gi_list, dim=1)
        Gi = torch.mean(Gi, dim=1)
        Gu = torch.stack(Gu_list, dim=1)
        Gu = torch.mean(Gu, dim=1)

        loss_cl = self.create_cl_loss(users, items, Gi_list, Gu_list, Zu_list, Zi_list)
        return Zu, Zi, loss_cl

    def create_cl_loss(self, users, items, Gi_list, Gu_list, Zu_list, Zi_list):
        items = torch.flatten(items, 0 ,1)
        loss = torch.FloatTensor(0)
        loss_list = []
        for layer in range(self.n_layers):
            gnn_u = nn.functional.normalize(Gu_list[layer][users],p=2,dim=1)
            hyper_u = nn.functional.normalize(Zu_list[layer][users],p=2,dim=1)
            pos_score = torch.exp((gnn_u*hyper_u).sum(1))
            neg_score = torch.exp((gnn_u@hyper_u.T)).sum(1)
            loss_u = torch.mean(((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))))
            loss_list.append(loss_u)
            gnn_i = nn.functional.normalize(Gi_list[layer][items],p=2,dim=1)
            hyper_i = nn.functional.normalize(Zi_list[layer][items],p=2,dim=1)
            pos_score = torch.exp((gnn_i * hyper_i).sum(1))
            neg_score = torch.exp((gnn_i @ hyper_i.T)).sum(1)
            loss_i = torch.mean(((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))))
            loss_list.append(loss_i)
        loss = torch.stack(loss_list, dim = 0).to(self.device)
        loss = torch.mean(loss).to(self.device)
        return loss



    def create_bpr_loss(self, loss_cl, users, items, labels):
        batch_size = users.shape[0]
        scores = (items * users.unsqueeze(1)).sum(dim=-1)
        # scores = torch.sigmoid(scores)
        rec_loss = F.cross_entropy(F.softmax(scores, dim = -1), torch.argmax(labels.to(self.device), dim=1))
        regularizer = (torch.norm(users) ** 2 + torch.norm(items) ** 2) / 2
        emb_loss = self.args.l2 * regularizer / batch_size
        return rec_loss + emb_loss + loss_cl, scores, rec_loss, emb_loss

    
    def forward(self, users, items, label):
        all_users, all_items, loss_cl = self.compute(users, items)
        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        # inner_pro = torch.mul(users_embs, items_embs)
        # gamma = torch.sum(inner_pro, dim=1)
        return self.create_bpr_loss(loss_cl, users_embs, items_embs, label)

    def test(self, users, items):
        all_users, all_items,  _ = self.compute(users, items)
        users_embs = all_users[users.long()]
        items_embs = all_items[items.long()]
        scores = (items_embs * users_embs.unsqueeze(1)).sum(dim=-1)
        # inner_pro = torch.mul(users_embs, items_embs)
        # gamma = torch.sum(inner_pro, dim=1)
        return scores

