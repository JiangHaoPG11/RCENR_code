import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


class RippleNet(nn.Module):
    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, ripple_set):
        super(RippleNet, self).__init__()
        self._parse_args(args)
        news_title_embedding = news_title_embedding.tolist()
        news_title_embedding.append(np.random.normal(-0.1, 0.1, 768))
        self.news_title_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(news_title_embedding))
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.ripple_set = ripple_set
        self.transform_matrix = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.news_to_entity = nn.Linear(self.title_dim, self.embedding_dim, bias=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def _parse_args(self, args):
        self.embedding_dim = args.embedding_size
        self.ripplenet_n_hop = args.ripplenet_n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.ripplenet_n_memory = args.ripplenet_n_memory
        self.update_mode = args.update_mode
        self.using_all_hops = args.using_all_hops
        self.title_dim = args.title_size

    def _get_feed_dict(self, user_index):
        memories_h, memories_r, memories_t = [], [], []
        for i in range(self.ripplenet_n_hop + 1):
            # for user in user_index:
            memories_h_one = self.ripple_set[user_index, i, 0, :]
            memories_r_one = self.ripple_set[user_index, i, 1, :]
            memories_t_one = self.ripple_set[user_index, i, 2, :]
            memories_h.append(memories_h_one)
            memories_r.append(memories_r_one)
            memories_t.append(memories_t_one)
        return memories_h, memories_r, memories_t

    def forward(self, user_index, candidate_newsindex, labels):
        user_index = user_index.to(self.device)
        candidate_newsindex = candidate_newsindex.to(self.device)
        labels = labels.to(self.device)
        memories_h, memories_r, memories_t = self._get_feed_dict(user_index)
        # print(memories_h)
        # print(memories_r)
        # print(memories_t)
        # [batch size, dim]
        news_embeddings = self.news_title_embedding(candidate_newsindex)
        news_embeddings = torch.tanh(self.news_to_entity(news_embeddings))
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.ripplenet_n_hop + 1):
            memories_h[i] = memories_h[i].to(self.device)
            memories_r[i] = memories_r[i].to(self.device)
            memories_t[i] = memories_t[i].to(self.device)
            #print(memories_h[i])
            #print(memories_r[i])
            #print(memories_t[i])
            if i == 0:
                #print(memories_h[i])
                # [batch size, ripplenet_n_memory, dim]
                h_emb_list.append(torch.tanh(self.news_to_entity(self.news_title_embedding(memories_h[i]))))
                # [batch size, ripplenet_n_memory, dim]
                r_emb_list.append(self.relation_embedding(memories_r[i]))
                # [batch size, ripplenet_n_memory, dim]
                t_emb_list.append(self.entity_embedding(memories_t[i]))
            elif i == self.ripplenet_n_hop + 1:
                # [batch size, ripplenet_n_memory, dim]
                h_emb_list.append(self.entity_embedding(memories_t[i]))
                # [batch size, ripplenet_n_memory, dim]
                r_emb_list.append(self.relation_embedding(memories_r[i]))
                # [batch size, ripplenet_n_memory, dim]
                t_emb_list.append(torch.tanh(self.news_to_entity(self.news_title_embedding(memories_h[i]))))
            else:
                # [batch size, ripplenet_n_memory, dim]
                h_emb_list.append(self.entity_embedding(memories_h[i]))
                # [batch size, ripplenet_n_memory, dim]
                r_emb_list.append(self.relation_embedding(memories_r[i]))
                # [batch size, ripplenet_n_memory, dim]
                t_emb_list.append(self.entity_embedding(memories_t[i]))
        o_list, news_embeddings = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, news_embeddings)
        #print('---')
        #print(o_list)
        scores = self.predict(news_embeddings, o_list)
        #print(scores)
        return_dict = self._compute_loss(scores, labels, h_emb_list, t_emb_list, r_emb_list)
        return_dict["scores"] = scores
        return return_dict

    def test(self, user_index, candidate_newsindex):
        user_index = user_index.to(self.device)
        candidate_newsindex = candidate_newsindex.to(self.device)

        memories_h, memories_r, memories_t = self._get_feed_dict(user_index)
        # [batch size, dim]
        news_embeddings = self.news_title_embedding(candidate_newsindex)
        news_embeddings = torch.tanh(self.news_to_entity(news_embeddings))
        h_emb_list = []
        r_emb_list = []
        t_emb_list = []
        for i in range(self.ripplenet_n_hop + 1):
            memories_h[i] = memories_h[i].to(self.device)
            memories_r[i] = memories_r[i].to(self.device)
            memories_t[i] = memories_t[i].to(self.device)
            if i == 0:
                # [batch size, ripplenet_n_memory, dim]
                h_emb_list.append(torch.tanh(self.news_to_entity(self.news_title_embedding(memories_h[i]))))
                # [batch size, ripplenet_n_memory, dim]
                r_emb_list.append(self.relation_embedding(memories_r[i]))
                # [batch size, ripplenet_n_memory, dim]
                t_emb_list.append(self.entity_embedding(memories_t[i]))
            elif i == self.ripplenet_n_hop + 1:
                # [batch size, ripplenet_n_memory, dim]
                h_emb_list.append(self.entity_embedding(memories_t[i]))
                # [batch size, ripplenet_n_memory, dim]
                r_emb_list.append(self.relation_embedding(memories_r[i]))
                # [batch size, ripplenet_n_memory, dim]
                t_emb_list.append(torch.tanh(self.news_to_entity(self.news_title_embedding(memories_h[i]))))
            else:
                # [batch size, ripplenet_n_memory, dim]
                h_emb_list.append(self.entity_embedding(memories_h[i]))
                # [batch size, ripplenet_n_memory, dim]
                r_emb_list.append(self.relation_embedding(memories_r[i]))
                # [batch size, ripplenet_n_memory, dim]
                t_emb_list.append(self.entity_embedding(memories_t[i]))
        o_list, news_embeddings = self._key_addressing(h_emb_list, r_emb_list, t_emb_list, news_embeddings)
        scores = self.predict(news_embeddings, o_list)
        return scores

    def _compute_loss(self, scores, labels, h_emb_list, t_emb_list, r_emb_list):
        base_loss = F.cross_entropy(scores, torch.argmax(labels, dim=1))
        kge_loss = 0
        for hop in range(self.ripplenet_n_hop):
            # [batch size, ripplenet_n_memory, dim, dim]
            hRt = torch.squeeze(torch.mul(torch.mul(h_emb_list[hop], r_emb_list[hop]), t_emb_list[hop]))
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss
        l2_loss = 0
        for hop in range(self.ripplenet_n_hop):
            l2_loss += (h_emb_list[hop] * h_emb_list[hop]).sum()
            l2_loss += (t_emb_list[hop] * t_emb_list[hop]).sum()
            l2_loss += (r_emb_list[hop] * r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss
        loss = base_loss +  kge_loss +  l2_loss
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)

    def _key_addressing(self, h_emb_list, r_emb_list, t_emb_list, news_embeddings):
        o_list = []
        for hop in range(self.ripplenet_n_hop + 1):
            # [batch_size, 5, ripplenet_n_memory, dim]
            Rh = (torch.mul(r_emb_list[hop], h_emb_list[hop])).unsqueeze(1)
            Rh = Rh.expand(Rh.shape[0],5,Rh.shape[2], Rh.shape[3])
            # [batch_size, 5, dim, 1]
            v = torch.unsqueeze(news_embeddings, dim=-1)
            # [batch_size, 5, ripplenet_n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))
            # [batch_size, 5,ripplenet_n_memory]
            probs_normalized = F.softmax(probs, dim=1)
            # [batch_size, 5, ripplenet_n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=-1)
            # [batch_size, 5, dim]
            #o = t_emb_list[hop].unsqueeze(1)
            #o = o.expand(o.shape[0], 5, o.shape[2], o.shape[3])
            #o = (o * probs_expanded).sum(dim=2)
            o = (t_emb_list[hop].unsqueeze(1).repeat(1,5,1,1) * probs_expanded).sum(dim=2)
            news_embeddings = self._update_news_embedding(news_embeddings, o)
            o_list.append(o)
        return o_list, news_embeddings

    def _update_news_embedding(self, news_embeddings, o):
        if self.update_mode == "replace":
            news_embeddings = o
        elif self.update_mode == "plus":
            news_embeddings = news_embeddings + o
        elif self.update_mode == "replace_transform":
            news_embeddings = self.transform_matrix(o)
        elif self.update_mode == "plus_transform":
            news_embeddings = self.transform_matrix(news_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.update_mode)
        return news_embeddings

    def predict(self, news_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.ripplenet_n_hop):
                y += o_list[i]
        # [batch_size, 5]
        #print('======')
        #print(y)
        #print('-----')
        #print(news_embeddings)
        scores = (news_embeddings * y).sum(dim=-1)
        #scores = F.softmax((news_embeddings * y).sum(dim=-1), dim = -1)
        #scores = torch.sigmoid(scores)
        #print(scores)
        return scores

