import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import csv
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#################################   Attentnion  ###############################
def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    return torch.softmax(m , -1)

def attention(Q, K, V, dim_attn):
    #Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K)/ dim_attn**0.5 #(batch_size, dim_attn, seq_length)
    return  torch.matmul(a,  V) #(batch_size, seq_length, seq_length)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_attn)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    def forward(self, x, dim_attn, kv = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x), dim_attn)
        #Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv), dim_attn)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        self.heads = nn.ModuleList(self.heads)
        self.fc = nn.Linear(n_heads * dim_attn, dim_attn, bias = False)
    def forward(self, x, dim_attn,  kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, dim_attn, kv = kv))
        a = torch.stack(a, dim = -1) #combine heads
        a = torch.flatten(a, start_dim = 2)  #flatten all head outputs
        x = self.fc(a)
        x = torch.tanh(x)
        return x

class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
        #self.fc2 = nn.Linear(5, dim_val)
    def forward(self, x):
        x = self.fc1(x)
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    def forward(self, x):
        x = self.fc1(x)
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    def forward(self, x):
        x = self.fc1(x)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadSelfAttention_2(nn.Module):
    def __init__(self, input_dim, d_model, num_attention_heads):
        super(MultiHeadSelfAttention_2, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads
        self.W_Q = nn.Linear(input_dim, d_model)
        self.W_K = nn.Linear(input_dim, d_model)
        self.W_V = nn.Linear(input_dim, d_model)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads, self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).expand(batch_size, maxlen).to(device) < length.view(-1, 1).to(device)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen, maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1,1)
        else:
            attn_mask = None
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context

class Additive_Attention(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(Additive_Attention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim, bias=True)
        self.attention_query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
    def forward(self, candidate_vector):
        # bz, title_word_size, title_word_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # bz, title_word_size, 1
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector), dim=-2)
        target = torch.matmul(candidate_weights.unsqueeze(dim=-2), candidate_vector).squeeze(dim=-2)
        # bz, title_word_dim
        return target

class Additive_Attention_printweight(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(Additive_Attention_printweight, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim, bias=True)
        self.attention_query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, candidate_vector, type_encoder, write = None):

        # bz, title_word_size, title_word_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # bz, title_word_size, 1
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector), dim=1)
        if type_encoder == 'newencoder' and write == 'yes':
            print(candidate_weights.shape)
            with open("weight.csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                # 写入多行用writerows
                writer.writerows(candidate_weights.detach().numpy())
            print(candidate_weights)
        else:
            print('no')
        target = torch.bmm(candidate_weights.unsqueeze(dim=1), candidate_vector).squeeze(dim=1)
        # bz, title_word_dim
        return target

class Additive_Attention_PENR(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(Additive_Attention_PENR, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim, bias=True)
        self.attention_query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
    def forward(self, candidate_vector):
        # bz, title_word_size, title_word_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # bz, title_word_size, 1
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query_vector), dim=-2).unsqueeze(-1)
        target = candidate_weights * candidate_vector
        # bz, title_word_size, title_word_dim
        return target

class QueryAttention(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(QueryAttention, self).__init__()
        self.fc1 = nn.Linear(query_vector_dim, candidate_vector_dim, bias=True)
    def forward(self, wrt_vector, candidate_vector):
        temp = torch.tanh(self.fc1(wrt_vector))
        # batch_size, candidate_size,
        candidate_weights = F.softmax(torch.bmm(temp, candidate_vector.transpose(1, 2)).squeeze(dim=1), dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1), candidate_vector).squeeze(dim=1)
        return target

class QueryAttention_2(torch.nn.Module):
    def __init__(self, query_vector_dim, candidate_vector_dim):
        super(QueryAttention_2, self).__init__()
        self.fc1 = nn.Linear(query_vector_dim, candidate_vector_dim, bias=True)
        self.fc2 = nn.Linear(candidate_vector_dim, candidate_vector_dim, bias=True)
    def forward(self, wrt_vector, candidate_vector):
        temp = self.fc1(wrt_vector)
        candidate_vector = torch.tanh(self.fc2(candidate_vector))
        # batch_size, candidate_size,
        candidate_weights = F.softmax(torch.bmm(temp, candidate_vector.transpose(1, 2)).squeeze(dim=1), dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1), candidate_vector).squeeze(dim=1)
        return target


class SimilarityAttention(torch.nn.Module):
    def __init__(self):
        super(SimilarityAttention, self).__init__()

    def forward(self, wrt_vector, candidate_vector):
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.bmm(
            candidate_vector, wrt_vector.unsqueeze(1).transpose(2,1)).squeeze(dim=2), dim=1)
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target


class NormalAttention(torch.nn.Module):
    def __init__(self, vector_dim):
        super(NormalAttention, self).__init__()
        self.fc1 = nn.Linear(vector_dim, 200)
        self.fc2 = nn.Linear(200, 1,bias=True)
    def forward(self, candidate_vector):
        # batch_size, candidate_size, candiate_dim
        temp = torch.tanh(self.fc2(self.fc1(candidate_vector)))
        candidate_weights = F.softmax(temp, dim=1)
        # batch_size, candidate_size, 1
        target = torch.bmm(candidate_weights.transpose(2,1), candidate_vector).squeeze()
        # batch_size, candidate_vector_dim
        return target


class Self_Attention(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(Self_Attention, self).__init__()

        self.trans_layer = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh()
        )
        self.gate_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_num, embedding_size]
        :param seq_lens: shape [batch_size, seq_num]
        :return: shape [batch_size, embedding_size]
        """
        gates = self.gate_layer(self.trans_layer(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output

#################################   Attentnion  ###############################
#################################   module_Net  ###############################
class gcn(nn.Module):
    def __init__(self, entity_size, embedding_dim, attention_dim):
        super(gcn, self).__init__()
        self.entity_size = entity_size
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.batch_size = 50
        self.fc1 = nn.Linear(self.embedding_dim, self.attention_dim, bias=True)
    def init_graph(self):
        dimension = self.entity_size
        graph = np.ones((dimension, dimension)) - np.eye(dimension)
        return torch.from_numpy(graph.astype(np.float32))
    def forward(self, entity_embedding):
        Adj = self.init_graph().to(device)
        Adj = Adj.unsqueeze(0)
        Adj = Adj.repeat(entity_embedding.shape[0], 1, 1)
        A_H = torch.transpose(torch.bmm(torch.transpose(entity_embedding, 2, 1), Adj), 2, 1)  # (None* M * embedding_size)
        A_H_W = self.fc1(A_H)
        return A_H_W

class KGAT(nn.Module):
    def __init__(self, attention_size, entity_embedding_dim):
        super(KGAT, self).__init__()
        self.attention_size = attention_size
        self.entity_embedding_dim = entity_embedding_dim
        self.fc1 = nn.Linear(3 * self.entity_embedding_dim, self.attention_size, bias=True)
        self.fc2 = nn.Linear(self.attention_size, 1, bias=True)
        # self.ratio = 0.3
    def forward(self, entity_embedding, neigh_entity_embedding, neigh_relation_embedding):
        entity_embedding_expand = entity_embedding.unsqueeze(2)
        entity_embedding_expand = entity_embedding_expand.repeat(1, 1, 5, 1)
        embeddings_concat = torch.cat([entity_embedding_expand, neigh_entity_embedding, neigh_relation_embedding], 3)
        fc1_out = self.fc1(embeddings_concat)
        fc2_out = self.fc2(fc1_out)
        attention_value = F.softmax(fc2_out, dim = 2)
        neighbor_att_embedding = torch.sum(attention_value * neigh_entity_embedding, dim=2)
        kgat_embedding = torch.cat([entity_embedding.squeeze(), neighbor_att_embedding], dim = -1)
        return kgat_embedding

class KCNN(torch.nn.Module):
    def __init__(self, size, title_word_szie, word_embedding_dim, entity_embedding_dim, num_filters, window_sizes, use_context = None):
        super(KCNN, self).__init__()
        self.use_context = use_context
        self.title_word_szie = title_word_szie
        self.word_embedding_dim = word_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.num_filters = num_filters
        self.window_sizes = window_sizes
        self.size = size
        self.transform_matrix = nn.Parameter(torch.zeros(self.size, self.entity_embedding_dim, self.word_embedding_dim))
        self.transform_bias = nn.Parameter(torch.zeros(self.title_word_szie, self.word_embedding_dim))
        self.reset_parameters()
        self.conv_filters = nn.ModuleDict(
            {
            str(x): nn.Conv2d(3 if self.use_context else 2,
                              self.num_filters,
                              (x, self.word_embedding_dim))
            for x in self.window_sizes
            })
        self.norm = nn.LayerNorm(word_embedding_dim)

    def reset_parameters(self):
        self.transform_matrix.data.uniform_(-0.1, 0.1)
        self.transform_bias.data.uniform_(-0.1, 0.1)

    def forward(self, word_embedding, entity_embedding, context_embedding = None):
        transformed_entity_embedding = torch.tanh(torch.add(torch.matmul(entity_embedding, self.transform_matrix), self.transform_bias))
        transformed_entity_embedding = self.norm(transformed_entity_embedding)
        if self.use_context == True:
            transformed_context_embedding = torch.tanh(torch.add(torch.matmul(context_embedding, self.transform_matrix),self.transform_bias))
        # 堆叠
        if self.use_context == True:
            multi_channel_embedding = torch.stack([word_embedding, transformed_entity_embedding, transformed_context_embedding],dim=1)
        else:
            multi_channel_embedding = torch.stack([word_embedding, transformed_entity_embedding], dim=1)
        # 池化&卷积
        pooled_embedding = []
        for x in self.window_sizes:
            convoluted = self.conv_filters[str(x)](multi_channel_embedding).squeeze(dim=3)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            pooled_embedding.append(pooled)
        # batch_size, len(window_sizes) =3 * num_filters 50
        new_rep = torch.cat(pooled_embedding, dim=1)
        return new_rep

class dkn_attention(torch.nn.Module):
    def __init__(self, window_sizes, num_filters, user_clicked_new_num):
        super(dkn_attention, self).__init__()
        self.window_sizes = window_sizes
        self.num_filters = num_filters
        self.user_clicked_new_num = user_clicked_new_num
        self.dnn = nn.Sequential(nn.Linear(len(self.window_sizes) * 2 * self.num_filters, 16),
                                 nn.Linear(16, 1))
    def forward(self, candidate_new_embedding, clicked_news_embedding):
        # num_clicked_news_a_user, len(window_sizes) * num_filters
        candidate_new_embedding_expand = candidate_new_embedding.expand(self.user_clicked_new_num, -1, -1)
        clicked_news_embedding_expand = clicked_news_embedding.unsqueeze(1).repeat(1, 5, 1)
        # num_clicked_news_a_user
        clicked_news_weights = F.softmax(self.dnn(
            torch.cat((clicked_news_embedding_expand, candidate_new_embedding_expand),
                      dim=-1)).squeeze(-1).transpose(0, 1), dim=1)
        #  len(window_sizes) * num_filters
        user_vector = torch.bmm(clicked_news_weights.unsqueeze(1),
                                clicked_news_embedding_expand.transpose(0, 1)).squeeze(1)
        return user_vector

class PCNN(torch.nn.Module):
    def __init__(self, title_word_size, word_embedding_dim, entity_embedding_dim, entity_cate_size):
        super(PCNN, self).__init__()
        self.title_word_size = title_word_size
        self.word_embedding_dim = word_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.num_filters = 8

        self.entity_cate_size = entity_cate_size
        self.entity_conv_filters = nn.ModuleDict(
            {
            str(x): nn.Conv2d(2, 10, kernel_size=(23, self.entity_embedding_dim))
            for x in range(self.num_filters)
            })
        self.word_conv_filters = nn.ModuleDict(
            {
            str(x): nn.Conv2d(1, 10,  kernel_size=(23, self.word_embedding_dim), stride=(2, 2))
            for x in range(self.num_filters)
            })
        self.norm = nn.LayerNorm(word_embedding_dim)
        self.cate_embedding = nn.Embedding(entity_cate_size, embedding_dim = 100)
        self.cate_transfor = nn.Linear(100, self.entity_embedding_dim, bias=True)
    def forward(self, word_embedding, entity_embedding, entity_cate):
        bz = word_embedding.shape[0]
        word_embedding = word_embedding.unsqueeze(1)
        entity_cate_rep = self.cate_embedding(entity_cate.to(torch.int64))
        entity_cate_rep = self.cate_transfor(entity_cate_rep)
        entity_embedding = torch.stack([entity_embedding, entity_cate_rep], dim = 1)
        # 实体池化&卷积
        entity_pooled_embedding = []
        for x in range(self.num_filters):
            convoluted = self.entity_conv_filters[str(x)](entity_embedding)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            entity_pooled_embedding.append(pooled.view(bz, -1))

        # batch_size, len(window_sizes) = 3 * num_filters 50
        entity_rep = torch.cat(entity_pooled_embedding, dim= 1)
        # 单词池化&卷积
        word_pooled_embedding = []
        for x in range(self.num_filters):
            convoluted = self.word_conv_filters[str(x)](word_embedding)
            activated = F.relu(convoluted)
            pooled = activated.max(dim=-1)[0]
            word_pooled_embedding.append(pooled.view(bz, -1))
        # batch_size, len(window_sizes) =3 * num_filters 50
        word_rep = torch.cat(word_pooled_embedding, dim = 1)
        new_rep = torch.cat([word_rep, entity_rep], dim = -1)
        return new_rep

class ANN(torch.nn.Module):
    def __init__(self, user_clicked_new_num, sample_num, title_dim, hidden_size, mode):
        super(ANN, self).__init__()
        self.user_clicked_new_num = user_clicked_new_num
        self.mode = mode
        if self.mode == "interest":
            self.sample_num = sample_num
            self.title_dim = title_dim
            self.fc1 = nn.Linear(self.title_dim , 16, bias = True)
            self.fc2 = nn.Linear(16, 1, bias=True)
        if self.mode == "history":
            self.hidden_size = hidden_size
            self.fc1 = nn.Linear(self.hidden_size, 16, bias = True)
            self.fc2 = nn.Linear(16, 1, bias=True)

    def forward(self, candidate_new_embedding, clicked_news_embedding):
        if self.mode == "interest":
            candidate_new_rep_expand = self.fc1(candidate_new_embedding.expand(self.user_clicked_new_num, -1, -1)).transpose(0, 1)
            clicked_news_rep_expand = self.fc1(clicked_news_embedding.unsqueeze(1).repeat(1, self.sample_num, 1)).transpose(0, 1)
            clicked_news_weights = F.softmax(self.fc2(candidate_new_rep_expand + clicked_news_rep_expand), dim=1)
            user_vector = torch.bmm(clicked_news_weights.transpose(-1, -2), clicked_news_embedding.unsqueeze(1).repeat(1, self.sample_num, 1).transpose(0, 1)).transpose(0, 1)
            return user_vector
        if self.mode == "history":
            candidate_new_rep = self.fc1(candidate_new_embedding)
            clicked_news_rep = self.fc1(candidate_new_embedding)
            weight = F.softmax(self.fc2(candidate_new_rep + clicked_news_rep), dim = 1)
            user_vector = torch.bmm(weight.transpose(-1, -2), candidate_new_embedding)
            return user_vector

class ARNN(torch.nn.Module):
    def __init__(self, user_clicked_num, word_dim, news_dim, sample_num):
        super(ARNN, self).__init__()
        self.hidden_size = 100
        self.news_dim = news_dim
        self.user_clicked_num = user_clicked_num
        self.num_layers = 1
        self.sample_num = sample_num
        self.lstm = nn.LSTM(input_size = self.news_dim, hidden_size= self.hidden_size, num_layers= self.num_layers, batch_first=True)
        self.ARNN = ANN(self.user_clicked_num, 5, 100, self.hidden_size, mode = 'history')
        self.final_fc = nn.Linear(self.news_dim + self.hidden_size, self.news_dim)
    def forward(self, candidate_new_rep, clicked_new_rep):
        clicked_news_lstm, (_,_) = self.lstm(clicked_new_rep.unsqueeze(0))
        user_vector = None
        for i in range(1, self.user_clicked_num):
            clicked_news_one = clicked_news_lstm[:, i,:].unsqueeze(1)
            clicked_news_select = clicked_news_lstm[:, :i,:]
            user_vector_one = self.ARNN(clicked_news_select, clicked_news_one)
            if i == 1:
                user_vector = user_vector_one
            else:
                user_vector = user_vector + user_vector_one
        user_vector = user_vector.repeat(1, self.sample_num, 1)
        user_vector = torch.cat([user_vector, candidate_new_rep], dim = -1)
        user_vector = self.final_fc(user_vector)
        return user_vector

class cnn(torch.nn.Module):
    def __init__(self,  title_word_szie, word_embedding_dim, dropout_pro, query_vector_dim,
                 num_filters, window_sizes):
        super(cnn, self).__init__()
        self.title_word_szie = title_word_szie
        self.word_embedding_dim = word_embedding_dim
        self.dropout_prob= dropout_pro
        self.num_filters = num_filters
        self.window_sizes = window_sizes
        self.conv = nn.Conv2d( in_channels= 1, out_channels=self.num_filters, kernel_size=(self.window_sizes, self.word_embedding_dim),
                               padding=(int((self.window_sizes - 1) / 2), 0))
        # self.additive_attention = Additive_Attention(query_vector_dim, num_filters)
        self.new_attention = NormalAttention(num_filters)

    def forward(self, word_embedding):
        convoluted_word_embedding = self.conv(word_embedding.unsqueeze(dim=1)).squeeze(dim=3)
        convoluted_word_embedding = F.dropout(F.relu(convoluted_word_embedding), p=self.dropout_prob, training=self.training)
        new_rep = self.new_attention(convoluted_word_embedding.transpose(1, 2))
        return new_rep

#################################   module_Net  ###############################
#################################   baseline  ###############################
class graphattention(torch.nn.Module):
    def __init__(self,  F,  F_, attn_heads, attn_heads_reduction, dropout_prob):
        super(graphattention, self).__init__()
        self.F = F  # Number of input features (F in the paper)
        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_prob = dropout_prob  # Internal dropout rate
        self.supports_masking = False
        self.kernel = nn.Linear(self.F, self.F_, bias=True)
        self.attn_kernel_self = nn.Linear(self.F_, 1, bias=True)
        self.attn_kernel_neighs = nn.Linear(self.F_, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, X_input, A_input):
        outputs = []
        for head in range(self.attn_heads):
            features = self.kernel(X_input)  # [B, N, F']
            attn_for_self = self.attn_kernel_self(features)  # [B, N, 1]
            attn_for_neighs = self.attn_kernel_neighs(features)  # [B, N, 1]
            # 获取两两注意力权重
            attention = F.leaky_relu(torch.bmm(attn_for_self, torch.transpose(attn_for_neighs, 2, 1)))  # [B, N, N]
            mask = -10e9 * (1.0 - A_input)
            attention += mask
            # Apply dropout to features and attention coefficients
            dropout_attn = F.dropout(F.softmax(attention, dim=2), p=self.dropout_prob,
                                     training=self.training)  # [B, N, N]
            dropout_feat = F.dropout(features, p=self.dropout_prob, training=self.training)  # [B, N, F']
            # Linear combination with neighbors' features
            node_features = torch.bmm(dropout_attn, dropout_feat)  # [B, N, F']
            # Add output of attention head to final output
            outputs.append(node_features)
            # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = torch.cat(outputs, dim=2)  # [B, N, KF']
        else:
            output = torch.mean(torch.stack(outputs, dim = 1), dim=1)  # [B, N, F']
        output = torch.tanh(output)  # (B, N, F') or # [B, N, KF']
        return output

## KIM
class GraphCoAttNet(torch.nn.Module):
    def __init__(self, entity_dim, output_dim):
        super(GraphCoAttNet, self).__init__()
        self.multiheadatt = MultiHeadSelfAttention_3(entity_dim, output_dim, 20)
        self.entity_co_att = nn.Linear(entity_dim, output_dim)
        self.candidate_co_att = nn.Linear(entity_dim, output_dim)
        self.entity_self_att = nn.Linear(entity_dim, output_dim)
        self.candidate_co_att = nn.Linear(entity_dim, output_dim)
        self.att = nn.Linear(output_dim, 1)

    def forward(self, entity_embedding, candiate_embedding):
        entity_vector = self.multiheadatt(entity_embedding)  #b, 50, max_num, neigh_num, 100
        entity_co_att = self.entity_co_att(entity_vector)    #b, 50, max_num, neigh_num, 100
        Iu = torch.matmul(entity_co_att, torch.transpose(candiate_embedding, -1, -2)) #b,max_num,max_num
        Iu = F.softmax(Iu, dim = 2)  #b, 50, max_num, neigh_num, neigh_num
        entity_self_att = self.entity_self_att(entity_vector) #b, 50, max_num, neigh_num, 100
        candidate_co_att = self.candidate_co_att(candiate_embedding) #b, 50, max_num, neigh_num, 100
        entity_att = torch.tanh(torch.add(entity_self_att, torch.matmul(Iu,candidate_co_att)))#b, 50, max_num, neigh_num, 100
        entity_att = self.att(entity_att) #b, 50, max_num, neigh_num, 1
        entity_att = F.softmax(entity_att, dim = -2) #b, 50, max_num, neigh_num, 1
        entity_rep = torch.matmul(torch.transpose(entity_att,-1,-2), entity_vector).squeeze() #b,max_num, output_dim
        return entity_rep

class GAT(torch.nn.Module):
    def __init__(self, entity_dim, output_dim):
        super(GAT, self).__init__()
        self.multiheadatt = MultiHeadSelfAttention_2(entity_dim, output_dim, 20)
        self.fc1 = nn.Linear(entity_dim, 200)
        self.att = nn.Linear(200, 1)
        self.dropput_prob = 0.2
    def forward(self, entity_embedding):
        # b, max_num, 5, embedding_dim max_entity_num,100
        entity_embedding = F.dropout(entity_embedding, p = self.dropput_prob, training=self.training)  # b, 50, max_num, 5, 100
        att = self.fc1(entity_embedding) # b, 50, max_num, 5, 200
        att = self.att(att)  # b, 50, max_num, 5, 1
        entity_rep = torch.matmul(torch.transpose(att, -1, -2), entity_embedding).squeeze() # b, 50, max_num, 100
        return entity_rep

class MultiHeadSelfAttention_3(nn.Module):
    def __init__(self, input_dim, d_model, num_attention_heads):
        super(MultiHeadSelfAttention_3, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads
        self.W_Q = nn.Linear(input_dim, d_model)
        self.W_K = nn.Linear(input_dim, d_model)
        self.W_V = nn.Linear(input_dim, d_model)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        if len(list(Q.size())) == 5:
            batch_size = Q.size(0)
            q_s = self.W_Q(Q).view(batch_size, Q.size(1), Q.size(2), Q.size(3), self.num_attention_heads, self.d_k).transpose(3, 4)
            k_s = self.W_K(K).view(batch_size, Q.size(1), Q.size(2), Q.size(3), self.num_attention_heads, self.d_k).transpose(3, 4)
            v_s = self.W_V(V).view(batch_size, Q.size(1), Q.size(2), Q.size(3), self.num_attention_heads, self.d_v).transpose(3, 4)
            attn_mask = None
            context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
            context = context.transpose(3, 4).contiguous().view(batch_size, Q.size(1), Q.size(2), Q.size(3), self.num_attention_heads * self.d_v)
        if len(list(Q.size())) == 4:
            batch_size = Q.size(0)
            q_s = self.W_Q(Q).view(batch_size, Q.size(1), Q.size(2), self.num_attention_heads, self.d_k).transpose(2, 3)
            k_s = self.W_K(K).view(batch_size, Q.size(1), Q.size(2), self.num_attention_heads,self.d_k).transpose(2, 3)
            v_s = self.W_V(V).view(batch_size, Q.size(1), Q.size(2), self.num_attention_heads,self.d_v).transpose(2, 3)
            attn_mask = None
            context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
            context = context.transpose(2, 3).contiguous().view(batch_size, Q.size(1), Q.size(2),self.num_attention_heads * self.d_v)
        return context

class get_agg(torch.nn.Module):
    def __init__(self, word_num):
        super(get_agg, self).__init__()
        self.word_num = word_num
    def forward(self, vec_input):
        vecs1 = vec_input[:, :, : self.word_num, :]
        vec2 = vec_input[:, :, self.word_num:, :]
        cross_att = torch.matmul(vecs1,torch.transpose(vec2, -1, -2)).squeeze()
        return cross_att

class get_context_aggergator(torch.nn.Module):
    def __init__(self, input_dim):
        super(get_context_aggergator, self).__init__()
        self.input_dim = input_dim
    def forward(self, vec_input):
        vecs1 = vec_input[:, :, :, :self.input_dim] #(bz,max_num,400)
        att = vec_input[:, :, :, self.input_dim:]  #(bz,max_num,1)
        output = torch.matmul(torch.transpose(att, -1, -2), vecs1).squeeze()
        return output
