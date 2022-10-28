import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import random
class Net(torch.nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.device = device
        self.actor_l1 = nn.Linear(100 * 3, 100)
        self.actor_l2 = nn.Linear(100, 100)
        self.actor_l3 = nn.Linear(100, 1)
        self.critic_l1 = nn.Linear(100 * 3, 100)
        self.critic_l2 = nn.Linear(100, 100)
        self.critic_l3 = nn.Linear(100, 1)
        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-2)

    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1],
                                                 state_input.shape[2])
            else:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2],
                                                 state_input.shape[3])
        # Actor
        actor_x = torch.tanh(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = torch.tanh(self.actor_l2(actor_x))
        act_probs = self.sigmoid(self.actor_l3(actor_out))
        # Critic
        critic_x = torch.tanh(self.critic_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = torch.tanh(self.critic_l2(critic_x))
        q_actions = self.sigmoid(self.critic_l3(critic_out))
        return act_probs, q_actions

class MRNNRL_simple_v5(torch.nn.Module):

    def __init__(self, args, news_entity_dict, entity_news_dict, user_click_dict ,
                 news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict,
                 news_title_embedding, entity_embedding, relation_embedding,
                 entity_adj, relation_adj, neibor_embedding, neibor_num, device):

        super(MRNNRL_simple_v5, self).__init__()
        self.args = args
        self.device = device
        self.entity_num = entity_embedding.shape[0]
        self.relation_num = relation_embedding.shape[0]
        # dict
        self.news_entity_dict = news_entity_dict
        self.entity_news_dict = entity_news_dict
        self.category_news_dict = category_news_dict
        self.subcategory_news_dict = subcategory_news_dict
        self.user_click_dict = user_click_dict
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index

        # KG
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.entity_embedding_origin = entity_embedding
        self.relation_embedding_origin = relation_embedding

        # no_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding).to(self.device)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding).to(self.device)
        self.type_embedding = nn.Embedding(5, self.args.embedding_size)
        self.news_embedding = torch.FloatTensor(np.array(news_title_embedding))
        self.news_embedding = nn.Embedding.from_pretrained(self.news_embedding).to(self.device)

        # embedding
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size).to(self.device)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size).to(self.device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size).to(self.device)
        self.node_embedding = self._reconstruct_node_embedding()

        # 重建异构矩阵
        self.news_node_dict, self.news_node_type_dict = self._reconstruct_news_node()
        self.re_entity_adj = self._reconstruct_entity_adj()
        self.re_relation_adj = self._reconstruct_relation_adj()

        # 激活函数
        self.MAX_DEPTH = 3
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

        # 强化学习网络
        self.policy_net = Net(self.device)
        # self.target_net = Net(self.device)

    def _reconstruct_news_node(self):
        re_news_category = self.news_category_dict + self.entity_num
        re_news_subcategory = self.news_subcategory_dict + self.entity_num + self.args.category_num
        news_node_dict = []
        news_node_type_dict = []
        for k, v in self.news_entity_dict.items():
            news_node_dict.append([])
            news_node_type_dict.append([])
            news_node_dict[-1].append(re_news_category[k])
            news_node_type_dict[-1].append(int(3)) # 3 表示主题
            news_node_dict[-1].append(re_news_subcategory[k])
            news_node_type_dict[-1].append(int(4)) # 4 表示主题
            news_node_dict[-1].extend(v[:10])
            news_node_type_dict[-1].extend([int(2)] * 10) # 2表示实体
            # random.shuffle(news_node_dict[-1])
        self.news_node_dict = torch.Tensor(news_node_dict).to(self.device)
        self.news_node_type_dict = torch.Tensor(news_node_type_dict).to(self.device)
        return self.news_node_dict, self.news_node_type_dict

    def _reconstruct_node_embedding(self):
        self.node_embedding = torch.cat([self.entity_embedding.weight,
                                         self.category_embedding.weight,
                                         self.subcategory_embedding.weight], dim=0).to(self.device)
        return self.node_embedding

    def _reconstruct_entity_adj(self):
        shape_adj = len(self.entity_adj[0])
        re_entity_adj = []
        for i in range(self.node_embedding.shape[0]):
            if i in self.entity_adj.keys():
                re_entity_adj.append(self.entity_adj[i])
            else:
                padding = [i] * shape_adj
                re_entity_adj.append(padding)
        self.re_entity_adj = torch.LongTensor(re_entity_adj).to(self.device)
        return self.re_entity_adj

    def _reconstruct_relation_adj(self):
        shape_adj = len(self.relation_adj[0])
        padding = [self.relation_num - 1] * shape_adj
        relation_adj = []
        for i in range(self.node_embedding.shape[0]):
            if i in self.relation_adj.keys():
                relation_adj.append(self.relation_adj[i])
            else:
                relation_adj.append(padding)
        self.relation_adj = torch.LongTensor(relation_adj).to(self.device)
        return self.relation_adj

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        trans_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_subgraph_list(self, subgraph_nodes, subgraph_type, batch_size):
        subgraph_list = []
        for i in range(batch_size):
            subgraph_list.append([])
        for i in range(len(subgraph_nodes)):
            for j in range(len(subgraph_nodes[i])):
                temp = []
                for m in range(len(subgraph_nodes[i][j].data.cpu().numpy())):
                    node = str(subgraph_nodes[i][j][m].data.cpu()) + str(subgraph_type[i][j][m].data.cpu())
                    temp.append(node)
                subgraph_list[j].extend(temp)
        return subgraph_list

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.matmul(z1, torch.transpose(z2, -2, -1))

    def get_sim_reward_batch(self, graph_embedding, origin_embedding, node_embedding):
        if len(node_embedding.shape) > len(graph_embedding.shape):
            graph_embedding = torch.unsqueeze(graph_embedding, 1)
            graph_embedding = graph_embedding.expand(graph_embedding.shape[0],
                                                     node_embedding.shape[1],
                                                     graph_embedding.shape[2])

        if len(node_embedding.shape) > len(origin_embedding.shape):
            origin_embedding = torch.unsqueeze(origin_embedding, 1)
            origin_embedding = origin_embedding.expand(origin_embedding.shape[0],
                                                       node_embedding.shape[1],
                                                       origin_embedding.shape[2])

        if graph_embedding.shape[1] != node_embedding.shape[1]:
            graph_embedding = graph_embedding.unsqueeze(-2)
            graph_embedding = graph_embedding.expand(graph_embedding.shape[0],
                                                     graph_embedding.shape[1],
                                                     int(node_embedding.shape[1] / graph_embedding.shape[1]),
                                                     graph_embedding.shape[3])
            graph_embedding = torch.flatten(graph_embedding, 1, 2)

        cos_rewards = self.cos(origin_embedding, graph_embedding + node_embedding)
        sim_rewards = self.sim(origin_embedding, graph_embedding + node_embedding).sum(-1)
        dot_rewards = F.softmax(torch.sum(origin_embedding * (graph_embedding + node_embedding), dim = -1), dim = -1)
        graph_embedding = (graph_embedding + node_embedding).to(self.device)
        return dot_rewards, graph_embedding

    # # 代改（暂时不添加）
    # def get_hit_rewards_batch(self, user_index, news_id_batch, node_type_input_batch, node_id_input_batch = None):
    #     hit_rewards = []
    #     hit_rewards_weak = []
    #     neibor = None
    #     if node_id_input_batch != None:
    #         for i in range(len(news_id_batch)):
    #             hit_rewards.append([])
    #             hit_rewards_weak.append([])
    #             for j in range(len(node_id_input_batch[i])):
    #                 if int(node_type_input_batch[i][j].data.cpu().numpy()) == 2:
    #                     if int(node_id_input_batch[i][j].data.cpu().numpy()) in self.entity_news_dict \
    #                             and news_id_batch[i] in self.user_click_dict:
    #                         neibor = set(self.entity_news_dict[int(node_id_input_batch[i][j].data.cpu().numpy())]).discard(news_id_batch[i])
    #                         news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]
    #
    #                 if int(node_type_input_batch[i][j].data.cpu().numpy()) == 3:
    #                     if int(node_id_input_batch[i][j].data.cpu().numpy()) in self.category_news_dict \
    #                             and news_id_batch[i] in self.user_click_dict:
    #                         neibor = set(self.category_news_dict[int(node_id_input_batch[i][j].data.cpu().numpy())]).discard(news_id_batch[i])
    #                         news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]
    #
    #                 if int(node_type_input_batch[i][j].data.cpu().numpy()) == 3:
    #                     if int(node_id_input_batch[i][j].data.cpu().numpy()) in self.subcategory_news_dict \
    #                             and news_id_batch[i] in self.user_click_dict:
    #                         neibor = set(self.subcategory_news_dict[int(node_id_input_batch[i][j].data.cpu().numpy())]).discard(news_id_batch[i])
    #                         news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]
    #                     if neibor != None:
    #                         if len(neibor & news_hit_neibor) > 0:
    #                             print("hit")
    #                             hit_rewards[-1].append(1.0)
    #                         else:
    #                             hit_rewards[-1].append(0.0)
    #                     else:
    #                         hit_rewards[-1].append(0.0)
    #                 else:
    #                     hit_rewards[-1].append(0.0)
    #     else:
    #         for i in range(len(news_id_batch)):
    #             hit_rewards.append([])
    #             hit_rewards_weak.append([])
    #             for news_id in news_id_batch[i]:
    #                 news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]
    #                 if news_id in news_hit_neibor:
    #                     hit_rewards[-1].append(1.0)
    #                 else:
    #                     hit_rewards[-1].append(0.0)
    #     return torch.FloatTensor(hit_rewards).to(self.device)

    def get_outer_constrast_reward(self, graph, depth, nodes_embedding, mode):
        if depth == 1:
            return None
        if depth > 1:
            history_graph = graph[depth - 2]
            history_node_embedding = self.get_nodes_embedding(history_graph, depth - 1, mode)
            nodes_embedding = nodes_embedding.reshape(self.args.batch_size * self.args.sample_size,
                                                      history_node_embedding.shape[1],
                                                      -1, self.args.embedding_size)
            history_node_embedding = history_node_embedding.unsqueeze(-2).repeat(1, 1, nodes_embedding.shape[2], 1)
            exter_const_reward = torch.flatten(F.normalize(self.sim(nodes_embedding, history_node_embedding).sum(-1)), -2, -1)
            return -exter_const_reward

    def get_inter_constrast_reward(self, graph, depth, nodes_embedding, mode):
        inter_const_reward = F.normalize(self.sim(nodes_embedding, nodes_embedding).sum(-1))
        return inter_const_reward

    def get_news_reward(self, graph, graph_embedding, user_embedding, action_input, depth):
        nodes_embedding = self.get_nodes_embedding(action_input, depth, mode = 'news')
        sim_reward, graph_embedding = self.get_sim_reward_batch(graph_embedding, user_embedding, nodes_embedding)
        inter_constrast_reward = self.get_inter_constrast_reward(graph, depth, nodes_embedding, mode ='news')
        outer_constrast_reward = self.get_outer_constrast_reward(graph, depth, nodes_embedding, mode='news')
        # hit_reward = self.get_hit_rewards_batch(user_index, news_id, type_input, action_input)
        if outer_constrast_reward != None:
            reward = 1 * inter_constrast_reward + 1 * outer_constrast_reward + sim_reward
        else:
            reward = inter_constrast_reward + sim_reward
        return reward, graph_embedding

    def get_user_reward(self, graph, graph_embedding, news_embedding, action_input, depth):
        nodes_embedding = self.get_nodes_embedding(action_input, depth, mode = 'user')
        sim_reward, graph_embedding = self.get_sim_reward_batch(graph_embedding, news_embedding, nodes_embedding)
        inter_constrast_reward = self.get_inter_constrast_reward(graph, depth, nodes_embedding, mode='user')
        outer_constrast_reward = self.get_outer_constrast_reward(graph, depth, nodes_embedding, mode='user')
        # hit_reward = self.get_hit_rewards_batch(user_index, newsid, type_input, action_input)
        if outer_constrast_reward != None:
            reward =  1 * inter_constrast_reward + 1 * outer_constrast_reward + sim_reward
        else:
            reward =  inter_constrast_reward + sim_reward 
        return reward, graph_embedding

    def get_nodes_embedding(self, node_input, depth, mode = None):
        if mode == 'news':
            node_embedding = self.node_embedding[node_input]
        if mode == 'user':
            if depth == 1:
                node_embedding = self.trans_news_embedding(node_input)
            else:
                node_embedding = self.node_embedding[node_input]
        return node_embedding

    # 代改（已改）
    def get_action(self, step_node, user_clicked_news_index, hop, mode):
        step_node = step_node.type(torch.long)
        if hop == 0:
            if mode == 'News':
                next_action_id = self.news_node_dict[step_node]
                next_action_r_id = torch.zeros([next_action_id.shape[0],
                                                next_action_id.shape[1]]).to(self.device) # mention = 0
                next_action_t_id = torch.full(next_action_r_id.shape, 2).to(self.device)

            elif mode == 'User':
                next_action_id = user_clicked_news_index[: , :10]
                next_action_r_id = torch.ones([next_action_id.shape[0],
                                               next_action_id.shape[1]]).to(self.device) # click = 1
                next_action_t_id = torch.ones(next_action_r_id.shape).to(self.device)
        elif hop == 1:
            if mode == 'News':
                next_action_id = self.re_entity_adj[step_node]
                next_action_r_id = self.re_relation_adj[step_node]
                next_action_t_id = torch.full(next_action_r_id.shape, 2).to(self.device)
            elif mode == 'User':
                next_action_id = self.news_node_dict[step_node]
                next_action_r_id = torch.zeros([next_action_id.shape[0],
                                                next_action_id.shape[1],
                                                next_action_id.shape[2]]).to(self.device) # click = 1
                next_action_t_id = self.news_node_type_dict[step_node]
        else:
            next_action_id = self.re_entity_adj[step_node]
            next_action_r_id = self.re_relation_adj[step_node]
            next_action_t_id = torch.full(next_action_r_id.shape, 2).to(self.device)

        return next_action_id.to(torch.long), next_action_r_id.to(torch.long), next_action_t_id.to(torch.long)

    def step_update(self, act_probs_step, q_values_step, step_reward, recommend_reward, reasoning_reward, alpha1=0.5, alpha2 = 0.5):
        recommend_reward = torch.unsqueeze(recommend_reward, dim=1).expand(recommend_reward.shape[0], 5)
        reasoning_reward = torch.unsqueeze(reasoning_reward, dim=1).expand(reasoning_reward.shape[0], 5)
        #recommend_reward = torch.unsqueeze(recommend_reward, dim=0).expand(250)
        #reasoning_reward = torch.unsqueeze(reasoning_reward, dim=0).expand(250)
        recommend_reward = recommend_reward.reshape(-1, 1)
        reasoning_reward = reasoning_reward.reshape(-1, 1)
        #curr_reward = step_reward
        curr_reward = step_reward + ( (1 - alpha2) * (alpha1 * recommend_reward + (1-alpha1) * reasoning_reward)) * 0.5
        #curr_reward = step_reward + ( (1 - alpha2) * ( alpha1* recommend_reward + (1-alpha1) * reasoning_reward))  
        advantage = curr_reward - q_values_step
        actor_loss = - act_probs_step * advantage.detach()
        critic_loss = advantage.pow(2)
        return critic_loss, actor_loss

    def get_subgraph_nodes(self, weights, q_values, action_id_input, relation_id_input, type_id_input, topk):
        if len(weights.shape) <= 3:
            weights = torch.unsqueeze(weights, 1) # bz, 1, entity_num, 1
            q_values = torch.unsqueeze(q_values, 1) # bz, 1, entity_num, 1
            action_id_input = torch.unsqueeze(action_id_input, 1) # bz, 1, entity_num
            relation_id_input = torch.unsqueeze(relation_id_input, 1) # bz, 1, entity_num
            type_id_input = torch.unsqueeze(type_id_input, 1)  # bz, 1, entity_num

        # 按计算概率选择动作
        weights = weights.squeeze(-1) # bz, 1, entity_num
        q_values = q_values.squeeze(-1) # bz, 1, entity_num
        m = Categorical(weights) # bz, 1, entity_num
        acts_idx = m.sample(sample_shape=torch.Size([topk])) # d(k-hop), bz, 1
        acts_idx = acts_idx.permute(1, 2, 0) # bz, 1, d(k-hop)
        shape0 = acts_idx.shape[0] # bz
        shape1 = acts_idx.shape[1] # 1

        # reshape
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2]) # bz , d(k-hop), bz
        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2]) # bz * 1 , entity_num
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])  # bz * 1 , entity_num
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])  # bz * 1 , entity_num
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])  # bz * 1 , entity_num
        type_id_input = type_id_input.reshape(type_id_input.shape[0] * type_id_input.shape[1], type_id_input.shape[2])  # bz * 1 , entity_num

        # 选择动作
        action_id_selected = action_id_input.gather(1, acts_idx) #  d(k-hop), bz
        relation_id_selected = relation_id_input.gather(1, acts_idx)#  d(k-hop), bz
        type_id_selected = type_id_input.gather(1, acts_idx)  # d(k-hop), bz
        # 输出
        weights = weights.gather(1, acts_idx) #  d(k-hop), bz
        q_values = q_values.gather(1, acts_idx) #  d(k-hop), bz
        weights = weights.reshape(shape0, shape1 * weights.shape[1]) # 1, d(k-hop) * bz
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1]) # 1, d(k-hop) * bz
        action_id_selected = action_id_selected.reshape(shape0, shape1 * action_id_selected.shape[1]) #  bz, d(k-hop) * 1
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 * relation_id_selected.shape[1]) #  bz, d(k-hop) * 1
        type_id_selected = type_id_selected.reshape(shape0, shape1 * type_id_selected.shape[1]) #  bz, d(k-hop) * 1

        return weights.to(self.device), q_values.to(self.device), action_id_selected.to(self.device), relation_id_selected.to(self.device),type_id_selected.to(self.device)

    def get_state_input(self, news_embedding, depth, history_entity, history_relation, history_type, mode = None):
        if depth == 0:
            state_embedding = torch.cat([news_embedding, torch.FloatTensor(np.zeros((news_embedding.shape[0], 100))).to(self.device)], dim=-1)
        else:
            if mode == 'news':
                history_action_embedding = self.node_embedding[history_entity]
            if mode == 'user':
                if depth == 1:
                    history_action_embedding = self.trans_news_embedding(history_entity)
                else:
                    history_action_embedding = self.node_embedding[history_entity]
            history_relation_embedding = self.relation_embedding(history_relation)  # bz,  d(k-hop-1), dim
            history_type_embedding = self.type_embedding(history_type)
            state_embedding_new = history_relation_embedding + history_action_embedding + history_type_embedding #  bz, d(k-hop-1), dim
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            state_embedding = torch.cat([news_embedding, state_embedding_new], dim=-1) #  bz,  d(k-hop-1), dim
        return state_embedding

    def get_action_embedding(self, action_id, relation_id, type_id, mode = None, hop_num = None):
        action_id = action_id.type(torch.long)
        relation_id = relation_id.type(torch.long)
        type_id = type_id.type(torch.long)
        if mode == 'news':
            action_embedding = self.node_embedding[action_id]
        if mode == 'user':
            if hop_num == 0:
                action_embedding = self.trans_news_embedding(action_id)
            else:
                action_embedding = self.node_embedding[action_id]
        relation_embedding = self.relation_embedding(relation_id)
        type_embedding = self.type_embedding(type_id)
        return action_embedding + relation_embedding + type_embedding


    def forward(self, user_index, user_clicked_news_index, candidate_newindex):
        self.node_embedding = self._reconstruct_node_embedding()
        candidate_newindex = torch.flatten(candidate_newindex, 0, 1).to(self.device)
        user_index = user_index.unsqueeze(1)
        user_index = user_index.expand(user_index.shape[0], 5)
        user_index = torch.flatten(user_index, 0, 1).to(self.device)

        user_clicked_news_index = user_clicked_news_index.unsqueeze(1)
        user_clicked_news_index = user_clicked_news_index.expand(user_clicked_news_index.shape[0], 5, user_clicked_news_index.shape[2])
        user_clicked_news_index = torch.flatten(user_clicked_news_index, 0, 1).to(self.device)

        news_embedding = self.trans_news_embedding(candidate_newindex)
        user_embedding = self.user_embedding(user_index)

        depth = 0
        news_graph = []
        news_graph_relation = []
        news_graph_type = []
        news_act_probs_steps = []
        news_step_rewards = []
        news_q_values_steps = []

        action_id, relation_id, type_id = self.get_action(candidate_newindex, user_clicked_news_index, depth,  mode = "News") # bz * 5, news_entity_num
        action_embedding = self.get_action_embedding(action_id, relation_id, type_id, mode="news", hop_num=depth)
        state_input = self.get_state_input(news_embedding, depth, news_graph, news_graph_relation,  news_graph_type, mode = "news") # bz, 2 * news_dim
        news_graph_embedding = news_embedding
        while (depth < self.MAX_DEPTH):
            topk = self.args.depth[depth]
            depth += 1
            act_probs, q_values = self.policy_net(state_input, action_embedding) # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            act_probs, q_values, step_news_graph_nodes, step_news_graph_relations, step_news_graph_types = self.get_subgraph_nodes(act_probs, q_values, action_id, relation_id, type_id, topk)
            action_id, relation_id, type_id = self.get_action(step_news_graph_nodes, user_clicked_news_index, depth, mode = "News")
            news_act_probs_steps.append(act_probs)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_q_values_steps.append(q_values)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_graph.append(step_news_graph_nodes)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_graph_relation.append(step_news_graph_relations)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_graph_type.append(step_news_graph_types)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]

            step_reward, news_graph_embedding = self.get_news_reward(news_graph, news_graph_embedding, user_embedding, step_news_graph_nodes, depth)  # bz, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            state_input = self.get_state_input(news_embedding, depth, step_news_graph_nodes, step_news_graph_relations, step_news_graph_types, mode = "news") # bz*5, dim # bz*5, dim # bz*5, dim
            action_embedding = self.get_action_embedding(action_id, relation_id, type_id, mode = "news", hop_num=depth)

            news_step_rewards.append(step_reward)  #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]


        depth = 0
        user_graph = []
        user_graph_relation = []
        user_graph_type = []
        user_act_probs_steps = []
        user_step_rewards = []
        user_q_values_steps = []

        action_id, relation_id, type_id = self.get_action(user_index, user_clicked_news_index, depth, mode="User")  # bz * 5, news_entity_num
        action_embedding = self.get_action_embedding(action_id, relation_id, type_id,  mode = "user", hop_num=depth)
        state_input = self.get_state_input(user_embedding, depth, user_graph, user_graph_relation, user_graph_type, mode = 'user')   # bz, 2 * news_dim
        user_graph_embedding = user_embedding
        while (depth < self.MAX_DEPTH):
            topk = self.args.depth[depth]
            depth += 1
            act_probs, q_values = self.policy_net(state_input, action_embedding) # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            act_probs, q_values, step_user_graph_nodes, step_user_graph_relations, step_user_graph_types = self.get_subgraph_nodes(act_probs, q_values, action_id, relation_id, type_id, topk)

            user_act_probs_steps.append(act_probs) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_q_values_steps.append(q_values) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_graph.append(step_user_graph_nodes) # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_graph_relation.append(step_user_graph_relations)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_graph_type.append(step_user_graph_types)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]

            action_id, relation_id, type_id = self.get_action(step_user_graph_nodes, user_clicked_news_index, depth, mode="User")
            step_reward, user_graph_embedding = self.get_user_reward(user_graph, user_graph_embedding, news_embedding, step_user_graph_nodes, depth)  # bz, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            state_input = self.get_state_input(user_embedding, depth, step_user_graph_nodes, step_user_graph_relations, step_user_graph_types, mode = 'user')  # bz*5, dim # bz*5, dim # bz*5, dim
            action_embedding = self.get_action_embedding(action_id, relation_id, type_id, mode = 'user', hop_num=depth)

            user_step_rewards.append(step_reward)
        return news_act_probs_steps, news_q_values_steps, news_step_rewards, news_graph, news_graph_relation, news_graph_type, \
               user_act_probs_steps, user_q_values_steps, user_step_rewards, user_graph, user_graph_relation, user_graph_type


