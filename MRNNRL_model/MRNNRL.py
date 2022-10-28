import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

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
        self.softmax = torch.nn.Softmax(dim=0)

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
        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.sigmoid(self.actor_l3(actor_out))
        # Critic
        critic_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        q_actions = self.sigmoid(self.critic_l3(critic_out))
        return act_probs, q_actions

class MRNNRL(torch.nn.Module):

    def __init__(self, args, news_entity_dict, entity_news_dict, user_click_dict ,
                 news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict,
                 news_title_embedding, entity_embedding, relation_embedding,
                 entity_adj, relation_adj, neibor_embedding, neibor_num, device):

        super(MRNNRL, self).__init__()
        self.args = args
        self.device = device
        self.news_entity_dict = news_entity_dict
        self.entity_news_dict = entity_news_dict
        self.category_news_dict = category_news_dict
        self.subcategory_news_dict = subcategory_news_dict
        self.user_click_dict = user_click_dict
        self.news_category_index = news_category_index
        self.news_subcategory_index = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        # new_embedding = news_title_embedding.tolist()
        # new_embedding.append(np.array([0 for i in range(768)]))
        self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(np.array(news_title_embedding)))
        # self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(news_title_embedding))
        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size)
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_size)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_size)
        self.type_embedding = nn.Embedding(5, self.args.embedding_size)

        self.neibor_num = neibor_num

        self.MAX_DEPTH = 3
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)

        self.policy_net = Net(self.device)
        self.target_net = Net(self.device)

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index)
        trans_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_subgraph_list(self, subgraph_nodes, batch_size):
        subgraph_list = []
        for i in range(batch_size):
            subgraph_list.append([])
        for i in range(len(subgraph_nodes)):
            for j in range(len(subgraph_nodes[i])):
                subgraph_list[j].extend(list(map(lambda x:str(x), subgraph_nodes[i][j].data.cpu().numpy())))
        return subgraph_list

    def get_sim_reward_batch(self, user_embedding, news_embedding_batch, neibor_news_embedding_avg_batch):
        if len(neibor_news_embedding_avg_batch.shape) > len(news_embedding_batch.shape):
            news_embedding_batch = torch.unsqueeze(news_embedding_batch, 1)
            news_embedding_batch = news_embedding_batch.expand(news_embedding_batch.shape[0],
                                                               neibor_news_embedding_avg_batch.shape[1],
                                                               news_embedding_batch.shape[2])
            user_embedding = torch.unsqueeze(user_embedding, 1)
            user_embedding = news_embedding_batch.expand(user_embedding.shape[0],
                                                         neibor_news_embedding_avg_batch.shape[1],
                                                         user_embedding.shape[2])
        cos_rewards = self.cos(user_embedding, news_embedding_batch + neibor_news_embedding_avg_batch)
        return cos_rewards

    # 代改（一改）
    def get_hit_rewards_batch(self, user_index, news_id_batch, node_type_input_batch, node_id_input_batch = None):
        hit_rewards = []
        hit_rewards_weak = []
        neibor = None
        if node_id_input_batch != None:
            for i in range(len(news_id_batch)):
                hit_rewards.append([])
                hit_rewards_weak.append([])
                for j in range(len(node_id_input_batch[i])):
                    if int(node_type_input_batch[i][j].data.cpu().numpy()) == 2:
                        if int(node_id_input_batch[i][j].data.cpu().numpy()) in self.entity_news_dict \
                                and news_id_batch[i] in self.user_click_dict:
                            neibor = set(self.entity_news_dict[int(node_id_input_batch[i][j].data.cpu().numpy())]).discard(news_id_batch[i])
                            news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]

                    if int(node_type_input_batch[i][j].data.cpu().numpy()) == 3:
                        if int(node_id_input_batch[i][j].data.cpu().numpy()) in self.category_news_dict \
                                and news_id_batch[i] in self.user_click_dict:
                            neibor = set(self.category_news_dict[int(node_id_input_batch[i][j].data.cpu().numpy())]).discard(news_id_batch[i])
                            news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]

                    if int(node_type_input_batch[i][j].data.cpu().numpy()) == 3:
                        if int(node_id_input_batch[i][j].data.cpu().numpy()) in self.subcategory_news_dict \
                                and news_id_batch[i] in self.user_click_dict:
                            neibor = set(self.subcategory_news_dict[int(node_id_input_batch[i][j].data.cpu().numpy())]).discard(news_id_batch[i])
                            news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]
                        if neibor != None:
                            if len(neibor & news_hit_neibor) > 0:
                                hit_rewards[-1].append(1.0)
                            else:
                                hit_rewards[-1].append(0.0)
                        else:
                            hit_rewards[-1].append(0.0)
                    else:
                        hit_rewards[-1].append(0.0)
        else:
            for i in range(len(news_id_batch)):
                hit_rewards.append([])
                hit_rewards_weak.append([])
                for news_id in news_id_batch[i]:
                    news_hit_neibor = self.user_click_dict[user_index[i].cpu().item()]
                    if news_id in news_hit_neibor:
                        hit_rewards[-1].append(1.0)
                    else:
                        hit_rewards[-1].append(0.0)
        return torch.FloatTensor(hit_rewards).to(self.device)

    def get_reward(self, user_index, news_id, user_embedding, news_embedding, type_input, action_input):
        neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, action_input)
        sim_reward = self.get_sim_reward_batch(user_embedding, news_embedding, neibor_news_embedding_avg)
        hit_reward = self.get_hit_rewards_batch(user_index, news_id, type_input, action_input)
        reward = 0.5 * hit_reward + (1-0.5) * sim_reward
        return reward

    # 代改
    def get_neiborhood_news_embedding_batch(self, news_embedding, entityids):
        neibor_news_embedding_avg = self.neibor_embedding(entityids)
        neibor_news_embedding_avg = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(neibor_news_embedding_avg))))
        neibor_num = []
        for i in range(len(entityids)):
            neibor_num.append(torch.index_select(self.neibor_num, 0, entityids[i].to(self.device)))
        neibor_num = torch.stack(neibor_num)
        if len(neibor_news_embedding_avg.shape) > len(news_embedding.shape):
            news_embedding = torch.unsqueeze(news_embedding, 1)
            neibor_num = torch.unsqueeze(neibor_num, 2)
            news_embedding = news_embedding.expand(news_embedding.shape[0], neibor_news_embedding_avg.shape[1], news_embedding.shape[2])
            neibor_num = neibor_num.expand(neibor_num.shape[0], neibor_num.shape[1], news_embedding.shape[2])
        neibor_news_embedding_avg = torch.div((neibor_news_embedding_avg - news_embedding), neibor_num).to(self.device)
        return neibor_news_embedding_avg

    # 代改
    def get_neiborhood_clicked_news_embedding_batch(self, news_embedding, newsid, user_clicked_news_index):
        neiborhood_clicked_news_embedding = []
        temp = None
        for i in range(user_clicked_news_index.shape[0]):
            for j in range(user_clicked_news_index.shape[1]):
                if j == 0:
                    temp = self.news_embedding(user_clicked_news_index[i, j]).squeeze()
                else:
                    temp = temp + self.news_embedding(user_clicked_news_index[i,j]).squeeze()
            neiborhood_clicked_news_embedding.append(temp)
        neibor_clicked_avg = torch.stack(neiborhood_clicked_news_embedding)

        neibor_clicked_news_embedding_avg = []
        for i in range(newsid.shape[0]):
            neibor_clicked_news_embedding_avg.append([])
            for j in range(newsid.shape[1]):
                neibor_clicked_news_embedding_avg[-1].append(np.array(neibor_clicked_avg[i]))
        neibor_clicked_embedding_avg = torch.FloatTensor(np.array(neibor_clicked_news_embedding_avg)).to(self.device)
        neibor_clicked_embedding_avg = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(neibor_clicked_embedding_avg))))

        if len(neibor_clicked_embedding_avg.shape) > len(news_embedding.shape):
            news_embedding = torch.unsqueeze(news_embedding, 1)
            news_embedding = news_embedding.expand(news_embedding.shape[0],neibor_clicked_embedding_avg.shape[1],news_embedding.shape[2])
        return torch.div((neibor_clicked_embedding_avg - news_embedding), 10).to(self.device)

    def get_user_reward(self, depth, user_clicked_news_index, user_index, newsid, user_embedding, news_embedding, type_input, action_input):
        if depth == 1:
            neibor_news_embedding_avg = self.get_neiborhood_clicked_news_embedding_batch(news_embedding, action_input, user_clicked_news_index)
            sim_reward = self.get_sim_reward_batch(user_embedding, news_embedding, neibor_news_embedding_avg)
            hit_reward = self.get_hit_rewards_batch(user_index, newsid, type_input)
            reward = 0.5 * hit_reward + (1-0.5) * sim_reward
        else:
            neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, action_input)
            sim_reward = self.get_sim_reward_batch(user_embedding, news_embedding, neibor_news_embedding_avg)
            hit_reward = self.get_hit_rewards_batch(user_index, newsid, type_input, action_input)
            reward = 0.5 * hit_reward + (1-0.5) * sim_reward
        return reward

    # 代改（已改）
    def get_action(self, step_node, step_node_type, user_clicked_news_index, mode):
        next_action_id = []
        next_action_r_id = []
        next_action_t_id = []
        if len(step_node_type) == 0:
            if mode == 'News':
                for i in range(len(step_node)):
                    next_action_id.append([])
                    next_action_r_id.append([])
                    next_action_t_id.append([])

                    next_action_id[-1].extend(self.news_entity_dict[int(step_node[i])])
                    next_action_r_id[-1].extend([0 for k in range(10)])
                    next_action_t_id[-1].extend([2 for k in range(10)]) # 2代表实体

                    next_action_id[-1].append(self.news_category_index[step_node[i]])
                    next_action_r_id[-1].append(0)
                    next_action_t_id[-1].append(3)  # 3代表主题

                    next_action_id[-1].append(self.news_subcategory_index[step_node[i]])
                    next_action_r_id[-1].append(0)
                    next_action_t_id[-1].append(4)  # 4代表副主题

            elif mode == 'User':
                for i in range(len(step_node)):
                    next_action_id.append(list(user_clicked_news_index[i].cpu().numpy()))
                    next_action_r_id.append([0 for k in range(10)])
                    next_action_t_id.append([1 for k in range(10)]) # 1代表新闻
            next_action_id = torch.LongTensor(next_action_id).to(self.device)  # bz, d(k-hop) , 20
            next_action_r_id = torch.LongTensor(next_action_r_id).to(self.device)  # bz, d(k-hop) , 20
            next_action_t_id = torch.LongTensor(next_action_t_id).to(self.device)  # bz, d(k-hop) , 20
            return next_action_id, next_action_r_id, next_action_t_id

        for i in range(len(step_node)):
            next_action_id.append([])
            next_action_r_id.append([])
            next_action_t_id.append([])
            # print('------')
            for j in range(len(step_node[i])):
                if step_node_type[i][j].item() == 0:
                    next_action_id[-1].append(user_clicked_news_index[i].cpu().numpy())
                    next_action_r_id[-1].extend([1 for k in range(10)])
                    next_action_t_id[-1].extend([1 for k in range(10)]) # 1代表新闻

                if step_node_type[i][j].item() == 1:
                    temp = self.news_entity_dict[step_node[i][j].item()] + \
                           [self.news_category_index[step_node[i][j]].item()] +\
                           [self.news_subcategory_index[step_node[i][j]].item()]

                    temp1 = [0 for k in range(len(self.news_entity_dict[step_node[i][j].item()]))] + \
                            [0] + \
                            [0]

                    temp2 = [2 for k in range(len(self.news_entity_dict[step_node[i][j].item()] ))] + \
                            [3] + \
                            [4]

                    next_action_id[-1].append(temp)
                    next_action_r_id[-1].append(temp1)
                    next_action_t_id[-1].append(temp2) # 2代表实体类别

                if step_node_type[i][j].item() == 2:
                    if int(step_node[i][j].data.cpu().numpy()) in self.entity_adj:
                        next_action_id[-1].append(self.entity_adj[int(step_node[i][j].data.cpu().numpy())])
                        next_action_r_id[-1].append(self.relation_adj[int(step_node[i][j].data.cpu().numpy())])
                        next_action_t_id[-1].append([2 for k in range(10)])
                    else:
                        next_action_id[-1].append([0 for k in range(10)])
                        next_action_r_id[-1].append([0 for k in range(10)])
                        next_action_t_id[-1].append([2 for k in range(10)])

                if step_node_type[i][j].item() == 3:
                    next_action_id[-1].append([0 for k in range(10)])
                    next_action_r_id[-1].append([0 for k in range(10)])
                    next_action_t_id[-1].append([3 for k in range(10)])  # 3代表主题类别
                    
                if step_node_type[i][j].item() == 4:
                    next_action_id[-1].append([0 for k in range(10)])
                    next_action_r_id[-1].append([0 for k in range(10)])
                    next_action_t_id[-1].append([4 for k in range(10)])  # 4代表副主题类别

        next_action_id = torch.LongTensor(next_action_id).to(self.device) # bz, d(k-hop) , 20
        next_action_r_id = torch.LongTensor(next_action_r_id).to(self.device)# bz, d(k-hop) , 20
        next_action_t_id = torch.LongTensor(next_action_t_id).to(self.device)# bz, d(k-hop) , 20

        return next_action_id, next_action_r_id, next_action_t_id

    def step_update(self, act_probs_step, q_values_step, step_reward, recommend_reward, reasoning_reward, alpha1=0.9, alpha2 = 0.1):
        recommend_reward = torch.unsqueeze(recommend_reward, dim=1).repeat(1,5,1)
        reasoning_reward = torch.unsqueeze(reasoning_reward, dim=1).repeat(1,5,1)
        recommend_reward = recommend_reward.view(-1, 1)
        reasoning_reward = reasoning_reward.view(-1, 1)
        curr_reward = alpha2 * step_reward + (1-alpha2) * (alpha1 * recommend_reward + (1-alpha1) * reasoning_reward)
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

        weights = weights.squeeze(-1) # bz, 1, entity_num
        q_values = q_values.squeeze(-1) # bz, 1, entity_num
        m = Categorical(weights) # bz, 1, entity_num
        acts_idx = m.sample(sample_shape=torch.Size([topk])) # d(k-hop), bz, 1
        acts_idx = acts_idx.permute(1, 2, 0) # bz, 1, d(k-hop)
        shape0 = acts_idx.shape[0] # bz
        shape1 = acts_idx.shape[1] # 1

        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2]) # bz , d(k-hop), bz
        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2]) # bz * 1 , entity_num
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])  # bz * 1 , entity_num
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])  # bz * 1 , entity_num
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])  # bz * 1 , entity_num
        type_id_input = type_id_input.reshape(type_id_input.shape[0] * type_id_input.shape[1],type_id_input.shape[2])  # bz * 1 , entity_num

        action_id_selected = action_id_input.gather(1, acts_idx) #  d(k-hop), bz
        relation_id_selected = relation_id_input.gather(1, acts_idx)#  d(k-hop), bz
        type_id_selected = type_id_input.gather(1, acts_idx)  # d(k-hop), bz

        weights = weights.gather(1, acts_idx) #  d(k-hop), bz
        q_values = q_values.gather(1, acts_idx) #  d(k-hop), bz
        weights = weights.reshape(shape0, shape1 * weights.shape[1]) # 1, d(k-hop) * bz
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1]) # 1, d(k-hop) * bz
        action_id_selected = action_id_selected.reshape(shape0, shape1 * action_id_selected.shape[1]) #  bz, d(k-hop) * 1
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 * relation_id_selected.shape[1]) #  bz, d(k-hop) * 1
        type_id_selected = type_id_selected.reshape(shape0, shape1 * type_id_selected.shape[1])  # bz, d(k-hop) * 1

        return weights.to(self.device), q_values.to(self.device), action_id_selected.to(self.device), relation_id_selected.to(self.device), type_id_selected.to(self.device)

    def get_state_input(self, news_embedding, depth, history_entity, history_relation, history_type):
        history_action_embedding_list = []
        history_type_embedding_list = []
        if depth == 0:
            state_embedding = torch.cat(
                [news_embedding, torch.FloatTensor(np.zeros((news_embedding.shape[0], 100))).to(self.device)], dim=-1)
        else:
            for i in range(history_entity.shape[0]):
                history_action_embedding_list.append([])
                history_type_embedding_list.append([])
                for j in range(history_entity.shape[1]):
                    type = history_type[i][j]
                    entity = history_entity[i][j]
                    if type.item() == 1:
                        history_action_embedding_list[-1].append(self.trans_news_embedding(entity))
                        history_type_embedding_list[-1].append(self.type_embedding(type))
                    elif type.item() == 2:
                        history_action_embedding_list[-1].append(self.entity_embedding(entity))
                        history_type_embedding_list[-1].append(self.type_embedding(type))
                    elif type.item() == 3:
                        history_action_embedding_list[-1].append(self.category_embedding(entity))
                        history_type_embedding_list[-1].append(self.type_embedding(type))
                    elif type.item() == 4:
                        history_action_embedding_list[-1].append(self.subcategory_embedding(entity))
                        history_type_embedding_list[-1].append(self.type_embedding(type))
                    else:
                        print("实体类型错误")
                history_action_embedding_list[-1] = torch.stack(history_action_embedding_list[-1])
            history_action_embedding = torch.stack(history_action_embedding_list)
            history_relation_embedding = self.relation_embedding(history_relation) #  bz,  d(k-hop-1), dim
            state_embedding_new = history_relation_embedding + history_action_embedding #  bz,  d(k-hop-1), dim
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            state_embedding = torch.cat([news_embedding, state_embedding_new], dim=-1) #  bz,  d(k-hop-1), dim
        return state_embedding.to(self.device)

    def predict_subgraph(self, newsid, news_feature):
        self.news_entity_dict[newsid] = news_feature[0]
        self.news_title_embedding = news_feature[1]
        prediction = self.forward([newsid], [newsid]).cpu().data.numpy()
        anchor_nodes = prediction[6]
        anchor_relation = prediction[8]
        return anchor_nodes, anchor_relation

    def get_action_embedding(self, action_id, relation_id, type_id):

        action_embedding_list = []
        type_embedding_list = []

        if len(action_id.shape) == 2:
            for i in range(action_id.shape[0]):
                action_embedding_list.append([])
                type_embedding_list.append([])
                for j in range(action_id.shape[1]):
                    type = type_id[i, j]
                    action = action_id[i, j]
                    if type.item() == 1:
                        action_embedding_list[-1].append(self.trans_news_embedding(action))
                        type_embedding_list[-1].append(self.type_embedding(torch.tensor(1).to(self.device)))
                    elif type.item() == 2:
                        action_embedding_list[-1].append(self.entity_embedding(action))
                        type_embedding_list[-1].append(self.type_embedding(torch.tensor(2).to(self.device)))
                    elif type.item() == 3:
                        action_embedding_list[-1].append(self.category_embedding(action))
                        type_embedding_list[-1].append(self.type_embedding(torch.tensor(3).to(self.device)))
                    elif type.item() == 4:
                        action_embedding_list[-1].append(self.subcategory_embedding(action))
                        type_embedding_list[-1].append(self.type_embedding(torch.tensor(4).to(self.device)))
                    else:
                        print("类型不匹配")
                action_embedding_list[-1] = torch.stack(action_embedding_list[-1])
                type_embedding_list[-1] = torch.stack(type_embedding_list[-1])

        # len(action_id.shape) == 3
        if len(action_id.shape) == 3:
            for i in range(action_id.shape[0]):
                action_embedding_list.append([])
                type_embedding_list.append([])
                for j in range(action_id.shape[1]):
                    action_embedding_list[-1].append([])
                    type_embedding_list[-1].append([])
                    for m in range(action_id.shape[2]):
                        type = type_id[i, j, m]
                        action = action_id[i, j, m]
                        if type.item() == 1:
                            action_embedding_list[-1][-1].append(self.trans_news_embedding(action))
                            type_embedding_list[-1][-1].append(self.type_embedding(torch.tensor(1).to(self.device)))
                        elif type.item() == 2:
                            action_embedding_list[-1][-1].append(self.entity_embedding(action))
                            type_embedding_list[-1][-1].append(self.type_embedding(torch.tensor(2).to(self.device)))
                        elif type.item() == 3:
                            action_embedding_list[-1][-1].append(self.category_embedding(action))
                            type_embedding_list[-1][-1].append(self.type_embedding(torch.tensor(3).to(self.device)))
                        elif type.item() == 4:
                            action_embedding_list[-1][-1].append(self.subcategory_embedding(action))
                            type_embedding_list[-1][-1].append(self.type_embedding(torch.tensor(4).to(self.device)))
                        else:
                            print("类型不匹配")
                    action_embedding_list[-1][-1] = torch.stack(action_embedding_list[-1][-1])
                    type_embedding_list[-1][-1] = torch.stack(type_embedding_list[-1][-1])
                action_embedding_list[-1] = torch.stack(action_embedding_list[-1])
                type_embedding_list[-1] = torch.stack(type_embedding_list[-1])
        action_embedding = torch.stack(action_embedding_list)
        relation_embedding = self.relation_embedding(relation_id)
        type_embedding = torch.stack(type_embedding_list)
        return action_embedding + relation_embedding + type_embedding


    def forward(self, user_index, user_clicked_news_index, candidate_newindex):
        candidate_newindex = torch.flatten(candidate_newindex, 0, 1).to(self.device)

        pt = user_index.unsqueeze(-1)
        for i in range(5):
            if i == 0:
                new_pt = pt
            else:
                new_pt = torch.cat([new_pt, pt], dim=-1)
        user_index = torch.flatten(new_pt, 0, 1).to(self.device)

        pt = user_clicked_news_index.unsqueeze(1)
        for i in range(5):
            if i == 0:
                new_pt = pt
            else:
                new_pt = torch.cat([new_pt, pt], dim=1)
        user_clicked_news_index = torch.flatten(new_pt, 0, 1).to(self.device)

        user_embedding = self.user_embedding(user_index)

        depth = 0
        news_graph = []
        news_graph_relation = []
        news_graph_type = []
        news_act_probs_steps = []
        news_step_rewards = []
        news_q_values_steps = []
        news_embedding = self.trans_news_embedding(candidate_newindex)

        action_id, relation_id, type_id = self.get_action(candidate_newindex, news_graph_type, user_clicked_news_index, mode = "News") # bz * 5, news_entity_num
        action_embedding = self.get_action_embedding(action_id, relation_id, type_id)
        state_input = self.get_state_input(news_embedding, depth,  news_graph, news_graph_relation, news_graph_type) # bz, 2 * news_dim

        while (depth < self.MAX_DEPTH):

            topk = self.args.depth[depth]
            depth += 1
            act_probs, q_values = self.policy_net(state_input, action_embedding) # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            anchor_act_probs, anchor_q_values, step_news_graph_nodes, step_news_graph_relations, step_news_graph_type = self.get_subgraph_nodes(act_probs, q_values, action_id, relation_id, type_id, topk)
            action_id, relation_id, type_id = self.get_action(step_news_graph_nodes, step_news_graph_type, user_clicked_news_index, mode = "News")
            step_reward = self.get_reward(user_index, candidate_newindex, user_embedding, news_embedding, step_news_graph_type, step_news_graph_nodes)  # bz, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            state_input = self.get_state_input(news_embedding, depth, step_news_graph_nodes, step_news_graph_relations, step_news_graph_type) # bz*5, dim # bz*5, dim # bz*5, dim
            action_embedding = self.get_action_embedding(action_id, relation_id, type_id)

            news_act_probs_steps.append(anchor_act_probs) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_q_values_steps.append(anchor_q_values) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_graph.append(step_news_graph_nodes) # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_graph_relation.append(step_news_graph_relations)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            news_graph_type.append(step_news_graph_type)
            news_step_rewards.append(step_reward)  #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]


        depth = 0
        news_id = None
        user_graph = []
        user_graph_relation = []
        user_graph_type = []
        user_act_probs_steps = []
        user_step_rewards = []
        user_q_values_steps = []

        action_id, relation_id, type_id = self.get_action(user_index, user_graph_type, user_clicked_news_index, mode="User")  # bz * 5, news_entity_num
        action_embedding = self.get_action_embedding(action_id, relation_id, type_id)
        state_input = self.get_state_input(user_embedding, depth, user_graph, user_graph_relation, user_graph_relation)   # bz, 2 * news_dim

        while (depth < self.MAX_DEPTH):
            topk = self.args.depth[depth]
            depth += 1
            act_probs, q_values = self.policy_net(state_input, action_embedding) # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            anchor_act_probs, anchor_q_values, step_user_graph_nodes, step_user_graph_relations, step_user_graph_type = self.get_subgraph_nodes(act_probs, q_values, action_id, relation_id, type_id, topk)

            if depth == 1:
                news_id = step_user_graph_nodes
                clicked_news_embedding= self.news_embedding(news_id)
                clicked_news_embedding = torch.div(torch.sum(clicked_news_embedding, dim = 1 ), topk)
                clicked_news_embedding = torch.tanh(self.news_compress_2(self.elu(self.news_compress_1(clicked_news_embedding))))

            action_id, relation_id, type_id = self.get_action(step_user_graph_nodes, step_user_graph_type, user_clicked_news_index, mode="User")
            step_reward = self.get_user_reward(depth, user_clicked_news_index, user_index, news_id, user_embedding, clicked_news_embedding,
                                               step_user_graph_type, step_user_graph_nodes)  # bz, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            state_input = self.get_state_input(user_embedding, depth, step_user_graph_nodes, step_user_graph_relations, step_user_graph_type)  # bz*5, dim # bz*5, dim # bz*5, dim
            action_embedding = self.get_action_embedding(action_id, relation_id, type_id)

            user_act_probs_steps.append(anchor_act_probs) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_q_values_steps.append(anchor_q_values) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_graph.append(step_user_graph_nodes) # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_graph_relation.append(step_user_graph_relations)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            user_graph_type.append(step_user_graph_type)
            user_step_rewards.append(step_reward)

        return news_act_probs_steps, news_q_values_steps, news_step_rewards, news_graph, news_graph_relation, news_graph_type, \
               user_act_probs_steps, user_q_values_steps, user_step_rewards, user_graph, user_graph_relation, user_graph_type


