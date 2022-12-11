import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import random

class Net(torch.nn.Module):
    def __init__(self, entity_dict, entity_embedding, device):
        super(Net, self).__init__()
        self.device = device
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.entity_dict = entity_dict
        self.actor_l1 = nn.Linear(100 * 5, 100)
        self.actor_l2 = nn.Linear(100, 100)
        self.actor_l3 = nn.Linear(100, 1)
        self.critic_l1 = nn.Linear(100 * 5, 100)
        self.critic_l2 = nn.Linear(100, 100)
        self.critic_l3 = nn.Linear(100, 1)
        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, state_input, action_input):
        if len(state_input.shape) < len(action_input.shape):
            if len(action_input.shape) == 3:
                state_input = torch.unsqueeze(state_input, 1)
                state_input = state_input.expand(state_input.shape[0],
                                                 action_input.shape[1],
                                                 state_input.shape[2])
            else:
                if len(state_input.shape) == 2:
                    state_input = torch.unsqueeze(state_input, 1)
                    state_input = torch.unsqueeze(state_input, 2)
                    state_input = state_input.expand(state_input.shape[0],
                                                     action_input.shape[1],
                                                     action_input.shape[2],
                                                     state_input.shape[3])
                if len(state_input.shape) == 3:
                    state_input = torch.unsqueeze(state_input, 2)
                    state_input = state_input.expand(state_input.shape[0],
                                                     state_input.shape[1],
                                                     action_input.shape[2],
                                                     state_input.shape[3])
        # Actor
        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.sigmoid(self.actor_l3(actor_out))

        # Critic
        critic_x = self.elu(self.critic_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        q_actions = self.sigmoid(self.critic_l3(critic_out))

        return act_probs, q_actions

class ADAC(torch.nn.Module):
    def __init__(self, args, news_entity_dict, entity_news_dict, user_click_dict, news_title_embedding, entity_embedding,
                 relation_embedding, entity_adj, relation_adj, entity_dict,
                 demo_paths_index, demo_type_index, demo_relations_index, device):
        super(ADAC, self).__init__()
        self.args = args
        self._parse_args(args)
        self.device = device

        # 数据
        self.news_entity_dict = news_entity_dict
        self.entity_news_dict = entity_news_dict
        self.user_click_dict = user_click_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.entity_adj, self.relation_adj, self.news_entity_dict = self._reconstruct_adj()
        self.entity_dict = entity_dict


        # 激活函数
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # 元数据处理网路
        self.user_embedding = nn.Embedding(self.user_size, self.embeddign_dim)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.type_embedding = nn.Embedding(3, self.embeddign_dim)
        self.news_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(news_title_embedding))

        self.news_compress_1 = nn.Linear(self.title_dim, self.embeddign_dim)
        self.news_compress_2 = nn.Linear(self.embeddign_dim, self.embeddign_dim)
        self.demo_paths_index = demo_paths_index.to(self.device)
        self.demo_type_index = demo_type_index.to(self.device)
        self.demo_relations_index = demo_relations_index.to(self.device)

        # 强化学习网络
        self.policy_net = Net(self.entity_dict, entity_embedding, self.device)

        # 判别网络
        self.path_discriminator_l1 = nn.Linear(100 * 5, 100)
        self.path_discriminator_l2 = nn.Linear(100, 1)
        self.meta_discriminator_l1 = {}
        for i in range(1, self.max_path_long + 1):
            self.meta_discriminator_l1[i] = nn.Linear(100 * (i), 100).to(self.device)
        self.meta_discriminator_l2 = nn.Linear(100, 1)


    def _reconstruct_adj(self):
        entity_adj_update = {}
        relation_adj_update = {}
        for key, value in self.entity_adj.items():
            if key != 0:
                temp = value.pop(0)
                if type(temp) == int:
                    temp = [temp]
                entity_adj_update[key] = temp
                relation_adj_update[key] = self.relation_adj[key][:len(temp)]

        news_entity_dict_update = {}
        for key, value in self.news_entity_dict.items():
            temp = value.pop(0)
            if type(temp) == int:
                if temp == 0:
                    temp = []
                else:
                    temp = [temp]
            news_entity_dict_update[key] = temp
        return entity_adj_update, relation_adj_update, news_entity_dict_update

    def _parse_args(self, args):
        self.user_size = args.user_size
        self.title_dim = args.title_size
        self.embeddign_dim = args.embedding_size
        self.max_path_long = args.ADAC_path_long
        self.max_path = args.ADAC_max_path
        self.sample_size = args.sample_size
        self.user_clicked_num = args.user_clicked_num
        self.batch_size = args.batch_size

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_embedding(news_index.to(self.device))  # bz * 5, news_entity_num, dim
        trans_news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_state_embedding(self, user_embedding, path_node, path_relation, path_type):
        padding_embedding = torch.tensor(np.zeros([user_embedding.shape[0], 100])).to(torch.float).to(self.device)

        if len(path_node) == 1:
            state_embedding = torch.cat([user_embedding, padding_embedding, padding_embedding, padding_embedding], dim=-1)

        elif len(path_node) == 2:
            curr_node_embedding = []
            for i in range(path_type[-1].shape[0]):
                curr_node_embedding_temp = []
                for j in range(path_type[-1].shape[1]):
                    node_index = path_node[-1][i][j]
                    if path_type[-1][i][j].item() == 0:
                        node_embedding = self.user_embedding(node_index)
                    elif path_type[-1][i][j].item() == 1:
                        node_embedding = self.trans_news_embedding(node_index)
                    else:
                        node_embedding = self.entity_embedding(node_index)
                    curr_node_embedding_temp.append(node_embedding)
                curr_node_embedding.append(torch.stack(curr_node_embedding_temp))
            curr_node_embedding = torch.stack(curr_node_embedding).to(self.device)


            if len(user_embedding.shape) != len(curr_node_embedding.shape):
                user_embedding = user_embedding.unsqueeze(1)
                user_embedding.expand(user_embedding.shape[0], curr_node_embedding.shape[1], user_embedding.shape[2])
                padding_embedding = torch.zeros([user_embedding.shape[0],
                                                 curr_node_embedding.shape[1],
                                                 self.embeddign_dim]).to(torch.float).to(self.device)

            if user_embedding.shape[1] != curr_node_embedding.shape[1]:
                user_embedding = user_embedding.repeat(1, curr_node_embedding.shape[1], 1)

            state_embedding = torch.cat([user_embedding,
                                         curr_node_embedding,
                                         padding_embedding,
                                         padding_embedding], dim= -1)  # bz,  d(k-hop-1), dim
        else:
            curr_node_embedding = []
            for i in range(path_type[-1].shape[0]):
                curr_node_embedding_temp = []
                for j in range(path_type[-1].shape[1]):
                    node_index = path_node[-1][i][j]
                    if path_type[-1][i][j].item() == 0:
                        node_embedding = self.user_embedding(node_index)
                    elif path_type[-1][i][j].item() == 1:
                        node_embedding = self.trans_news_embedding(node_index)
                    else:
                        node_embedding = self.entity_embedding(node_index)
                    curr_node_embedding_temp.append(node_embedding)
                curr_node_embedding.append(torch.stack(curr_node_embedding_temp))
            curr_node_embedding = torch.stack(curr_node_embedding)

            if len(user_embedding.shape) != len(curr_node_embedding.shape):
                user_embedding = user_embedding.unsqueeze(1)
                user_embedding.expand(user_embedding.shape[0],
                                      curr_node_embedding.shape[1],
                                      user_embedding.shape[2])

            if user_embedding.shape[1] != curr_node_embedding.shape[1]:
                user_embedding = user_embedding.repeat(1, curr_node_embedding.shape[1], 1)

            last_node_embedding = []
            for i in range(path_type[-2].shape[0]):
                last_node_embedding_temp = []
                for j in range(path_type[-2].shape[1]):
                    node_index = path_node[-2][i][j]
                    if path_type[-2][i][j].item() == 0:
                        node_embedding = self.user_embedding(node_index)
                    elif path_type[-2][i][j].item() == 1:
                        node_embedding = self.trans_news_embedding(node_index)
                    else:
                        node_embedding = self.entity_embedding(node_index)
                    last_node_embedding_temp.append(node_embedding)
                last_node_embedding.append(torch.stack(last_node_embedding_temp))

            last_node_embedding = torch.stack(last_node_embedding)
            last_relation_embedding = self.relation_embedding(path_relation[-2])  # bz, 1, dim

            if last_node_embedding.shape != curr_node_embedding.shape:
                last_node_embedding = last_node_embedding.unsqueeze(2)
                last_node_embedding = torch.flatten(last_node_embedding.expand(last_node_embedding.shape[0],
                                                                               last_node_embedding.shape[1],
                                                                               self.Topk[len(path_node)-2],
                                                                               last_node_embedding.shape[3]),
                                                    1, 2)
                last_relation_embedding = last_relation_embedding.unsqueeze(2)
                last_relation_embedding = torch.flatten(last_relation_embedding.expand(last_relation_embedding.shape[0],
                                                                                       last_relation_embedding.shape[1],
                                                                                       self.Topk[len(path_node)-2],
                                                                                       last_relation_embedding.shape[3]),
                                                        1, 2)
            state_embedding = torch.cat([user_embedding, curr_node_embedding,
                                         last_node_embedding, last_relation_embedding], dim=-1)  # bz,  d(k-hop-1), dim

        return state_embedding

    def get_reward(self, user_embedding, path_nodes, path_type, user_score_max, done = None):
        target_score_list = []
        if done == None:
            return torch.FloatTensor([0 for i in range(path_nodes[0].shape[0])])
        else:
            for i in range(path_nodes[-1].shape[0]):
                if path_type[-1][i].item() == 1:
                    curr_news_id = path_nodes[-1][i]
                    news_embedding = self.trans_news_embedding(curr_news_id).squeeze()
                    target_score = torch.sum(user_embedding[i, :] * news_embedding, dim=-1)
                else:
                    target_score = torch.tensor(0).to(self.device)
                target_score_list.append(target_score)
            target_score_list = torch.stack(target_score_list)
            target_score = torch.sigmoid(target_score_list / user_score_max)
        return target_score

    def get_total_reward(self, step_rewards, path_reward_list, meta_reward_list, ap = 0.004, am = 0.01 ):
        total_reward_list = []
        for i in range(len(step_rewards)):
            total_reward = (torch.tensor(1 - ap - am) * torch.FloatTensor(step_rewards[i])).unsqueeze(1).to(self.device) + \
                           torch.tensor(ap).to(self.device) * path_reward_list[i].to(self.device) + \
                           torch.tensor(am).to(self.device) * meta_reward_list[i].to(self.device)
            total_reward_list.append(total_reward)
        return total_reward_list

    def get_best_reward(self, candidate_newsindex, user_embedding, path_nodes, user_score_max):
        total_path_num = 0
        curr_node_id = path_nodes[-1]
        curr_node_id_np = curr_node_id.detach().cpu().numpy()
        candidate_newsindex = candidate_newsindex.detach().cpu().numpy()

        for i in range(curr_node_id_np.shape[0]):
            if candidate_newsindex[i] in curr_node_id_np[i] and candidate_newsindex[i] != self.args.title_size - 1:
                index = np.argwhere(curr_node_id_np[i] == candidate_newsindex[i])
                path_num = len(index)
                total_path_num += path_num
        print('搜索到路径：{}'.format(total_path_num))
        news_embedding = self.trans_news_embedding(curr_node_id)
        user_embedding = user_embedding.unsqueeze(1)
        user_embedding.expand(user_embedding.shape[0], news_embedding.shape[1], user_embedding.shape[2])

        target_score = torch.sigmoid(torch.sum(user_embedding * news_embedding, dim = -1))
        target_score = torch.sigmoid(
            target_score / user_score_max.unsqueeze(-1).repeat(1, target_score.shape[1]))
        return target_score, total_path_num

    def cal_max_user_score(self, user_embedding, news_embedding):
        user_score = torch.sum(user_embedding * news_embedding, dim=-1)
        return user_score

    def step_update(self, act_probs_step, q_values_step, step_reward):
        curr_reward = step_reward
        advantage = curr_reward - q_values_step
        actor_loss = - act_probs_step * advantage.detach()
        critic_loss = advantage.pow(2)
        return critic_loss, actor_loss

    def get_path_nodes(self, weights, q_values, next_action, next_action_relation, next_action_type, topk):
        if len(weights.shape) <= 3:
            weights = torch.unsqueeze(weights, 1) # bz, 1, input_num, 1
            q_values = torch.unsqueeze(q_values, 1) # bz, 1, input_num, 1
            next_action = torch.unsqueeze(next_action, 1) # bz, 1, input_num
            next_action_relation = torch.unsqueeze(next_action_relation, 1) # bz, 1, input_num, 1
            next_action_type = torch.unsqueeze(next_action_type, 1)  # bz, 1, input_num, 1

        weights = weights.squeeze(-1) # bz, 1, input_num
        q_values = q_values.squeeze(-1) # bz, 1, input_num
        m = Categorical(weights) # bz, 1, input_num
        acts_idx = m.sample(sample_shape=torch.Size([topk])) # select_num, bz, 1
        acts_idx = acts_idx.permute(1,2,0) # bz, 1, dselect_num
        shape0 = acts_idx.shape[0] # bz
        shape1 = acts_idx.shape[1] # 1

        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2]) # bz , select_num, bz
        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2]) # bz * 1 , input_num
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])  # bz * 1 , entity_num

        next_action = next_action.reshape(next_action.shape[0] * next_action.shape[1], next_action.shape[2])  # bz * 1 , input_num
        next_action_relation = next_action_relation.reshape(next_action_relation.shape[0] * next_action_relation.shape[1], next_action_relation.shape[2])  # bz * 1 , input_num
        next_action_type = next_action_type.reshape(next_action_type.shape[0] * next_action_type.shape[1],  next_action_type.shape[2])  # bz * 1 , input_num

        next_action_select = next_action.gather(1, acts_idx) #  select_num, bz
        next_action_relation_select = next_action_relation.gather(1, acts_idx)#  select_num, bz
        next_action_type_select = next_action_type.gather(1, acts_idx)  # select_num, bz

        weights = weights.gather(1, acts_idx) #  select_num, bz
        q_values = q_values.gather(1, acts_idx) # select_num, bz
        weights = weights.reshape(shape0, shape1 *  weights.shape[1]) # 1, select_num * bz
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1]) # 1, select_num * bz

        next_action_select = next_action_select.reshape(shape0, shape1 * next_action_select.shape[1]) #  bz, select_num * 1
        next_action_relation_select = next_action_relation_select.reshape(shape0, shape1 * next_action_relation_select.shape[1]) #  bz, select_num * 1
        next_action_type_select = next_action_type_select.reshape(shape0, shape1 * next_action_type_select.shape[1])  # bz, select_num * 1

        return weights.to(self.device), q_values.to(self.device), next_action_select.to(self.device), \
               next_action_relation_select.to(self.device), next_action_type_select.to(self.device)

    def get_action_input(self, step_path_node, step_path_type):
        next_action = []
        next_action_relation = []
        next_action_type = []
        if len(step_path_node.shape) < 2:
            step_path_node = step_path_node.unsqueeze(-1)
            step_path_type = step_path_type.unsqueeze(-1)

        for i in range(len(step_path_type)):
            next_action.append([])
            next_action_relation.append([])
            next_action_type.append([])
            for j in range(len(step_path_node[i])):
                if (step_path_type[i][j]).item() == 0:
                    next_action[-1].append(self.user_click_dict[int(step_path_node[i][j])][:10])
                    next_action_relation[-1].append([1 for k in range(10)])
                    next_action_type[-1].append([1 for k in range(10)])

                elif (step_path_type[i][j]).item() == 1:
                    if len(self.news_entity_dict[int(step_path_node[i][j])]) < 10:
                        temp = self.news_entity_dict[int(step_path_node[i][j])]
                        temp1 = [2 for k in range(len(temp))]
                        temp.extend([int(step_path_node[i][j])] * (10-len(temp)))
                        temp1.extend([1] * (10-len(temp1)))
                    else:
                        temp = self.news_entity_dict[int(step_path_node[i][j])][:10]
                        temp1 = [2 for k in range(10)]

                    next_action[-1].append(temp)
                    next_action_relation[-1].append(temp1)
                    next_action_type[-1].append([2 for k in range(10)])
                else:
                    entity_action = []
                    entity_relation = []
                    entity_type = []
                    if int(step_path_node[i][j].data.cpu().numpy()) in self.entity_adj:
                        entity_action.extend(self.entity_adj[int(step_path_node[i][j].data.cpu().numpy())])
                        entity_relation.extend(self.relation_adj[int(step_path_node[i][j].data.cpu().numpy())])
                        entity_type.extend([2 for k in range(len(self.entity_adj[int(step_path_node[i][j].data.cpu().numpy())]))])

                    if int(step_path_node[i][j].data.cpu().numpy()) in self.entity_news_dict:
                        entity_action.extend(self.entity_news_dict[int(step_path_node[i][j].data.cpu().numpy())])
                        entity_relation.extend([0 for k in range(len(entity_action))])
                        entity_type.extend([1 for k in range(len(entity_action))])

                    if len(entity_action) < 10:
                        entity_relation.extend([0 for k in range(10 - len(entity_action))])
                        entity_type.extend([2 for k in range(10 - len(entity_action))])
                        entity_action.extend([int(step_path_node[i][j].data.cpu().numpy()) for k in range(10-len(entity_action))])


                    entity_action = entity_action[:10]
                    entity_relation = entity_relation[:10]
                    entity_type = entity_type[:10]

                    # index_list = list(np.random.randint(len(entity_action), size=10))
                    # entity_action_select = []
                    # entity_relation_select = []
                    # entity_type_select = []
                    # for index in index_list:
                    #     entity_action_select.append(entity_action[index])
                    #     entity_relation_select.append(entity_relation[index])
                    #     entity_type_select.append(entity_type[index])

                    next_action[-1].append(entity_action)
                    next_action_relation[-1].append(entity_relation)
                    next_action_type[-1].append(entity_type)
                if len(next_action[-1][-1]) != 10 or \
                    len(next_action_relation[-1][-1]) != 10 or\
                    len(next_action_type[-1][-1]) != 10:
                    print(len(next_action_relation[-1][-1]))
        next_action = torch.tensor(next_action).to(self.device)
        next_action_relation = torch.tensor(next_action_relation).to(self.device)
        next_action_type = torch.tensor(next_action_type).to(self.device)
        return next_action, next_action_relation, next_action_type

    def get_final_action(self, step_path_node, step_path_type, step_path_relations):
        next_action = []
        next_action_relation = []
        next_action_type = []
        if len(step_path_node.shape) < 2:
            step_path_node = step_path_node.unsqueeze(-1)
            step_path_type = step_path_type.unsqueeze(-1)
            step_path_relations = step_path_relations.unsqueeze(-1)
        for i in range(len(step_path_type)):
            next_action.append([])
            next_action_relation.append([])
            next_action_type.append([])
            for j in range(len(step_path_node[i])):
                if (step_path_type[i][j]).item() == 0:
                    next_action[-1].append(self.user_click_dict[int(step_path_node[i][j])])
                    next_action_relation[-1].append([1 for k in range(len(self.user_click_dict[int(step_path_node[i][j])]))])
                    next_action_type[-1].append([1 for k in range(len(self.user_click_dict[int(step_path_node[i][j])]))])

                elif (step_path_type[i][j]).item() == 1:
                    next_action[-1].append([(step_path_node[i][j]).item() for k in range(10)])
                    next_action_relation[-1].append([0 for k in range(10)])
                    next_action_type[-1].append([1 for k in range(10)])
                else:
                    if len(self.entity_news_dict[int(step_path_node[i][j])]) < 10:
                        temp = self.entity_news_dict[int(step_path_node[i][j])]
                        temp.extend([self.args.title_num -1] * (10-len(temp)))
                        next_action[-1].append(temp)
                    else:
                        next_action[-1].append(self.entity_news_dict[int(step_path_node[i][j])][:10])
                    next_action_relation[-1].append([0 for k in range(10)])
                    next_action_type[-1].append([1 for k in range(10)])

        next_action = torch.tensor(next_action).to(self.device)
        next_action_relation = torch.tensor(next_action_relation).to(self.device)
        next_action_type = torch.tensor(next_action_type).to(self.device)
        return next_action, next_action_relation, next_action_type

    def get_action_embedding(self, next_action, next_action_relation, next_action_type):
        action_embedding = []
        for i in range(next_action.shape[0]):
            action_embedding.append([])
            for j in range(next_action.shape[1]):
                action_embedding[-1].append([])
                for m in range(next_action.shape[2]):
                    next_action_index = int(next_action[i][j][m].data.cpu().numpy())
                    if (next_action_type[i][j][m]).item() == 0:
                        action_embedding[-1][-1].append(self.user_embedding(torch.tensor(next_action_index).to(self.device)).detach().cpu().numpy())
                    elif (next_action_type[i][j][m]).item() == 1:
                        action_embedding[-1][-1].append(self.trans_news_embedding(torch.tensor(next_action_index).to(self.device)).detach().cpu().numpy())
                    else:
                        action_embedding[-1][-1].append(self.entity_embedding(torch.tensor(next_action_index).to(self.device)).detach().cpu().numpy())
        action_type_embedding = self.type_embedding(next_action_type)
        action_relation_embedding = self.relation_embedding(next_action_relation)
        action_embedding = torch.FloatTensor(np.array(action_embedding)).to(self.device)
        return action_embedding + action_relation_embedding + action_type_embedding

    def sample_demo_path(self, demon_path_node, demon_path_relation, demon_path_type):
        sample_demon_path_node = []
        sample_demon_path_relation = []
        sample_demon_path_type = []
        sample = random.sample(range(0, 10), 1)
        for depth in range(len(demon_path_node)):
            sample_demon_path_node.append(demon_path_node[depth][:, sample])
            sample_demon_path_relation.append(demon_path_relation[depth][:, sample])
            sample_demon_path_type.append(demon_path_type[depth][:, sample])
        return sample_demon_path_node, sample_demon_path_relation, sample_demon_path_type

    def get_curr_action_embedding(self, path_node, path_type):
        curr_node_embedding = []
        for i in range(path_type.shape[0]):
            curr_node_embedding_temp = []
            for j in range(path_type.shape[1]):
                node_index = path_node[i][j]
                if path_type[i][j].item() == 0:
                    node_embedding = self.user_embedding(node_index.to(self.device))
                elif path_type[i][j].item() == 1:
                    node_embedding = self.trans_news_embedding(node_index.to(self.device))
                else:
                    node_embedding = self.entity_embedding(node_index.to(self.device))
                curr_node_embedding_temp.append(node_embedding)
            curr_node_embedding.append(torch.stack(curr_node_embedding_temp))
        curr_node_embedding = torch.stack(curr_node_embedding).to(self.device)
        return curr_node_embedding

    def Path_discriminator(self, path_nodes, path_relations, path_type, demon_path_node, demon_path_relation, demon_path_type, user_embedding):
        path_loss_list = []
        path_reward_list = []
        for depth in range(1, self.max_path_long + 1):
            state_input = self.get_state_embedding(user_embedding, path_nodes[:depth], path_relations[:depth], path_type[:depth])
            action_input = self.get_curr_action_embedding(path_nodes[depth], path_type[depth])
            hp = torch.tanh(torch.cat([state_input.squeeze(), action_input.squeeze()], dim = -1))
            Dp = torch.sigmoid(self.path_discriminator_l2(torch.tanh(self.path_discriminator_l1(hp))))

            demon_state = self.get_state_embedding(user_embedding, demon_path_node[:depth], demon_path_relation[:depth], demon_path_type[:depth])
            demon_action = self.get_curr_action_embedding(demon_path_node[depth], demon_path_type[depth])
            demon_hp = torch.tanh(torch.cat([demon_state.squeeze(), demon_action.squeeze()], dim=-1))
            demon_Dp = torch.sigmoid(self.path_discriminator_l2(torch.tanh(self.path_discriminator_l1(demon_hp))))
            path_loss = nn.BCELoss()(demon_Dp, torch.ones_like(demon_Dp))
            # path_loss = - torch.log(demon_Dp) + torch.log(1-Dp)
            path_reward = torch.log(Dp) - torch.log(1 - Dp)
            path_loss_list.append(path_loss)
            path_reward_list.append(path_reward)
        return path_loss_list, path_reward_list

    def Meta_discriminator(self, path_type, demon_path_type):
        meta_loss_list = []
        meta_reward_list = []
        type_input = None
        demon_type= None
        for depth in range(1, self.max_path_long + 1):
            path_type_select = path_type[:depth]
            demon_type_select = demon_path_type[:depth]

            for i in range(len(path_type_select)):
                if i == 0:
                    type_input = self.type_embedding(path_type_select[i].to(torch.int64)).squeeze()
                    demon_type = self.type_embedding(demon_type_select[i].to(torch.int64)).squeeze()
                else:
                    type_input = torch.cat([type_input, self.type_embedding(path_type_select[i].to(torch.int64)).squeeze()], dim = -1).to(self.device)
                    demon_type = torch.cat([demon_type, self.type_embedding(demon_type_select[i].to(torch.int64)).squeeze()], dim = -1).to(self.device)
            Dm = torch.sigmoid(self.meta_discriminator_l2(torch.tanh(self.meta_discriminator_l1[depth](type_input.squeeze()))))
            demon_Dm = torch.sigmoid(self.meta_discriminator_l2(torch.tanh(self.meta_discriminator_l1[depth](demon_type.squeeze()))))
            # meta_loss = - torch.log(demon_Dm) + torch.log(1 - Dm)
            meta_loss = nn.BCELoss()(demon_Dm, torch.ones_like(demon_Dm))
            meta_reward = torch.log(Dm) - torch.log(1 - Dm)
            meta_loss_list.append(meta_loss)
            meta_reward_list.append(meta_reward)
        return meta_loss_list, meta_reward_list

    def get_batch_demo_path(self, user_index):
        demo_paths_index = self.demo_paths_index[user_index]
        demo_type_index = self.demo_type_index[user_index]
        demo_relations_index = self.demo_relations_index[user_index]
        batch_path_index = []
        batch_type_index = []
        batch_relation_index = []
        for i in range(demo_paths_index.shape[2]):
            batch_path_index.append(demo_paths_index[:, :, i])
            batch_type_index.append(demo_type_index[:, :, i])
            batch_relation_index.append(demo_relations_index[:, :, i])
        return batch_path_index, batch_type_index, batch_relation_index


    def get_rec_score(self, user_embedding, news_embedding, path_nodes, path_relations, path_type):
        rec_score = torch.sum((user_embedding * news_embedding), dim=-1)
        return rec_score


    def forward(self, user_index, candidate_newsindex, user_clicked_newsindex):
        depth = 0

        user_index_exp = user_index.unsqueeze(1)
        user_index_exp = user_index_exp.expand(user_index_exp.shape[0], self.sample_size)
        user_index = torch.flatten(user_index_exp, 0, 1).to(self.device)
        candidate_newsindex = torch.flatten(candidate_newsindex, 0, 1).to(self.device)

        news_embedding = self.trans_news_embedding(candidate_newsindex)  # bz * 5, dim
        user_embedding = self.tanh(self.user_embedding(user_index))
        
        batch_demo_path_index, batch_demo_type_index, batch_demo_relation_index = self.get_batch_demo_path(user_index)  # bz * 5, user_clicked_num, path_long
        
        user_score_max = self.cal_max_user_score(user_embedding, news_embedding)

        # 开始路径探索
        path_nodes = [user_index]
        path_type = [torch.zeros(user_index.shape[0]).to(self.device)]
        path_relations = [torch.zeros(user_index.shape[0]).to(self.device)]
        act_probs_steps = []
        step_rewards = []
        q_values_steps = []

        next_action, next_action_relation, next_action_type = self.get_action_input(path_nodes[-1], path_type[-1])
        action_embedding = self.get_action_embedding(next_action, next_action_relation, next_action_type)  # torch.Size([250, 1, 10, 100])
        state_embedding = self.get_state_embedding(user_embedding, path_nodes, path_relations, path_type)  # bz*5， state_dim

        topk = 1
        while (depth <= self.max_path_long):
            depth += 1
            act_probs, q_values = self.policy_net(state_embedding, action_embedding) # bz*5,input_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            act_probs, q_values, \
            step_nodes, step_relations, step_type = self.get_path_nodes(act_probs, q_values, next_action, next_action_relation, next_action_type, topk)
            path_nodes.append(step_nodes) # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            path_relations.append(step_relations) # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            path_type.append(step_type) # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            act_probs_steps.append(act_probs)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            q_values_steps.append(q_values)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            state_embedding = self.get_state_embedding(news_embedding, path_nodes, path_relations, path_type)  # bz*5, dim # bz*5, dim # bz*5, dim
            if depth < self.max_path_long-1:
                next_action, next_action_relation, next_action_type = self.get_action_input(path_nodes[-1], path_type[-1])
            elif depth == self.max_path_long-1:
                next_action, next_action_relation, next_action_type = self.get_final_action(path_nodes[-1], path_type[-1], path_relations[-1])
            else:
                break
            action_embedding = self.get_action_embedding(next_action, next_action_relation, next_action_type)  # torch.Size([250, 1, 10, 100])
            step_reward = self.get_reward(user_embedding, path_nodes, path_type, user_score_max)  # bz, 1 # bz, 1 # bz, 1
            step_rewards.append(step_reward)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]

        sample_demon_path_node, sample_demon_path_relation, sample_demon_path_type = self.sample_demo_path(batch_demo_path_index,
                                                                                                           batch_demo_relation_index,
                                                                                                           batch_demo_type_index)
        path_step_loss, path_reward_list = self.Path_discriminator(path_nodes, path_relations, path_type,
                                                                   sample_demon_path_node, sample_demon_path_relation, sample_demon_path_type,
                                                                   user_embedding)

        meta_step_loss, meta_reward_list = self.Meta_discriminator(path_type, sample_demon_path_type)
        total_rewards_list = self.get_total_reward(step_rewards, path_reward_list, meta_reward_list)

        score = self.get_reward(user_embedding, path_nodes, path_type, user_score_max, done = True)
        return act_probs_steps, q_values_steps, total_rewards_list, \
               path_nodes, path_relations, score, path_step_loss, meta_step_loss

    def test(self, user_index, candidate_newsindex, user_clicked_newsindex):
        depth = 0
        user_index_exp = user_index.unsqueeze(1)
        user_index_exp = user_index_exp.expand(user_index_exp.shape[0], self.sample_size)
        user_index = torch.flatten(user_index_exp, 0, 1).to(self.device)
        candidate_newsindex = torch.flatten(candidate_newsindex, 0, 1).to(self.device)

        news_embedding = self.trans_news_embedding(candidate_newsindex)  # bz * 5, dim
        user_embedding = self.tanh(self.user_embedding(user_index))

        user_score = self.cal_max_user_score(user_embedding, news_embedding)

        # 开始路径探索
        path_nodes = [user_index]
        path_type = [torch.zeros(user_index.shape[0]).to(self.device)]
        path_relations = [torch.zeros(user_index.shape[0]).to(self.device)]
        act_probs_steps = []
        q_values_steps = []

        next_action, next_action_relation, next_action_type = self.get_action_input(path_nodes[-1], path_type[-1])
        action_embedding = self.get_action_embedding(next_action, next_action_relation, next_action_type)  # torch.Size([250, 1, 10, 100])
        state_embedding = self.get_state_embedding(user_embedding, path_nodes, path_relations, path_type)  # bz*5， state_dim
        self.Topk = [5, 3, 2, 1, 1]
        
        while (depth < self.max_path_long):
            topk = self.Topk[depth]
            depth += 1
            act_probs, q_values = self.policy_net(state_embedding, action_embedding)  # bz*5,input_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            act_probs, q_values, \
            step_nodes, step_relations, step_type = self.get_path_nodes(act_probs, q_values,
                                                                        next_action,
                                                                        next_action_relation,
                                                                        next_action_type,
                                                                        topk)
            path_nodes.append(step_nodes)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            path_relations.append(step_relations)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            path_type.append(step_type)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            act_probs_steps.append(act_probs)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            q_values_steps.append(q_values)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            state_embedding = self.get_state_embedding(news_embedding,
                                                       path_nodes,
                                                       path_relations,
                                                       path_type)  # bz*5, dim # bz*5, dim # bz*5, dim

            if depth < self.max_path_long - 2:
                next_action, \
                next_action_relation, \
                next_action_type = self.get_action_input(path_nodes[-1], path_type[-1])
            elif depth == self.max_path_long - 2:
                next_action, \
                next_action_relation, \
                next_action_type = self.get_final_action(path_nodes[-1], path_type[-1], path_relations[-1])
                path_nodes.append(torch.flatten(next_action, -2, -1))
                break
            else:
                break
            action_embedding = self.get_action_embedding(next_action, next_action_relation, next_action_type)  # torch.Size([250, 1, 10, 100])


        score, total_path_num = self.get_best_reward(candidate_newsindex, user_embedding, path_nodes, user_score)
        best = torch.max(score, dim=-1)
        best_score = best.values
        best_path = best.indices
        return path_nodes, total_path_num, path_relations, best_score, best_path

