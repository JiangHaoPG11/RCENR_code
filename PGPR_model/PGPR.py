import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception('history length should be one of {0, 1, 2}')

    def __call__(self, user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                 older_relation_embed):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed])
        elif self.history_len == 2:
            return np.concatenate([user_embed, node_embed, last_node_embed, last_relation_embed, older_node_embed,
                                   older_relation_embed])
        else:
            raise Exception('mode should be one of {full, current}')

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
                state_input = state_input.expand(state_input.shape[0], action_input.shape[1],state_input.shape[2])
            else:
                if len(state_input.shape) == 2:
                    state_input = torch.unsqueeze(state_input, 1)
                    state_input = torch.unsqueeze(state_input, 2)
                    state_input = state_input.expand(state_input.shape[0], action_input.shape[1], action_input.shape[2], state_input.shape[3])
                if len(state_input.shape) == 3:
                    state_input = torch.unsqueeze(state_input, 2)
                    state_input = state_input.expand(state_input.shape[0], state_input.shape[1], action_input.shape[2], state_input.shape[3])
        # Actor
        actor_x = self.elu(self.actor_l1(torch.cat([state_input, action_input], dim=-1)))
        actor_out = self.elu(self.actor_l2(actor_x))
        act_probs = self.sigmoid(self.actor_l3(actor_out))
        # Critic
        critic_x = self.elu(self.critic_l1(torch.cat([state_input, action_input], dim=-1)))
        critic_out = self.elu(self.critic_l2(critic_x))
        q_actions = self.sigmoid(self.critic_l3(critic_out))
        return act_probs, q_actions

class PGPR(torch.nn.Module):

    def __init__(self, args, news_entity_dict, entity_news_dict, news_title_embedding, entity_embedding,
                 relation_embedding, entity_adj, relation_adj, entity_dict, device):
        super(PGPR, self).__init__()
        self.args = args
        self.device = device
        self.news_entity_dict = news_entity_dict
        self.entity_news_dict = entity_news_dict

        print(entity_adj)
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.entity_adj, self.relation_adj = self._reconstruct_kg_adj()

        self.entity_dict = entity_dict

        self.MAX_DEPTH = 3
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        #news_title_embedding = news_title_embedding.tolist()
        #news_title_embedding.append(np.random.normal(-0.1, 0.1, 768))
        self.news_title_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(news_title_embedding))
        self.user_embedding = nn.Embedding(self.args.user_size, self.args.embedding_size)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)


        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.innews_relation = nn.Embedding(1, self.args.embedding_size)

        self.policy_net = Net(self.entity_dict, entity_embedding, self.device)
        self.target_net = Net(self.entity_dict, entity_embedding, self.device)
        self.state_gen = KGState(self.args.embedding_size)

    def _reconstruct_kg_adj(self):
        entity_adj_update = {}
        relation_adj_update = {}
        for key, value in self.entity_adj.items():
            if key != 0:
                temp = value.pop(0)
                if type(temp) == int:
                    temp = [temp]
                entity_adj_update[key] = temp
                relation_adj_update[key] = self.relation_adj[key][:len(temp)]
        return entity_adj_update, relation_adj_update

    def trans_news_embedding(self, news_index):
        trans_news_embedding = self.news_title_embedding(news_index)  # bz * 5, news_entity_num, dim
        trans_news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(trans_news_embedding))))
        return trans_news_embedding

    def get_path_list(self, path_nodes, batch_size):
        path_nodes_list = []
        for i in range(batch_size):
            path_nodes_list.append([])
        for i in range(len(path_nodes)):
            for j in range(len(path_nodes[i])):
                path_nodes_list[j].extend(list(map(lambda x:str(x), path_nodes[i][j].data.cpu().numpy())))
        return path_nodes_list

    def get_entities_news_batch(self, entity_id):
        news = []
        relations = []
        if entity_id.shape[1] == 1:
            for i in range(len(entity_id)):
                if int(entity_id[i]) == 0:
                    news.append([(self.args.title_num - 1) for k in range(20)])
                elif len(self.entity_news_dict[int(entity_id[i])]) < 20:
                    temp = self.entity_news_dict[int(entity_id[i])]
                    temp.extend([self.args.title_num - 1] * (20-len(self.entity_news_dict[int(entity_id[i])])))
                    news.append(temp)
                else:
                    news.append(self.entity_news_dict[int(entity_id[i])][:20])
                relations.append([0 for k in range(len(self.entity_news_dict[int(entity_id[i])]))])
            news = torch.LongTensor(news).to(self.device)
            relations = torch.LongTensor(relations).to(self.device)
        else:
            for i in range(len(entity_id)):
                news_single = []
                relations_single = []
                for entity in entity_id[i]:
                    if int(entity) == 0:
                        news_single.append([(self.args.title_num - 1)for k in range(20)])
                    elif len(self.entity_news_dict[int(entity)]) < 20:
                        temp = self.entity_news_dict[int(entity)]
                        temp.extend([self.args.title_num - 1] * (20 - len(self.entity_news_dict[int(entity)])))
                        news_single.append(temp)
                    else:
                        news_single.append(self.entity_news_dict[int(entity)][:20])
                    relations_single.append([0 for k in range(20)])
                news.append(news_single)
                relations.append(relations_single)
            news = torch.LongTensor(news).to(self.device)
            relations = torch.LongTensor(relations).to(self.device)
        return news, relations

    def get_reward(self, user_embedding, path_nodes, user_clicked_score_max, done = None):
        if done == None:
            return [0 for i in range(path_nodes[0].shape[0])]
        else:
            curr_news_id = path_nodes[-1]
            news_embedding = self.news_title_embedding(curr_news_id).squeeze()
            news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(news_embedding))))
            target_score = torch.sum(user_embedding * news_embedding, dim = -1)
            # target_score = F.softmax(target_score, dim = -1)
            target_score = torch.sigmoid(target_score / user_clicked_score_max)
        target_score = target_score.to(self.device)
        return target_score

    def get_best_reward(self, candidate_newsindex, user_embedding, path_nodes, user_clicked_score_max):
        total_path_num = 0
        curr_node_id = path_nodes[-1]
        curr_node_id_np = curr_node_id.detach().cpu().numpy()
        candidate_newsindex = candidate_newsindex.detach().cpu().numpy()

        for i in range(curr_node_id_np.shape[0]):
            if candidate_newsindex[i] in curr_node_id_np[i] and candidate_newsindex[i] != self.args.title_num - 1:
                index = np.argwhere(curr_node_id_np[i] == candidate_newsindex[i])
                path_num = len(index)
                total_path_num += path_num
        print('搜索到路径：{}'.format(total_path_num))

        news_embedding = self.news_title_embedding(curr_node_id).squeeze()
        news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(news_embedding))))
        user_embedding = user_embedding.unsqueeze(1).repeat(1, news_embedding.shape[1], 1)
        target_score = torch.sum(user_embedding * news_embedding, dim = -1)
        # target_score = F.softmax(target_score, dim = -1)
        target_score = torch.sigmoid(target_score / user_clicked_score_max.unsqueeze(-1).repeat(1, target_score.shape[1]))
        return target_score, total_path_num

    def get_news_entities_batch(self, newsids):
        news_entities = []
        news_relations = []
        for i in range(len(newsids)):
            news_entities.append(self.news_entity_dict[int(newsids[i])])
            news_relations.append([0 for k in range(len(self.news_entity_dict[int(newsids[i])]))])
        news_entities = torch.tensor(news_entities).to(self.device)
        news_relations = torch.tensor(news_relations).to(self.device)# bz, news_entity_num
        return news_entities, news_relations

    def get_user_clicked_batch(self, user_clicked_index):
        clicked_relations = []
        for i in range(len(user_clicked_index)):
            clicked_relations.append([0 for k in range(user_clicked_index.shape[1])])
        # user_clicked_news = torch.tensor(user_clicked_index).to(self.device)
        clicked_relations = torch.tensor(clicked_relations).to(self.device)
        return user_clicked_index, clicked_relations

    def get_next_action(self, state_id_input_batch):
        next_action_id = []
        next_action_r_id = []
        for i in range(len(state_id_input_batch)):
            next_action_id.append([])
            next_action_r_id.append([])
            for j in range(len(state_id_input_batch[i])):
                if int(state_id_input_batch[i][j].data.cpu().numpy()) in self.entity_adj:
                    next_action_id[-1].append(self.entity_adj[int(state_id_input_batch[i][j].data.cpu().numpy())])
                    next_action_r_id[-1].append(self.relation_adj[int(state_id_input_batch[i][j].data.cpu().numpy())])
                else:
                    next_action_id[-1].append([int(state_id_input_batch[i][j].data.cpu().numpy()) for k in range(20)])
                    next_action_r_id[-1].append([0 for k in range(20)])
                if len(next_action_id[-1][-1]) < 20:
                    next_action_id[-1][-1].extend([int(state_id_input_batch[i][j].data.cpu().numpy()) for k in range(20 - len(next_action_id[-1][-1]))])
                    next_action_r_id[-1][-1].extend([0 for k in range(20 - len(next_action_r_id[-1][-1]))])
        next_action_space_id = torch.LongTensor(next_action_id).to(self.device) # bz, d(k-hop) , 20
        next_state_space_embedding = self.entity_embedding(next_action_space_id)  # bz, d(k-hop) , 20 dim
        next_action_r_id = torch.LongTensor(next_action_r_id).to(self.device)# bz, d(k-hop) , 20, 20
        next_action_r = self.relation_embedding(next_action_r_id) # bz, d(k-hop) , 20,  dim
        return next_action_space_id, next_state_space_embedding, next_action_r_id, next_action_r

    def get_advantage(self, curr_reward,  q_value):
        advantage = curr_reward - q_value
        return advantage

    def get_actor_loss(self, act_probs_step, advantage):
        actor_loss = -act_probs_step * advantage.detach()
        return actor_loss

    def get_critic_loss(self, advantage):
        critic_loss = torch.pow(advantage, 2)
        return critic_loss

    def step_update(self, act_probs_step, q_values_step, step_reward, alpha1=0.9, alpha2 = 0.1):
        curr_reward = step_reward.to(self.device)
        advantage = self.get_advantage(curr_reward, q_values_step) #curr_reward - q_values_step
        actor_loss = self.get_actor_loss(torch.log(act_probs_step), advantage)#self.get_actor_loss(act_probs_step, q_values_step)#self.get_actor_loss(torch.log(act_probs_step), advantage) # -act_probs_step * advantage #problem2
        critic_loss = advantage.pow(2)#self.get_critic_loss(advantage) #advantage.pow(2)
        return critic_loss, actor_loss

    def get_path_nodes(self, weights, q_values, action_id_input, relation_id_input, topk):
        if len(weights.shape) <= 3:
            weights =torch.unsqueeze(weights, 1) # bz, 1, entity_num, 1
            q_values = torch.unsqueeze(q_values, 1) # bz, 1, entity_num, 1
            action_id_input = torch.unsqueeze(action_id_input, 1) # bz, 1, entity_num
            relation_id_input = torch.unsqueeze(relation_id_input, 1) # bz, 1, entity_num
        weights = weights.squeeze(-1) # bz, 1, entity_num
        q_values = q_values.squeeze(-1) # bz, 1, entity_num
        m = Categorical(weights) # bz, 1, entity_num
        acts_idx = m.sample(sample_shape=torch.Size([topk])) # d(k-hop), bz, 1
        acts_idx = acts_idx.permute(1,2,0) # bz, 1, d(k-hop)
        shape0 = acts_idx.shape[0] # bz
        shape1 = acts_idx.shape[1] # 1
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2]) # bz , d(k-hop), bz
        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2]) # bz * 1 , entity_num
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])  # bz * 1 , entity_num
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])  # bz * 1 , entity_num
        relation_id_input = relation_id_input.reshape(relation_id_input.shape[0] * relation_id_input.shape[1], relation_id_input.shape[2])  # bz * 1 , entity_num
        state_id_input_value = action_id_input.gather(1, acts_idx) #  d(k-hop), bz
        relation_id_selected = relation_id_input.gather(1, acts_idx)#  d(k-hop), bz
        weights = weights.gather(1, acts_idx) #  d(k-hop), bz
        q_values = q_values.gather(1, acts_idx) #  d(k-hop), bz
        weights = weights.reshape(shape0, shape1 *  weights.shape[1]) # 1, d(k-hop) * bz
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1]) # 1, d(k-hop) * bz
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 * state_id_input_value.shape[1]) #  bz, d(k-hop) * 1
        relation_id_selected = relation_id_selected.reshape(shape0, shape1 * relation_id_selected.shape[1]) #  bz, d(k-hop) * 1
        return weights, q_values, state_id_input_value, relation_id_selected

    def get_last_path_nodes(self, weights, q_values, action_id_input, topk):
        if len(weights.shape) <= 3:
            weights =torch.unsqueeze(weights, 1) # bz, 1, entity_num, 1
            q_values = torch.unsqueeze(q_values, 1) # bz, 1, entity_num, 1
            action_id_input = torch.unsqueeze(action_id_input, 1) # bz, 1, entity_num
        weights = weights.squeeze(-1) # bz, 1, entity_num
        q_values = q_values.squeeze(-1) # bz, 1, entity_num
        m = Categorical(weights) # bz, 1, entity_num
        acts_idx = m.sample(sample_shape=torch.Size([topk])) # d(k-hop), bz, 1
        acts_idx = acts_idx.permute(1,2,0) # bz, 1, d(k-hop)
        shape0 = acts_idx.shape[0] # bz
        shape1 = acts_idx.shape[1] # 1
        acts_idx = acts_idx.reshape(acts_idx.shape[0] * acts_idx.shape[1], acts_idx.shape[2]) # bz , d(k-hop), bz
        weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2]) # bz * 1 , entity_num
        q_values = q_values.reshape(q_values.shape[0] * q_values.shape[1], q_values.shape[2])  # bz * 1 , entity_num
        action_id_input = action_id_input.reshape(action_id_input.shape[0] * action_id_input.shape[1], action_id_input.shape[2])  # bz * 1 , entity_num
        state_id_input_value = action_id_input.gather(1, acts_idx) #  d(k-hop), bz
        weights = weights.gather(1, acts_idx) #  d(k-hop), bz
        q_values = q_values.gather(1, acts_idx) #  d(k-hop), bz
        weights = weights.reshape(shape0, shape1 *  weights.shape[1]) # 1, d(k-hop) * bz
        q_values = q_values.reshape(shape0, shape1 * q_values.shape[1]) # 1, d(k-hop) * bz
        state_id_input_value = state_id_input_value.reshape(shape0, shape1 * state_id_input_value.shape[1]) #  bz, d(k-hop) * 1
        return weights, q_values, torch.LongTensor(state_id_input_value)

    def get_state_input(self, user_embedding, path_node, path_relation):
        padding_embedding = torch.tensor(np.zeros([user_embedding.shape[0], 100])).to(torch.float).to(self.device)
        if len(path_node) == 0:
            state_embedding = torch.cat([user_embedding, padding_embedding, padding_embedding, padding_embedding], dim=-1)
        elif len(path_node) == 1:
            curr_clicked_news_embedding = self.news_title_embedding(path_node[-1]).squeeze()  # bz,  1 ，dim
            curr_clicked_news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(curr_clicked_news_embedding))))
            state_embedding = torch.cat([user_embedding, curr_clicked_news_embedding, padding_embedding, padding_embedding],dim= -1)  # bz,  d(k-hop-1), dim
        elif len(path_node) == 2:
            curr_clicked_news_embedding = self.news_title_embedding(path_node[-2]).squeeze()  # bz,  1 ，dim
            curr_clicked_news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(curr_clicked_news_embedding))))
            curr_entity_embedding = self.entity_embedding(path_node[-1]).squeeze() #  bz,  1 ，dim
            if len(user_embedding.shape) != len(curr_entity_embedding.shape):
                user_embedding = user_embedding.unsqueeze(1).repeat(1, curr_entity_embedding.shape[1], 1)
                curr_clicked_news_embedding = curr_clicked_news_embedding.unsqueeze(1).repeat(1, curr_entity_embedding.shape[1], 1)
                padding_embedding = torch.tensor(np.zeros([user_embedding.shape[0], curr_entity_embedding.shape[1], 100])).to(torch.float)
            state_embedding = torch.cat([user_embedding.to(self.device), curr_entity_embedding.to(self.device), curr_clicked_news_embedding.to(self.device), padding_embedding.to(self.device)], dim=-1) #  bz,  d(k-hop-1), dim
        else:
            curr_entity_embedding = self.entity_embedding(path_node[-1]).squeeze()   # bz, 1，dim
            last_entity_embedding = self.entity_embedding(path_node[-2]).squeeze()   # bz, 1，dim
            last_relation_embedding = self.relation_embedding(path_relation[-2]).squeeze()   # bz, 1, dim
            if len(user_embedding.shape) != len(curr_entity_embedding.shape):
                user_embedding = user_embedding.unsqueeze(1).repeat(1, curr_entity_embedding.shape[1], 1)
                last_entity_embedding = last_entity_embedding.unsqueeze(2)
                last_entity_embedding = torch.flatten(last_entity_embedding.repeat(1, 1, self.Topk[len(path_node)-2], 1), 1, 2)
                last_relation_embedding = last_relation_embedding.unsqueeze(2)
                last_relation_embedding = torch.flatten(last_relation_embedding.repeat(1, 1, self.Topk[len(path_node)-2], 1), 1, 2)

            state_embedding = torch.cat([user_embedding.to(self.device), curr_entity_embedding.to(self.device), last_entity_embedding.to(self.device), last_relation_embedding.to(self.device)],dim=-1)  # bz,  d(k-hop-1), dim
        return state_embedding

    def predict_path(self, newsid, uid, news_feature):
        self.news_entity_dict[newsid] = news_feature[0]
        self.news_title_embedding = news_feature[1]
        prediction = self.forward([newsid], [uid]).cpu().data.numpy()
        path_nodes = prediction[6]
        path_relation = prediction[8]
        return path_nodes, path_relation

    def cal_user_clicked_score(self, user_index, user_clicked_newindex):
        user_embedding = (self.tanh(self.user_embedding(user_index))).unsqueeze(1).repeat(1,user_clicked_newindex.shape[1],1)
        clicked_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(self.news_title_embedding(user_clicked_newindex)))))
        user_clicked_score = torch.sum(user_embedding * clicked_embedding, dim=-1)
        user_clicked_score_max = torch.max(user_clicked_score, dim=-1).values
        user_clicked_score_max = torch.flatten(user_clicked_score_max.unsqueeze(1).repeat(1, self.args.sample_size), 0,  1)
        return user_clicked_score_max

    def init_user_clicked(self, user_index, user_clicked_newindex, user_embedding):
        path_nodes = []
        path_relations = []
        act_probs_steps = []
        step_rewards = []
        q_values_steps = []
        # 选择用户点击新闻
        user_clicked_score = self.cal_user_clicked_score(user_index, user_clicked_newindex)
        input_clicked, input_relations = self.get_user_clicked_batch(torch.flatten(user_clicked_newindex.unsqueeze(1).repeat(1, 5, 1), 0, 1))
        action_embedding = self.trans_news_embedding(input_clicked) # bz * 5, news_entity_num, dim
        relation_embedding = self.relation_embedding(input_relations)  # bz * 5, news_entity_num, dim
        action_input = action_embedding + relation_embedding
        state_input = self.get_state_input(user_embedding, path_nodes, path_relations)
        act_probs, q_values = self.policy_net(state_input, action_input)
        act_probs, q_values, step_nodes, step_relations = self.get_path_nodes(act_probs, q_values, input_clicked,input_relations, topk=1)
        path_nodes.append(step_nodes)
        path_relations.append(step_relations)
        act_probs_steps.append(act_probs)  # [[bz*5, 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
        q_values_steps.append(q_values)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
        step_reward = self.get_reward(user_embedding, path_nodes, user_clicked_score)  # bz, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
        step_rewards.append(step_reward)
        return path_nodes, path_relations, act_probs_steps, step_rewards, q_values_steps, step_nodes

    def forward(self, user_index, candidate_newindex, user_clicked_newindex):
        depth = 0
        score = 0
        
        user_index = user_index.to(self.device)
        candidate_newindex = candidate_newindex.to(self.device)
        user_clicked_newindex = user_clicked_newindex[:, :50].to(self.device)

        user_clicked_score = self.cal_user_clicked_score(user_index, user_clicked_newindex)
        candidate_newindex = torch.flatten(candidate_newindex, 0, 1)
        news_embedding = self.trans_news_embedding(candidate_newindex)  # bz * 5, news_dim
        user_embedding = self.tanh(self.user_embedding( torch.flatten(user_index.unsqueeze(1).repeat(1, self.args.sample_size, 1), 0, 1))).squeeze()
        path_nodes, path_relations, act_probs_steps, step_rewards, q_values_steps, step_nodes = self.init_user_clicked(user_index,
                                                                                                                       user_clicked_newindex,
                                                                                                                       user_embedding)

        input_entities, input_relations = self.get_news_entities_batch(step_nodes)  # bz * 5, news_entity_num
        action_embedding = self.entity_embedding(input_entities)  # bz * 5, news_entity_num, dim
        relation_embedding = self.relation_embedding(input_relations)  # bz * 5, news_entity_num, dim
        action_input = action_embedding + relation_embedding  # bz * 5, news_entity_num, dim
        action_id = input_entities  # bz * 5, news_entity_num
        relation_id = input_relations  # bz * 5, news_entity_num
        state_input = self.get_state_input(user_embedding, path_nodes, path_relations)  # bz, 2 * news_dim
        topk = 1
        while (depth <= self.MAX_DEPTH):
            if depth < self.MAX_DEPTH:
                depth += 1
                act_probs, q_values = self.policy_net(state_input,  action_input)  # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
                act_probs, q_values, step_nodes, step_relations = self.get_path_nodes(act_probs, q_values, action_id, relation_id, topk)
                path_nodes.append(step_nodes)
                path_relations.append(step_relations)
                state_input = self.get_state_input(news_embedding, path_nodes, path_relations)  # bz*5, dim # bz*5, dim # bz*5, dim
                act_probs_steps.append(act_probs)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
                q_values_steps.append(q_values)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
                actionid_lookup, action_lookup, action_rid_lookup, action_r_lookup = self.get_next_action(step_nodes)
                action_id = actionid_lookup  # bz*5, 1，20  # bz*5, 1，20 # bz*5, 1，20
                relation_id = action_rid_lookup  # bz*5, 1，20  # bz*5, 1，20 # bz*5, 1，20
                action_input = self.entity_embedding(action_id) + self.relation_embedding( relation_id)  # bz*5, 1, 20, dim # bz*5, 1, 20, dim, # bz*5, 1, 20, dim
                step_reward = self.get_reward(user_embedding, path_nodes, user_clicked_score)  # bz, 1 # bz, 1 # bz, 1
                step_rewards.append(step_reward)  # [[bz*5, 1], [bz*5, 1], [bz*5, 1]]
            else:
                depth += 1
                state_input = self.get_state_input(news_embedding, path_nodes, path_relations)
                path_last_entityid = path_nodes[-1]
                action_id, relations_id = self.get_entities_news_batch(path_last_entityid)
                action_input = self.trans_news_embedding(action_id) + self.relation_embedding(relations_id)
                act_probs, q_values = self.policy_net(state_input, action_input)
                act_probs, q_values, predict_news, step_relations = self.get_path_nodes(act_probs, q_values, action_id, relations_id, topk)
                path_nodes.append(predict_news)
                path_relations.append(step_relations)
                act_probs_steps.append(act_probs)
                q_values_steps.append(q_values)

                score = self.get_reward(user_embedding, path_nodes, user_clicked_score, done=True)
                step_rewards.append(score.tolist())

        rec_score = self.get_rec_score(user_embedding, news_embedding, path_nodes, path_relations, mode = "train")
        return act_probs_steps, q_values_steps, step_rewards, path_nodes, path_relations, rec_score

    def get_rec_score(self, user_embedding, news_embedding, path_nodes, path_relations, mode):
        path_embedding_list = []
        if mode == 'train':
            for i in range(len(path_nodes)):
                if i == len(path_nodes) - 1 or i == 0:
                    path_embedding = self.trans_news_embedding(path_nodes[i])
                else:
                    path_embedding = self.entity_embedding(path_nodes[i])
                path_embedding_list.append(path_embedding)
            path_embedding = torch.stack(path_embedding_list).sum(0).squeeze()
            path_relations_embedding = self.relation_embedding(torch.stack(path_relations)).sum(0).squeeze()
            user_rep = (path_embedding + user_embedding + path_relations_embedding) / 3
            rec_score = torch.sum((user_rep * news_embedding), dim = -1)
            return rec_score

        path_relation_embedding_list = []
        if mode == "test":
            for i in range(len(path_nodes)):
                if i == len(path_nodes) - 1 or i == 0:
                    path_embedding = self.trans_news_embedding(path_nodes[i])
                    path_relations_embedding = self.relation_embedding(path_relations[i])
                    temp = path_embedding.shape[1]
                    path_embedding = path_embedding.sum(1) / temp
                    path_relations_embedding = path_relations_embedding.sum(1) / temp
                else:
                    path_embedding = self.entity_embedding(path_nodes[i])
                    path_relations_embedding = self.relation_embedding(path_relations[i])
                    temp = path_embedding.shape[1]
                    path_embedding = path_embedding.sum(1) / temp
                    path_relations_embedding = path_relations_embedding.sum(1) / temp
                path_embedding_list.append(path_embedding)
                path_relation_embedding_list.append(path_relations_embedding)
            path_embedding = torch.stack(path_embedding_list).sum(0).squeeze()
            path_relation_embedding = torch.stack(path_relation_embedding_list).sum(0).squeeze()
            user_rep = (path_embedding + user_embedding + path_relation_embedding) / 3
            rec_score = torch.sum((user_rep * news_embedding), dim=-1)
            return rec_score


    def test(self, user_index, candidate_newindex, user_clicked_newindex):
        depth = 0

        user_index = user_index.to(self.device)
        candidate_newindex = candidate_newindex.to(self.device)
        user_clicked_newindex = user_clicked_newindex[ : , :50].to(self.device)

        # 选择用户点击新闻
        user_clicked_score = self.cal_user_clicked_score(user_index, user_clicked_newindex)
        candidate_newindex = torch.flatten(candidate_newindex, 0, 1)
        news_embedding = self.trans_news_embedding(candidate_newindex)  # bz * 5, news_dim
        user_embedding = self.tanh(self.user_embedding( torch.flatten(user_index.unsqueeze(1).repeat(1, self.args.sample_size, 1), 0, 1))).squeeze()
        path_nodes, path_relations, _, _, _, step_nodes = self.init_user_clicked(user_index, user_clicked_newindex, user_embedding)

        # 查找点击新闻
        input_entities, input_relations = self.get_news_entities_batch(step_nodes)  # bz * 5, news_entity_num
        action_embedding = self.entity_embedding(input_entities)  # bz * 5, news_entity_num, dim
        relation_embedding = self.relation_embedding(input_relations)  # bz * 5, news_entity_num, dim
        action_input = action_embedding + relation_embedding  # bz * 5, news_entity_num, dim
        action_id = input_entities  # bz * 5, news_entity_num
        relation_id = input_relations  # bz * 5, news_entity_num
        state_input = self.get_state_input(user_embedding, path_nodes, path_relations)  # bz, 2 * news_dim
        self.Topk = [20, 5, 1]

        while (depth <= self.MAX_DEPTH):
            if depth < self.MAX_DEPTH:
                # 计算Q值和输出动作的可能性
                act_probs, q_values = self.policy_net(state_input, action_input)  # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
                topk = self.Topk[depth]
                # 根据动作的可能性随机选择K跳深度个动作
                act_probs, q_values, step_nodes, step_relations = self.get_path_nodes(act_probs, q_values, action_id, relation_id, topk)
                path_nodes.append(step_nodes)
                path_relations.append(step_relations)
                state_input = self.get_state_input(news_embedding, path_nodes, path_relations)  # bz*5, dim # bz*5, dim # bz*5, dim
                actionid_lookup, action_lookup, action_rid_lookup, action_r_lookup = self.get_next_action(step_nodes)
                action_id = actionid_lookup  # bz*5, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
                relation_id = action_rid_lookup  # bz*5, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
                action_input = self.entity_embedding(action_id) + self.relation_embedding(relation_id)  # bz*5,d(1-hop),20,dim #bz*5,d(1-hop) * d(2-hop),20,dim #bz*5,d(1-hop)*d(2-hop),20,dim
                depth += 1
            else:
                path_last_entityid = path_nodes[-1]
                action_id, relations_id = self.get_entities_news_batch(path_last_entityid)
                predict_news = torch.flatten(action_id, -2, -1)
                predict_relation = torch.flatten(relations_id, -2, -1)
                path_nodes.append(predict_news)
                path_relations.append(predict_relation)
                depth += 1

        score, path_num = self.get_best_reward(candidate_newindex, user_embedding, path_nodes, user_clicked_score)
        best = torch.max(score, dim=-1)
        best_score = best.values
        best_path = best.indices
        depth += 1
        rec_score = self.get_rec_score(user_embedding, news_embedding, path_nodes, path_relations, mode="test")
        return path_num, path_nodes, path_relations, rec_score, best_path

