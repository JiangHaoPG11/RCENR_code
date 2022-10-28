import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class Net(torch.nn.Module):
    def __init__(self, entity_dict, news_title_embedding, entity_embedding, device):
        super(Net, self).__init__()
        self.device = device
        self.news_title_embedding = news_title_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.entity_dict = entity_dict

        self.actor_l1 = nn.Linear(100 * 3, 100)
        self.actor_l2 = nn.Linear(100, 100)
        self.actor_l3 = nn.Linear(100, 1)

        self.critic_l1 = nn.Linear(100 * 3, 100)
        self.critic_l2 = nn.Linear(100, 100)
        self.critic_l3 = nn.Linear(100, 1)

        self.elu = torch.nn.ELU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=0)

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.news_title_embedding[newsid]).to(self.device))
        return torch.stack(news_embeddings)

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

class AnchorKG(torch.nn.Module):

    def __init__(self, args, news_entity_dict, entity_news_dict, news_title_embedding, entity_embedding,
                 relation_embedding, entity_adj, relation_adj, kg_env, entity_dict, neibor_embedding, neibor_num, device):
        super(AnchorKG, self).__init__()
        self.args = args
        self.device = device
        self.news_entity_dict = news_entity_dict
        self.entity_news_dict = entity_news_dict
        self.news_title_embedding = news_title_embedding

        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.MAX_DEPTH = 3
        self.entity_dict = entity_dict
        self.neibor_embedding = nn.Embedding.from_pretrained(neibor_embedding)
        self.neibor_num = neibor_num

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=-1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_compress_1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress_2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.innews_relation = nn.Embedding(1, self.args.embedding_size)

        self.anchor_embedding_layer = nn.Linear(self.args.embedding_size, self.args.embedding_size)#todo *2 diyige
        self.anchor_weighs1_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.anchor_weighs1_layer2 = nn.Linear(self.args.embedding_size, 1)

        self.policy_net = Net(self.entity_dict, self.news_title_embedding, entity_embedding, self.device)
        self.target_net = Net(self.entity_dict, self.news_title_embedding, entity_embedding, self.device)

    def get_neiborhood_news_embedding_batch(self, news_embedding, entityids):
        neibor_news_embedding_avg = self.neibor_embedding(entityids)
        neibor_num = []
        for i in range(len(entityids)):
            neibor_num.append(torch.index_select(self.neibor_num, 0, entityids[i]))
        neibor_num = torch.stack(neibor_num)
        if len(neibor_news_embedding_avg.shape) > len(news_embedding.shape):
            news_embedding = torch.unsqueeze(news_embedding, 1)
            neibor_num = torch.unsqueeze(neibor_num, 2)
            news_embedding = news_embedding.expand(news_embedding.shape[0] ,neibor_news_embedding_avg.shape[1] ,news_embedding.shape[2])
            neibor_num = neibor_num.expand(neibor_num.shape[0], neibor_num.shape[1],news_embedding.shape[2])
        neibor_news_embedding_avg = torch.div((neibor_news_embedding_avg - news_embedding), neibor_num)
        return neibor_news_embedding_avg

    def get_anchor_graph_list(self, anchor_graph_nodes, batch_size):
        anchor_graph_list = []
        for i in range(batch_size):
            anchor_graph_list.append([])
        for i in range(len(anchor_graph_nodes)):
            for j in range(len(anchor_graph_nodes[i])):
                anchor_graph_list[j].extend(list(map(lambda x:str(x), anchor_graph_nodes[i][j].data.cpu().numpy())))
        return anchor_graph_list

    def get_sim_reward_batch(self, news_embedding_batch, neibor_news_embedding_avg_batch):
        if len(neibor_news_embedding_avg_batch.shape) > len(news_embedding_batch.shape):
            news_embedding_batch = torch.unsqueeze(news_embedding_batch, 1)
            news_embedding_batch = news_embedding_batch.expand(news_embedding_batch.shape[0],neibor_news_embedding_avg_batch.shape[1] ,news_embedding_batch.shape[2])
        cos_rewards = self.cos(news_embedding_batch, neibor_news_embedding_avg_batch)
        return cos_rewards

    def get_hit_rewards_batch(self, newsid_batch, state_id_input_batch):
        hit_rewards = []
        hit_rewards_weak = []
        for i in range(len(newsid_batch)):
            hit_rewards.append([])
            hit_rewards_weak.append([])
            for j in range(len(state_id_input_batch[i])):
                #if int(state_id_input_batch[i][j].data.cpu().numpy()) in self.entity_news_dict and newsid_batch[i] in self.hit_dict:
                if int(state_id_input_batch[i][j].data.cpu().numpy()) in self.entity_news_dict:
                    entity_neibor = set(self.entity_news_dict[int(state_id_input_batch[i][j].data.cpu().numpy())]).discard(newsid_batch[i])
                    # news_hit_neibor = self.hit_dict[newsid_batch[i]]
                    if entity_neibor != None:
                        # if len(entity_neibor & news_hit_neibor)>0:
                        if len(entity_neibor) > 0:
                            hit_rewards[-1].append(1.0)
                        else:
                            hit_rewards[-1].append(0.0)
                    else:
                        hit_rewards[-1].append(0.0)
                else:
                    hit_rewards[-1].append(0.0)
        return torch.FloatTensor(hit_rewards).to(self.device)

    def get_reward(self, newsid, news_embedding, state_input):
        entity_value = state_input
        neibor_news_embedding_avg = self.get_neiborhood_news_embedding_batch(news_embedding, entity_value)
        sim_reward = self.get_sim_reward_batch(news_embedding, neibor_news_embedding_avg)
        hit_reward = self.get_hit_rewards_batch(newsid, entity_value)
        reward =0.5*hit_reward + (1-0.5)*sim_reward
        return reward

    def get_news_entities_batch(self, newsids):
        news_entities = []
        news_relations = []
        for i in range(len(newsids)):
            news_entities.append(self.news_entity_dict[int(newsids[i])])
            news_relations.append([0 for k in range(20)])
        news_entities = torch.tensor(news_entities).to(self.device)
        news_relations = torch.tensor(news_relations).to(self.device)# bz, news_entity_num
        return news_entities, news_relations

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.news_title_embedding[newsid]).to(self.device))
        return torch.stack(news_embeddings)

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
                    next_action_id[-1].append([0 for k in range(20)])
                    next_action_r_id[-1].append([0 for k in range(20)])
        next_action_space_id = torch.LongTensor(next_action_id) # bz, d(k-hop) , 20
        next_state_space_embedding = self.entity_embedding(next_action_space_id)  # bz, d(k-hop) , 20 dim
        next_action_r_id = torch.LongTensor(next_action_r_id)# bz, d(k-hop) , 20, 20
        next_action_r = self.relation_embedding(next_action_r_id) # bz, d(k-hop) , 20,  dim
        return next_action_space_id, next_state_space_embedding, next_action_r_id, next_action_r

    def step_update(self, act_probs_step, q_values_step, step_reward, recommend_reward, reasoning_reward, alpha1=0.9, alpha2 = 0.1):
        recommend_reward = torch.unsqueeze(recommend_reward, dim=0)
        recommend_reward = torch.unsqueeze(recommend_reward, dim=1)
        reasoning_reward = torch.unsqueeze(reasoning_reward, dim=0)
        reasoning_reward = torch.unsqueeze(reasoning_reward, dim=1)
        recommend_reward = recommend_reward.expand(step_reward.shape[0], step_reward.shape[1])
        reasoning_reward = reasoning_reward.expand(step_reward.shape[0], step_reward.shape[1])
        curr_reward = alpha2 * step_reward + (1-alpha2)*(alpha1*recommend_reward + (1-alpha1)*reasoning_reward)
        advantage = curr_reward - q_values_step #curr_reward - q_values_step
        actor_loss = -act_probs_step * advantage.detach() #self.get_actor_loss(act_probs_step, q_values_step)#self.get_actor_loss(torch.log(act_probs_step), advantage) # -act_probs_step * advantage #problem2
        critic_loss = advantage.pow(2)#self.get_critic_loss(advantage) #advantage.pow(2)

        return critic_loss, actor_loss

    def get_anchor_nodes(self, weights, q_values, action_id_input, relation_id_input, topk):
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

    def get_news_embedding_input(self, entity_ids, news_embeddings):
        entity_ids_index = entity_ids._indices()[0]
        news_embedding_batch = news_embeddings(entity_ids_index)
        return news_embedding_batch

    def get_state_input(self, news_embedding, depth, anchor_graph, history_entity_1, history_relation_1):
        if depth == 0:
            state_embedding = torch.cat(
                [news_embedding, torch.tensor(np.zeros((news_embedding.shape[0], 100))).float().to(self.device)], dim=-1)
        else:
            history_entity_embedding = self.entity_embedding(history_entity_1) #  bz,  d(k-hop-1)，dim
            history_relation_embedding = self.relation_embedding(history_relation_1) #  bz,  d(k-hop-1), dim
            state_embedding_new = history_relation_embedding + history_entity_embedding #  bz,  d(k-hop-1), dim
            state_embedding_new = torch.mean(state_embedding_new, dim=1, keepdim=False)
            state_embedding = torch.cat([news_embedding, state_embedding_new], dim=-1) #  bz,  d(k-hop-1), dim
        return state_embedding

    def predict_anchor(self, newsid, news_feature):
        self.news_entity_dict[newsid] = news_feature[0]
        self.news_title_embedding = news_feature[1]
        prediction = self.forward([newsid], [newsid]).cpu().data.numpy()
        anchor_nodes = prediction[6]
        anchor_relation = prediction[8]
        return anchor_nodes, anchor_relation

    def forward(self, candidate_newindex):
        candidate_newindex = torch.flatten(candidate_newindex, 0, 1)
        depth = 0
        history_entity_1 = []
        history_relation_1 = []

        anchor_graph1 = []
        anchor_relation1 = []
        act_probs_steps1 = []
        step_rewards1 = []
        q_values_steps1 = []

        news_embedding = self.tanh(self.news_compress_2(self.elu(self.news_compress_1(self.get_news_embedding_batch(candidate_newindex))))) # bz * 5, news_dim
        news_embedding_origin = self.get_news_embedding_batch(candidate_newindex)  # bz * 5, news_dim_origin
        input_entities, input_relations = self.get_news_entities_batch(candidate_newindex) # bz * 5, news_entity_num
        action_embedding = self.entity_embedding(input_entities) # bz * 5, news_entity_num, dim
        relation_embedding = self.relation_embedding(input_relations) # bz * 5, news_entity_num, dim
        action_embedding = action_embedding + relation_embedding # bz * 5, news_entity_num, dim
        action_id = input_entities # bz * 5, news_entity_num
        relation_id = input_relations # bz * 5, news_entity_num
        state_input = self.get_state_input(news_embedding, depth, anchor_graph1, history_entity_1, history_relation_1) # bz, 2 * news_dim

        while (depth < self.MAX_DEPTH):
            topk = self.args.depth[depth]
            # 计算Q值和输出动作的可能性
            act_probs, q_values = self.policy_net(state_input, action_embedding) # bz*5,entity_num,1; bz*5,entity_num,1 #bz,d(1-hop),20,1;bz*5,d(1-hop),20,1
            # 根据动作的可能性随机选择K跳深度个动作
            anchor_act_probs, anchor_q_values, anchor_nodes, anchor_relations = self.get_anchor_nodes(act_probs, q_values, action_id, relation_id, topk)
            history_entity_1 = anchor_nodes #bz*5,d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            history_relation_1 = anchor_relations  #bz*5,d(1-hop) # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            depth = depth + 1
            state_input = self.get_state_input(news_embedding, depth, anchor_graph1, history_entity_1, history_relation_1) # bz*5, dim # bz*5, dim # bz*5, dim
            # 得到动作可能性
            act_probs_steps1.append(anchor_act_probs) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            q_values_steps1.append(anchor_q_values) #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            actionid_lookup, action_lookup, action_rid_lookup, action_r_lookup = self.get_next_action(anchor_nodes)
            action_id = actionid_lookup      # bz*5, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            relation_id = action_rid_lookup  # bz*5, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            action_embedding = self.entity_embedding(action_id) + self.relation_embedding(relation_id) #bz*5,d(1-hop),20,dim #bz*5,d(1-hop) * d(2-hop),20,dim #bz*5,d(1-hop)*d(2-hop),20,dim
            # 得到锚图节点
            anchor_graph1.append(anchor_nodes) # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            # 得到锚图关系
            anchor_relation1.append(anchor_relations)  # [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
            step_reward = self.get_reward(candidate_newindex, news_embedding_origin, anchor_nodes)  # bz, d(1-hop)  # bz*5, d(1-hop) * d(2-hop) # bz*5, d(1-hop) * d(2-hop) * d(3-hop)
            step_rewards1.append(step_reward)  #  [[bz*5, d(1-hop) * 1], [bz*5, d(1-hop) * d(2-hop)], [bz*5, d(1-hop) * d(2-hop) * d(3-hop)]]
        return act_probs_steps1, q_values_steps1, step_rewards1, anchor_graph1, anchor_relation1

