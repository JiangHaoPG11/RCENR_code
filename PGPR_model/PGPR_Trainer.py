import torch
import torch.nn.functional as F
from AnchorKG_model.AnchorKG import *
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *


class Trainer():
    def __init__(self, args, Agent_model, optimizer_agent, data):
        self.args = args
        self.Agent_model = Agent_model
        self.optimizer_agent = optimizer_agent

        self.save_period = 100
        self.vaild_period = 40
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.news_embedding = data[3]
        self.entity_dict = data[6]
        self.entity_embedding = data[12]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]


    def get_batch_reward(self, step_reward):
        num_steps = len(step_reward)
        batch_rewards = None
        for i in range(num_steps):
            temp = torch.FloatTensor(step_reward[i]).unsqueeze(1)
            if i == 0:
                batch_rewards = temp
            else:
                batch_rewards = torch.cat([batch_rewards, temp], dim = -1)
        # batch_rewards = batch_rewards.to(self.device)
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += 0.99 * batch_rewards[:, num_steps - i]
        new_batch_rewards = []
        for i in range(batch_rewards.shape[1]):
            new_batch_rewards.append(batch_rewards[: , i])
        return new_batch_rewards

    def cal_auc(self, score, label):
        rec_loss = F.cross_entropy(score.cpu(), torch.argmax(label, dim=1))
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        return rec_loss, rec_auc

    def optimize_agent(self, batch_rewards, q_values_steps, act_probs_steps, rec_loss, all_loss):
        all_loss_list = []
        all_actor_loss = []
        all_critic_loss = []
        for i in range(len(batch_rewards)):
            batch_reward = batch_rewards[i]
            q_values_step = q_values_steps[i]
            act_probs_step = act_probs_steps[i]
            critic_loss, actor_loss = self.Agent_model.step_update(act_probs_step, q_values_step, batch_reward)
            #print(critic_loss)
            #print(actor_loss)
            all_actor_loss.append(actor_loss.cpu().mean())
            all_critic_loss.append(critic_loss.cpu().mean())
            all_loss_list.append(actor_loss.cpu().mean())
            all_loss_list.append(critic_loss.cpu().mean())

        #print('------')
        #print(actor_loss)
        #print(critic_loss)
        print(rec_loss)
        #print('------')

        all_loss_list.append(rec_loss)
        self.optimizer_agent.zero_grad()
        if all_loss_list != []:
            loss = torch.stack(all_loss_list).sum()  # sum up all the loss
            loss.backward()
            self.optimizer_agent.step()
            all_loss = all_loss + loss.data
        return loss

    def _train_epoch(self):
        self.Agent_model.train()
        all_loss_list = []

        auc_list = []
        all_loss = 0

        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newindex, user_index, user_clicked_newindex, label = data
            act_probs_steps, q_values_steps, step_rewards, path_node, path_relation, score = self.Agent_model(user_index, candidate_newindex, user_clicked_newindex)
            score = score.view(self.args.batch_size, -1)
            # print(score)
            rec_loss, rec_auc = self.cal_auc(score, label)
            batch_rewards = self.get_batch_reward(step_rewards)
            agent_loss = self.optimize_agent(batch_rewards, q_values_steps, act_probs_steps, rec_loss, all_loss)
            all_loss = agent_loss
            all_loss_list.append(all_loss.cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)
            torch.cuda.empty_cache()

        pbar.close()
        return mean(all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.Agent_model.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newindex, user_index, user_clicked_newindex, label = data
                act_probs_steps, q_values_steps, step_rewards, anchor_graph, anchor_relation1, score = self.Agent_model(user_index, candidate_newindex, user_clicked_newindex)
                score = score.view(self.args.batch_size, -1)
                rec_loss, rec_auc = self.cal_auc(score, label)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_agent = self.Agent_model.state_dict()
        filename_news_anchor = self.args.checkpoint_dir + ('checkpoint-news-anchor-epoch{}.pth'.format(epoch))
        torch.save(state_agent, filename_news_anchor)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            loss, auc= self._train_epoch()
            print("epoch：{}--- loss:{}------auc:{}------".format(epoch, str(loss), str(auc)))
            if epoch % self.vaild_period == 0:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch：{}---vaild auc：{} ".format(epoch, str(rec_auc)))
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
            predict_path = False
            if predict_path:
                path_nodes = []
                user_anchor_graph_nodes = []
                for data in self.train_dataloader:
                    candidate_newindex, user_index, user_clicked_newindex, _ = data
                    path_nodes.extend(self.Agent_model.get_path_list(self.Agent_model(user_index, candidate_newindex)[3], self.args.batch_size))
                    fp_anchor_file = open("./PGPR_out/cand_anchor_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        fp_anchor_file.write(candidate_newindex[i] + '\t' + ' '.join(list(set(path_nodes[i]))) + '\n')
        self._save_checkpoint('final')

    def test(self):
        pbar = tqdm(total= self.testdata_size)
        self.Agent_model.eval()
        pred_label_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newindex, user_index, user_clicked_newindex = data
                path_nodes, path_relations, best_score, best_path = self.Agent_model.test(user_index, candidate_newindex, user_clicked_newindex)
                best_score = best_score.view(self.args.batch_size, -1)
                pred_label_list.extend(best_score.cpu().numpy())
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()
        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))



