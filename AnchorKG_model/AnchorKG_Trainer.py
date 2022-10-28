import torch.nn.functional as F
from AnchorKG_model.AnchorKG import *
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *


class Trainer():
    def __init__(self, args, news_model_anchor, user_model_anchor, model_recommender, model_reasoner,
                 optimizer_news_anchor, optimizer_user_anchor, optimizer_recommender, optimizer_reasoner, data):
        self.args = args
        self.news_model_anchor = news_model_anchor
        self.user_model_anchor = user_model_anchor
        self.model_recommender = model_recommender
        self.model_reasoner = model_reasoner
        self.optimizer_news_anchor = optimizer_news_anchor
        self.optimizer_user_anchor = optimizer_user_anchor
        self.optimizer_recommender = optimizer_recommender
        self.optimizer_reasoner = optimizer_reasoner
        self.save_period = 100
        self.vaild_period = 1
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
        batch_rewards = step_reward  # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        num_steps1 = len(batch_rewards)
        if len(self.args.depth) == 3:
            # [bz, d(1-hop) * d(2-hop)]
            batch_rewards[1] = torch.reshape(batch_rewards[1], (
            batch_rewards[1].shape[0], self.args.depth[0], self.args.depth[1]))  # bz, d(1-hop) , d(2-hop)
            # [bz, d(1-hop) * d(2-hop) * d(3-hop)]
            batch_rewards[2] = torch.reshape(batch_rewards[2], (
            batch_rewards[2].shape[0], self.args.depth[0], self.args.depth[1], self.args.depth[2]))  # bz, d(1-hop), d(2-hop), d(3-hop)
        else:
            print("error, layer num not match")
        for i in range(1, num_steps1):
            batch_rewards[num_steps1 - i - 1] = batch_rewards[num_steps1 - i - 1] + 0.1 * torch.mean(
                batch_rewards[num_steps1 - i], dim=-1)
        if len(self.args.depth) == 3:
            batch_rewards[1] = torch.reshape(batch_rewards[1],(batch_rewards[1].shape[0], self.args.depth[0] * self.args.depth[1]))
            batch_rewards[2] = torch.reshape(batch_rewards[2],(batch_rewards[2].shape[0], self.args.depth[0] * self.args.depth[1] * self.args.depth[2]))
        else:
            print("error, layer num not match")
        return batch_rewards

    def optimize_recommender(self, rec_score, label):
        rec_loss = F.cross_entropy(rec_score, torch.argmax(label, dim=1))
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(rec_score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        self.optimizer_recommender.zero_grad()
        rec_loss.backward(retain_graph=True)
        self.optimizer_recommender.step()
        return rec_loss, rec_auc

    def optimize_reasoner(self, rea_score, label):
        rea_loss = F.cross_entropy(rea_score, torch.argmax(label, dim=1))
        rea_loss.requires_grad_(True)
        self.optimizer_reasoner.zero_grad()
        rea_loss.backward(retain_graph=True)
        self.optimizer_reasoner.step()
        return rea_loss

    def optimize_news_anchor(self, batch_rewards, q_values_steps, act_probs_steps, rec_loss, reasoning_loss, anchor_all_loss):
        all_loss_list = []
        actor_loss_list = []
        critic_loss_list = []
        news1_actor_loss = []
        news1_critic_loss = []
        for i in range(len(batch_rewards)):
            batch_reward = batch_rewards[i]
            q_values_step = q_values_steps[i]
            act_probs_step = act_probs_steps[i]
            critic_loss, actor_loss = self.news_model_anchor.step_update(act_probs_step, q_values_step, batch_reward,
                                                                         rec_loss.detach(),
                                                                         reasoning_loss.detach(),
                                                                         0.9,
                                                                         0.1)
            news1_actor_loss.append(actor_loss.mean())
            news1_critic_loss.append(critic_loss.mean())
            actor_loss_list.append(actor_loss.mean())
            critic_loss_list.append(critic_loss.mean())
            all_loss_list.append(actor_loss.mean())
            all_loss_list.append(critic_loss.mean())
        self.optimizer_news_anchor.zero_grad()
        if all_loss_list != []:
            loss = torch.stack(all_loss_list).sum()  # sum up all the loss
            loss.backward()
            self.optimizer_news_anchor.step()
            anchor_all_loss = anchor_all_loss + loss.data
        return anchor_all_loss

    def optimize_user_anchor(self, batch_rewards, q_values_steps, act_probs_steps, rec_loss, reasoning_loss, anchor_all_loss):
        all_loss_list = []
        actor_loss_list = []
        critic_loss_list = []
        news1_actor_loss = []
        news1_critic_loss = []
        for i in range(len(batch_rewards)):
            batch_reward = batch_rewards[i]
            q_values_step = q_values_steps[i]
            act_probs_step = act_probs_steps[i]
            critic_loss, actor_loss = self.user_model_anchor.step_update(act_probs_step, q_values_step, batch_reward,
                                                                         rec_loss.detach(),
                                                                         reasoning_loss.detach(),
                                                                         0.9,
                                                                         0.1)
            news1_actor_loss.append(actor_loss.mean())
            news1_critic_loss.append(critic_loss.mean())
            actor_loss_list.append(actor_loss.mean())
            critic_loss_list.append(critic_loss.mean())
            all_loss_list.append(actor_loss.mean())
            all_loss_list.append(critic_loss.mean())
        self.optimizer_user_anchor.zero_grad()
        if all_loss_list != []:
            loss = torch.stack(all_loss_list).sum()  # sum up all the loss
            loss.backward()
            self.optimizer_user_anchor.step()
            anchor_all_loss = anchor_all_loss + loss.data
        return anchor_all_loss

    def _train_epoch(self):
        self.news_model_anchor.train()
        self.user_model_anchor.train()
        self.model_recommender.train()
        self.model_reasoner.train()
        news_anchor_all_loss_list = []
        user_anchor_all_loss_list = []
        rec_all_loss_list = []
        rea_all_loss_list = []
        auc_list = []
        news_anchor_all_loss = 0
        user_anchor_all_loss = 0
        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newindex, user_index, user_clicked_newindex, label = data
            act_probs_steps1, q_values_steps1, step_rewards1, anchor_graph1, anchor_relation1 = self.news_model_anchor(candidate_newindex) # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
            act_probs_steps2, q_values_steps2, step_rewards2, anchor_graph2, anchor_relation2 = self.user_model_anchor(user_clicked_newindex)
            rea_score = self.model_reasoner(candidate_newindex, user_clicked_newindex, anchor_graph1, anchor_graph2, anchor_relation1, anchor_relation2)[0]
            rea_loss = self.optimize_reasoner(rea_score, label)
            rec_score = self.model_recommender(candidate_newindex, user_clicked_newindex, anchor_graph1, anchor_graph2)
            rec_loss, rec_auc = self.optimize_recommender(rec_score, label)
            batch_rewards1 = self.get_batch_reward(step_rewards1)
            batch_rewards2 = self.get_batch_reward(step_rewards2)
            news_anchor_loss = self.optimize_news_anchor(batch_rewards1, q_values_steps1, act_probs_steps1, rec_loss, rea_loss, news_anchor_all_loss)
            user_anchor_loss = self.optimize_user_anchor(batch_rewards2, q_values_steps2, act_probs_steps2, rec_loss, rea_loss, user_anchor_all_loss)

            news_anchor_all_loss += news_anchor_loss
            user_anchor_all_loss += user_anchor_loss


            news_anchor_all_loss_list.append(news_anchor_loss.cpu().item())
            user_anchor_all_loss_list.append(user_anchor_loss.cpu().item())
            rec_all_loss_list.append(rec_loss.cpu().item())
            rea_all_loss_list.append(rea_loss.cpu().item())
            auc_list.append(rec_auc)

            pbar.update(self.args.batch_size)
            torch.cuda.empty_cache()

        pbar.close()
        return mean(news_anchor_all_loss_list), mean(user_anchor_all_loss_list), mean(rec_all_loss_list), mean(rea_all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.news_model_anchor.eval()
        self.user_model_anchor.eval()
        self.model_recommender.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newindex, user_index, user_clicked_newindex, label = data
                act_probs_steps1, q_values_steps1, step_rewards1, anchor_graph1, anchor_relation1 = self.news_model_anchor(
                    candidate_newindex)  # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
                act_probs_steps2, q_values_steps2, step_rewards2, anchor_graph2, anchor_relation2 = self.user_model_anchor(
                    user_clicked_newindex)
                rec_score = self.model_recommender(candidate_newindex, user_clicked_newindex, anchor_graph1, anchor_graph2)
                pbar.update(self.args.batch_size)
                try:
                    rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(rec_score.cpu(), dim=1).detach().numpy())
                except ValueError:
                    rec_auc = 0.5
                rec_auc_list.append(rec_auc)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_news_anchor = self.news_model_anchor.state_dict()
        state_user_anchor = self.user_model_anchor.state_dict()
        state_recommender = self.model_recommender.state_dict()
        state_reasoner = self.model_reasoner.state_dict()
        filename_news_anchor = self.args.checkpoint_dir + ('checkpoint-news-anchor-epoch{}.pth'.format(epoch))
        torch.save(state_news_anchor, filename_news_anchor)
        filename_user_anchor = self.args.checkpoint_dir + ('checkpoint-user-anchor-epoch{}.pth'.format(epoch))
        torch.save(state_user_anchor, filename_user_anchor)
        filename_recommender = self.args.checkpoint_dir + ('checkpoint-recommender-epoch{}.pth'.format(epoch))
        torch.save(state_recommender, filename_recommender)
        filename_reasoner = self.args.checkpoint_dir + ('checkpoint-reasoner-epoch{}.pth'.format(epoch))
        torch.save(state_reasoner, filename_reasoner)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            news_anchor_loss, user_anchor_loss, rec_loss, rea_loss, rec_auc = self._train_epoch()

            print("epoch：{}---news anchor loss:{}--user anchor loss:{}---recommend loss：{}---reason loss：{}----rec auc：{} ".
                  format(epoch, str(news_anchor_loss), str(user_anchor_loss), str(rec_loss), str(rea_loss), str(rec_auc)))

            if epoch % self.vaild_period == 0:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch：{}---vaild auc：{} ".format(epoch, str(rec_auc)))

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)

            predict_anchor_graph = False
            if predict_anchor_graph:
                cand_anchor_graph_nodes = []
                user_anchor_graph_nodes = []
                for data in self.train_dataloader:
                    candidate_newindex, _, user_clicked_newindex, _ = data
                    cand_anchor_graph_nodes.extend(self.news_model_anchor.get_anchor_graph_list(self.news_model_anchor(candidate_newindex)[3], self.args.batch_size))
                    user_anchor_graph_nodes.extend(self.user_model_anchor.get_anchor_graph_list(self.user_model_anchor(user_clicked_newindex)[3],self.args.batch_size))
                    fp_anchor_file = open("./AnchorKG_out/cand_anchor_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        fp_anchor_file.write(candidate_newindex[i] + '\t' + ' '.join(list(set(cand_anchor_graph_nodes[i]))) + '\n')
                    fp_anchor_file = open("./AnchorKG_out/user_anchor_file2_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        fp_anchor_file.write(user_clicked_newindex['item1'][i] + '\t' + ' '.join(list(user_anchor_graph_nodes[i])) + '\n')

        self._save_checkpoint('final')

    def test(self):
        pbar = tqdm(total= self.testdata_size)
        self.news_model_anchor.eval()
        self.user_model_anchor.eval()
        self.model_recommender.eval()
        pred_label_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newindex, user_index, user_clicked_newindex = data
                act_probs_steps1, q_values_steps1, step_rewards1, anchor_graph1, anchor_relation1 = self.news_model_anchor(candidate_newindex)  # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
                act_probs_steps2, q_values_steps2, step_rewards2, anchor_graph2, anchor_relation2 = self.user_model_anchor(user_clicked_newindex)
                rec_score = self.model_recommender(candidate_newindex, user_clicked_newindex, anchor_graph1, anchor_graph2)
                pbar.update(self.args.batch_size)
                pred_label_list.extend(rec_score.cpu().numpy())

            pred_label_list = np.vstack(pred_label_list)
            pbar.close()
        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))



