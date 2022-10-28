import os
import pandas as pd
import torch.nn.functional as F
from AnchorKG_model.AnchorKG import *
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *


class Trainer():
    def __init__(self, args, subgraph_model, model_recommender, model_reasoner,
                 optimizer_subgraph, optimizer_recommender, optimizer_reasoner, data):
        self.args = args
        self.subgraph_model = subgraph_model
        self.model_recommender = model_recommender
        self.model_reasoner = model_reasoner
        self.optimizer_subgraph = optimizer_subgraph
        self.optimizer_recommender = optimizer_recommender
        self.optimizer_reasoner = optimizer_reasoner
        self.save_period = 1
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

        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #self.device = torch.device("cpu")

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
            # print(torch.mean(batch_rewards[num_steps1 - i], dim=-1).shape)
            # print(batch_rewards[num_steps1 - i - 1].shape)
            batch_rewards[num_steps1 - i - 1] = batch_rewards[num_steps1 - i - 1] + 0.1 * torch.mean(batch_rewards[num_steps1 - i], dim=-1)

        if len(self.args.depth) == 3:
            batch_rewards[1] = torch.reshape(batch_rewards[1],
                                             (batch_rewards[1].shape[0], self.args.depth[0] * self.args.depth[1]))
            batch_rewards[2] = torch.reshape(batch_rewards[2],
                                             (batch_rewards[2].shape[0], self.args.depth[0] * self.args.depth[1] * self.args.depth[2]))
        else:
            print("error, layer num not match")
        return batch_rewards

    def optimize_recommender(self, rec_score, label):
        # rec_loss = F.cross_entropy(rec_score, torch.argmax(label, dim=1))
        rec_loss = self.criterion(rec_score, torch.argmax(label, dim=1).to(self.device))
        rec_loss.requires_grad_(True)
        rec_loss_mean = torch.mean(rec_loss)
        #print(rec_loss_mean)

        self.optimizer_recommender.zero_grad()
        rec_loss_mean.backward(retain_graph=True)
        self.optimizer_recommender.step()
        return rec_loss

    def optimize_reasoner(self, rea_score, overlap_score, label):
        # rea_loss = self.criterion(rea_score, torch.argmax(label, dim=1))
        rea_loss = self.criterion(rea_score, torch.argmax(label, dim=1).to(self.device))
        overlap_loss = self.criterion(overlap_score, torch.argmax(label, dim=1).to(self.device))
        rea_loss.requires_grad_(True)
        rea_loss_mean = torch.mean(rea_loss)

        self.optimizer_reasoner.zero_grad()
        rea_loss_mean.backward(retain_graph=True)
        # rea_all_loss.backward(retain_graph=True)
        self.optimizer_reasoner.step()
        return rea_loss, overlap_loss

    def cal_auc(self, label, rea_score, rec_score):
        score = 0.5 * rea_score + 0.5 * rec_score
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return auc

    def optimize_subgraph(self,
                          batch_rewards1, q_values_steps1, act_probs_steps1,
                          batch_rewards2, q_values_steps2, act_probs_steps2,
                          rec_loss, reasoning_loss):
        all_loss_list = []
        actor_loss_list = []
        critic_loss_list = []
        news1_actor_loss = []
        news1_critic_loss = []
        for i in range(3):
            batch_reward1 = batch_rewards1[i]
            q_values_step1 = q_values_steps1[i]
            act_probs_step1 = act_probs_steps1[i]
            critic_loss1, actor_loss1 = self.subgraph_model.step_update(act_probs_step1, q_values_step1,
                                                                      batch_reward1,
                                                                      1 - rec_loss.detach(),
                                                                      1 - reasoning_loss.detach(),
                                                                      0.9,
                                                                      0.1)
            news1_actor_loss.append(actor_loss1.mean())
            news1_critic_loss.append(critic_loss1.mean())
            actor_loss_list.append(actor_loss1.mean())
            critic_loss_list.append(critic_loss1.mean())
            all_loss_list.append(actor_loss1.mean())
            all_loss_list.append(critic_loss1.mean())

        news2_actor_loss = []
        news2_critic_loss = []
        for i in range(3):
            batch_reward2 = batch_rewards2[i]
            q_values_step2 = q_values_steps2[i]
            act_probs_step2 = act_probs_steps2[i]
            critic_loss2, actor_loss2 = self.subgraph_model.step_update(act_probs_step2, q_values_step2,
                                                                        batch_reward2,
                                                                        1 - rec_loss.detach(),
                                                                        1 - reasoning_loss.detach(),
                                                                        0.9,
                                                                        0.1)
            news2_actor_loss.append(actor_loss2.mean())
            news2_critic_loss.append(critic_loss2.mean())
            actor_loss_list.append(actor_loss2.mean())
            critic_loss_list.append(critic_loss2.mean())
            all_loss_list.append(actor_loss2.mean())
            all_loss_list.append(critic_loss2.mean())

        self.optimizer_subgraph.zero_grad()
        if all_loss_list != []:
            loss = torch.stack(all_loss_list).sum()  # sum up all the loss
            loss.backward()
            self.optimizer_subgraph.step()
            #anchor_all_loss = anchor_all_loss + loss.data
        return loss


    def _train_epoch(self, epoch):
        self.subgraph_model.train()
        self.model_recommender.train()
        self.model_reasoner.train()
        subgraph_all_loss_list = []

        rec_all_loss_list = []
        rea_all_loss_list = []
        auc_list = []
        overlap_all_loss_list = []
        subgraph_all_loss = 0
        rea_all_loss = 0
        rec_all_loss = 0
        overlap_all_loss = 0

        pbar = tqdm(total=self.traindata_size,  desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
        for data in self.train_dataloader:
            candidate_newindex, user_index, user_clicked_newindex, label = data
            news_act_probs_steps, news_q_values_steps, news_step_rewards, news_graph, news_graph_relation, news_graph_type, \
            user_act_probs_steps, user_q_values_steps, user_step_rewards, user_graph, user_graph_relation, user_graph_type  = self.subgraph_model( user_index, user_clicked_newindex, candidate_newindex) # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]

            rea_score, overlap_score = self.model_reasoner(candidate_newindex, user_index, news_graph, user_graph,
                                                           news_graph_relation, user_graph_relation,
                                                           news_graph_type, user_graph_type)[:2]
            rea_loss, overlap_loss = self.optimize_reasoner(rea_score, overlap_score, label.to(self.device))

            rec_score = self.model_recommender(candidate_newindex, user_index, user_clicked_newindex, news_graph, user_graph, news_graph_type, user_graph_type)
            rec_loss = self.optimize_recommender(rec_score, label.to(self.device))

            rec_auc = self.cal_auc(label, rea_score, rec_score)
            news_batch_rewards = self.get_batch_reward(news_step_rewards)
            user_batch_rewards = self.get_batch_reward(user_step_rewards)
            subgraph_loss = self.optimize_subgraph(news_batch_rewards, news_q_values_steps, news_act_probs_steps,
                                                     user_batch_rewards, user_q_values_steps, user_act_probs_steps,
                                                     rec_loss, rea_loss)
            #print(subgraph_loss)
            subgraph_all_loss = subgraph_all_loss + subgraph_loss.data
            rea_all_loss = rea_all_loss + rea_loss.data
            rec_all_loss = rec_all_loss + rec_loss.data
            overlap_all_loss = overlap_all_loss + overlap_loss.data

            subgraph_all_loss_list.append(subgraph_loss.cpu().item())
            rec_all_loss_list.append(torch.mean(rec_loss).cpu().item())
            rea_all_loss_list.append(torch.mean(rea_loss).cpu().item())
            overlap_all_loss_list.append(torch.mean(overlap_loss).cpu().item())
            auc_list.append(rec_auc)

            pbar.update(self.args.batch_size)
            torch.cuda.empty_cache()

        pbar.close()
        return mean(subgraph_all_loss_list), mean(rec_all_loss_list), mean(rea_all_loss_list), mean(auc_list), mean(overlap_all_loss_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.subgraph_model.eval()
        self.model_reasoner.eval()
        self.model_recommender.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newindex, user_index, user_clicked_newindex, label = data
                news_act_probs_steps, news_q_values_steps, news_step_rewards, news_graph, news_graph_relation, news_graph_type, \
                user_act_probs_steps, user_q_values_steps, user_step_rewards, user_graph, user_graph_relation, user_graph_type = self.subgraph_model(user_index, user_clicked_newindex,candidate_newindex)  # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
                rea_score = self.model_reasoner(candidate_newindex, user_index, news_graph, user_graph, news_graph_relation, user_graph_relation)[0]
                rec_score = self.model_recommender(candidate_newindex, user_index, user_clicked_newindex, news_graph, user_graph)
                score = 0.5 * rea_score + 0.5 * rec_score * rec_score
                pbar.update(self.args.batch_size)
                try:
                    rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
                except ValueError:
                    rec_auc = 0.5
                rec_auc_list.append(rec_auc)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_subgraph_model = self.subgraph_model.state_dict()
        state_recommender = self.model_recommender.state_dict()
        state_reasoner = self.model_reasoner.state_dict()
        filename_subgraph = self.args.checkpoint_dir + ('checkpoint-subgraph-epoch{}.pth'.format(epoch))
        torch.save(state_subgraph_model, filename_subgraph)
        filename_recommender = self.args.checkpoint_dir + ('checkpoint-recommender-epoch{}.pth'.format(epoch))
        torch.save(state_recommender, filename_recommender)
        filename_reasoner = self.args.checkpoint_dir + ('checkpoint-reasoner-epoch{}.pth'.format(epoch))
        torch.save(state_reasoner, filename_reasoner)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            subgraph_loss, rec_loss, rea_loss, rec_auc, overlap_loss = self._train_epoch(epoch)
            print("epoch：{}---subgraph loss:{}---recommend loss：{}---reason loss：{}----overlap loss：{}---rec auc：{} ".
                  format(epoch, str(subgraph_loss),  str(rec_loss), str(rea_loss), str(overlap_loss), str(rec_auc)))

            if epoch % self.vaild_period == 10:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch：{}---vaild auc：{} ".format(epoch, str(rec_auc)))

            if epoch % self.save_period == 60:
                self._save_checkpoint(epoch)

            predict_graph = False
            if predict_graph:
                news_graph_nodes = []
                user_graph_nodes = []
                for data in self.train_dataloader:
                    candidate_newindex, user_index, user_clicked_newindex, _ = data
                    news_graph_nodes.extend(self.subgraph_model.get_subgraph_list(self.subgraph_model(user_index, user_clicked_newindex, candidate_newindex)[3], self.args.batch_size))
                    user_graph_nodes.extend(self.subgraph_model.get_anchor_graph_list(self.subgraph_model(user_index, user_clicked_newindex, candidate_newindex)[8],self.args.batch_size))
                    news_graph_file = open("./MRNNRL_out/news_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        news_graph_file.write(candidate_newindex[i] + '\t' + ' '.join(list(set(news_graph_nodes[i]))) + '\n')
                    user_graph_file = open("./MRNNRL_out/user_file_" + str(epoch) + ".tsv", 'w', encoding='utf-8')
                    for i in range(self.args.batch_size):
                        user_graph_file.write(user_clicked_newindex['item1'][i] + '\t' + ' '.join(list(user_graph_nodes[i])) + '\n')

        self._save_checkpoint('final')

    def every_nth(self, lst, nth):
        new_lst = []
        for i in range(int(len(lst)/nth)):
            print(lst[i * 5])
            new_lst.append(lst[i * 5])
        return new_lst

    def test(self):
        print('start testing...')
        pbar = tqdm(total= self.testdata_size)
        self.subgraph_model.eval()
        self.model_reasoner.eval()
        self.model_recommender.eval()
        pred_label_list = []
        user_index_list = []
        candidate_newindex_list = []
        pred_reasoning_paths_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newindex, user_index, user_clicked_newindex = data
                news_act_probs_steps, news_q_values_steps, news_step_rewards, news_graph, news_graph_relation, news_graph_type, \
                user_act_probs_steps, user_q_values_steps, user_step_rewards, user_graph, user_graph_relation, user_graph_type = self.subgraph_model(user_index, user_clicked_newindex,candidate_newindex)  # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
                rea_score, _, reasoning_paths, _, _ = self.model_reasoner(candidate_newindex, user_index, news_graph, user_graph,
                                                                          news_graph_relation, user_graph_relation,
                                                                          news_graph_type, user_graph_type)
                rec_score = self.model_recommender(candidate_newindex, user_index, user_clicked_newindex,
                                                   news_graph, user_graph,
                                                   news_graph_type, user_graph_type)
                score = 0.5 * rea_score + 0.5 * rec_score * rec_score
                pred_label_list.extend(score.cpu().numpy())
                user_index_list.extend(user_index.cpu().numpy())
                candidate_newindex_list.extend(candidate_newindex.cpu().numpy())
                pred_reasoning_paths_list.extend(reasoning_paths)
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()

        # 存储预测结果
        folder_path = '../predict/MRNNR/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['candidate_news'] = candidate_newindex_list
        predict_df['path'] = self.every_nth(pred_reasoning_paths_list, 5)
        predict_df['score'] = pred_label_list[:, 0]
        predict_df.to_csv('MRNNRL_predict.csv', index = False)

        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))



