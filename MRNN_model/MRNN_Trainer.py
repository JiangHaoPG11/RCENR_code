import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *
import torch
import torch.nn as nn

# train_dataloader, test_dataloader, vaild_dataloader, \
# news_title_embedding, entity_adj, relation_adj, entity_dict, \
# kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
# news_title_word_index, news_category_index, news_subcategory_index, \
# category_news_dict, subcategory_news_dict, word_embedding, \
# neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
# vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

class Trainer():
    def __init__(self, args, MRNN_model, optimizer_MRNN, data):
        self.args = args
        self.MRNN_model = MRNN_model
        self.optimizer_MRNN = optimizer_MRNN
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

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #self.device = torch.device("cpu")

    def cal_auc(self, label, rec_score):
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(rec_score.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return auc

    def optimize_model(self, rec_score, label):
        rec_loss = self.criterion(rec_score, torch.argmax(label, dim=1).to(self.device))
        self.optimizer_MRNN.zero_grad()
        rec_loss.backward()
        #for name, param in self.model_recommender.named_parameters():
        #    print('%14s : %s' % (name, param.grad))
        self.optimizer_MRNN.step()
        return rec_loss

    def _train_epoch(self, epoch):
        self.MRNN_model.train()

        rec_all_loss_list = []
        auc_list = []
        rec_all_loss = 0

        pbar = tqdm(total=self.traindata_size,  desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
        for data in self.train_dataloader:
            candidate_newindex, user_index, user_clicked_newindex, label = data
            rec_score = self.MRNN_model(candidate_newindex, user_clicked_newindex)
            rec_loss = self.optimize_model(rec_score, label.to(self.device))
            rec_auc = self.cal_auc(label, rec_score)
            rec_all_loss = rec_all_loss + rec_loss.data

            rec_all_loss_list.append(torch.mean(rec_loss).cpu().item())
            auc_list.append(rec_auc)

            pbar.update(self.args.batch_size)
            print("----recommend loss：{}-----rec auc：{} ".
                  format(str(torch.mean(rec_loss).cpu().item()), str(rec_auc)))
           # torch.cuda.empty_cache()

        pbar.close()
        return  mean(rec_all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        self.MRNN_model.eval()
        rec_auc_list = []
        with no_grad():
            pbar = tqdm(total=self.vailddata_size)
            for data in self.vaild_dataloader:
                candidate_newindex, user_index, user_clicked_newindex, label = data
                rec_score = self.optimize_model(candidate_newindex, user_clicked_newindex)
                rec_auc = self.cal_auc(label, rec_score)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_MRNN_model = self.MRNN_model.state_dict()
        filename_MRNN = self.args.checkpoint_dir + ('checkpoint-subgraph-epoch{}.pth'.format(epoch))
        torch.save(state_MRNN_model, filename_MRNN)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            rec_loss, rec_auc = self._train_epoch(epoch)
            print("epoch：{}----recommend loss：{}------rec auc：{} ".
                  format(epoch, str(rec_loss), str(rec_auc)))

            if epoch % self.vaild_period == 10:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch：{}---vaild auc：{} ".format(epoch, str(rec_auc)))

            if epoch % self.save_period == 60:
                self._save_checkpoint(epoch)
        self._save_checkpoint('final')

    def test(self):
        print('start testing...')
        pbar = tqdm(total= self.testdata_size)
        self.MRNN_model.eval()
        pred_label_list = []
        user_index_list = []
        candidate_newindex_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newindex, user_index, user_clicked_newindex = data
                rec_score = self.MRNN_model(candidate_newindex, user_clicked_newindex)
                score = rec_score
                pred_label_list.extend(score.cpu().numpy())
                user_index_list.extend(user_index.cpu().numpy())
                candidate_newindex_list.extend(candidate_newindex.cpu().numpy()[:,0])
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()
        # 存储预测结果
        folder_path = '../predict/MRNN/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['candidate_news'] = candidate_newindex_list
        predict_df['score'] = pred_label_list[:, 0]
        predict_df['label'] = self.label_test[:len(user_index_list)]
        predict_df.to_csv('MRNNRL_simplev5_predict.csv', index = False)
        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))



