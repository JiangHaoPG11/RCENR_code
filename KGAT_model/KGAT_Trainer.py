import torch
import torch.nn.functional as F
from AnchorKG_model.AnchorKG import *
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *


class Trainer():
    def __init__(self, args, KGAT_model, optimizer_KGAT, data):
        self.args = args
        self.KGAT_model = KGAT_model
        self.optimizer_KGAT = optimizer_KGAT
        self.save_period = 100
        self.vaild_period = 40
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]

    def cal_auc(self, score, label):
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        return rec_auc

    def optimize_KGAT(self, loss,):
        self.optimizer_KGAT.zero_grad()
        loss.backward()
        self.optimizer_KGAT.step()

    def _train_epoch(self):
        self.KGAT_model.train()
        rec_all_loss_list = []
        auc_list = []
        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newindex, user_index, user_clicked_newindex, label = data
            loss, scores, rec_loss, emb_loss = self.KGAT_model(user_index, candidate_newindex, label)
            self.optimize_KGAT(loss)
            rec_auc = self.cal_auc(scores, label)
            rec_all_loss_list.append(rec_loss.cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)
        pbar.close()
        return mean(rec_all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.KGAT_model.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newindex, user_index, user_clicked_newindex, label = data
                loss, scores, rec_loss, emb_loss = self.KGAT_model(user_index, candidate_newindex, label)
                rec_auc = self.cal_auc(scores, label)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_agent = self.KGAT_model.state_dict()
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
        self._save_checkpoint('final')

    def test(self):
        pbar = tqdm(total= self.testdata_size)
        self.KGAT_model.eval()
        pred_label_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newindex, user_index, user_clicked_newindex = data
                scores = self.KGAT_model.test(user_index, candidate_newindex)
                pred_label_list.extend(scores.cpu().numpy())
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()
        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))



