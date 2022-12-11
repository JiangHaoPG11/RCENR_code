import torch
import torch.nn.functional as F
from AnchorKG_model.AnchorKG import *
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *


class Trainer():
    def __init__(self, args, KPRN_model, optimizer_kprn, data):
        self.args = args
        self.KPRN_model = KPRN_model
        self.optimizer_kprn = optimizer_kprn
        self.save_period = 100
        self.vaild_period = 100
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]
 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def optimize_kprn(self, rec_score, label):
        rec_loss = F.cross_entropy(rec_score, torch.argmax(label, dim=1).to(self.device))
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), F.softmax(rec_score.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        self.optimizer_kprn.zero_grad()
        rec_loss.backward()
        #for name, param in self.KPRN_model.named_parameters():
        #    print('%14s : %s' % (name, param.grad))
        self.optimizer_kprn.step()
        return rec_loss, rec_auc

    def cal_auc(self, score, label):
        rec_loss = F.cross_entropy(score, torch.argmax(label, dim=1).to(self.device))
        try:
            rec_auc = roc_auc_score(label.cpu().numpy(), score.cpu().detach().numpy())
        except ValueError:
            rec_auc = 0.5
        return rec_loss, rec_auc

    def _train_epoch(self):
        self.KPRN_model.train()
        rec_all_loss_list = []
        auc_list = []
        pbar = tqdm(total=self.traindata_size)
        for data in self.train_dataloader:
            candidate_newindex, user_index, _, label = data
            score = self.KPRN_model(candidate_newindex, user_index)
            rec_loss, rec_auc = self.optimize_kprn(score, label)
            rec_all_loss_list.append(rec_loss.cpu().item())
            auc_list.append(rec_auc)
            pbar.update(self.args.batch_size)
            torch.cuda.empty_cache()
        pbar.close()
        return mean(rec_all_loss_list), mean(auc_list)

    def _vaild_epoch(self):
        pbar = tqdm(total=self.vailddata_size)
        self.KPRN_model.eval()
        rec_auc_list = []
        with no_grad():
            for data in self.vaild_dataloader:
                candidate_newindex, user_index, _, label = data
                score = self.KPRN_model(candidate_newindex, user_index)
                _, rec_auc = self.cal_auc(score, label)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_agent = self.KPRN_model.state_dict()
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
        self.KPRN_model.eval()
        pred_label_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newindex, user_index, _ = data
                score = self.KPRN_model(candidate_newindex, user_index)
                pred_label_list.extend(score.cpu().numpy())
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()
        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))



