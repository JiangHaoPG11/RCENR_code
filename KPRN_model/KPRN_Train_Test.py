import numpy as np

from KPRN_model.KPRN import KPRN
from KPRN_model.KPRN_Trainer import Trainer
import torch

class KPRN_Train_Test():
    def __init__(self, args, data, device):
        # train_dataloader, test_dataloader, vaild_dataloader, \
        # new_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, \
        # neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, len(vaild_data), len(train_data), len(test_data), label_test, bound_test
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        total_paths_index, total_relations_index, total_type_index = self.load_total_path()
        kprn = KPRN(args, news_title_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, news_entity_dict, entity_news_dict,
                    total_paths_index, total_relations_index, total_type_index).to(device)
        optimizer_kprn = torch.optim.Adam(kprn.parameters(), lr = 0.01)
        self.trainer = Trainer(args, kprn, optimizer_kprn, data)

    def Train(self):
        print('training begining ...')
        self.trainer.train()

    def Test(self):
        print('testing begining ...')
        self.trainer.test()

    def load_total_path(self):
        kprn_path_index = np.load('./Data/pathdata/kprn_path_index.npy')
        kprn_type_index = np.load('./Data/pathdata/kprn_type_index.npy')
        kprn_relations_index = np.load('./Data/pathdata/kprn_relations_index.npy')
        total_paths_index = torch.transpose(torch.IntTensor(kprn_path_index), 1, 0) # total_size, sample_size, max_path, path_long
        total_type_index = torch.transpose(torch.IntTensor(kprn_type_index), 1, 0) # total_size, sample_size, max_path, path_long
        total_relations_index = torch.transpose(torch.IntTensor(kprn_relations_index), 1, 0) # total_size, sample_size, max_path, path_long
        return total_paths_index, total_relations_index, total_type_index
