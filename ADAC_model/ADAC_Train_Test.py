from ADAC_model.ADAC_Trainer import Trainer
from ADAC_model.ADAC import ADAC
import torch
import numpy as np

class ADAC_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        demo_paths_index, demo_type_index, demo_relations_index = self.load_demo_path()
        ADAC_model = ADAC(args, news_entity_dict, entity_news_dict, user_click_dict, news_title_embedding, entity_embedding,
                          relation_embedding, entity_adj, relation_adj, entity_dict,
                          demo_paths_index, demo_type_index, demo_relations_index, device).to(device)
        optimizer_agent = torch.optim.Adam(ADAC_model.parameters(), lr=0.0001)
        self.trainer = Trainer(args, ADAC_model, optimizer_agent, data, device)

    def Train(self):
        print('training begining ...')
        # AnchorKG_model_bak.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

    def load_demo_path(self):
        adac_path_index = np.load('./Data/pathdata/adac_path_index.npy')
        adac_type_index = np.load('./Data/pathdata/adac_type_index.npy')
        adac_relations_index = np.load('./Data/pathdata/adac_relations_index.npy')
        demo_paths_index = torch.transpose(torch.IntTensor(adac_path_index), 1, 0).squeeze() # total_size, clicked_num, path_long
        demo_type_index = torch.transpose(torch.IntTensor(adac_type_index), 1, 0).squeeze() # total_size, clicked_num, path_long
        demo_relations_index = torch.transpose(torch.IntTensor(adac_relations_index), 1, 0).squeeze() # total_size, clicked_num, path_long

        return demo_paths_index, demo_type_index, demo_relations_index
