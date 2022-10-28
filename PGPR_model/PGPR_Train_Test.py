from PGPR_model.PGPR_Trainer import Trainer
from PGPR_model.PGPR import PGPR
import torch

class PGPR_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        Agent_model = PGPR(args, news_entity_dict, entity_news_dict, news_title_embedding, entity_embedding,
                           relation_embedding, entity_adj, relation_adj, entity_dict, device).to(device)
        optimizer_agent = torch.optim.Adam(Agent_model.parameters(), lr=0.0001)
        self.trainer = Trainer(args, Agent_model, optimizer_agent, data)

    def Train(self):
        print('training begining ...')
        self.trainer.train()

    def Test(self):
        self.trainer.test()

