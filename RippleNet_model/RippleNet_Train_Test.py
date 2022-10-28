from RippleNet_model.RippleNet import RippleNet
from RippleNet_model.RippleNet_Trainer import Trainer
import torch

class RippleNet_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        ripplenet = RippleNet(args, news_title_embedding, entity_embedding, relation_embedding, ripple_set).to(device)
        optimizer_ripplenet = torch.optim.Adam(ripplenet.parameters(), lr= 0.0005)
        self.trainer = Trainer(args, ripplenet, optimizer_ripplenet, data)

    def Train(self):
        print('training begining ...')
        self.trainer.train()

    def Test(self):
        print('testing begining ...')
        self.trainer.test()
