from AnchorKG_model.AnchorKG import AnchorKG
from AnchorKG_model.Recommender import Recommender
from AnchorKG_model.Reasoner import Reasoner
from AnchorKG_model.AnchorKG_Trainer import Trainer
import torch

class AnchorKG_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader,  test_dataloader, vaild_dataloader, \
        new_title_embedding, entity_adj, relation_adj, entity_dict, \
        kg_env, news_entity_dict, entity_news_dict, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, vailddata_size, traindata_size, testdata_size, label_test, bound_test = data
        news_model_anchor = AnchorKG(args, news_entity_dict, entity_news_dict, new_title_embedding, entity_embedding,
                                     relation_embedding, entity_adj, relation_adj, kg_env, entity_dict,
                                     neibor_embedding, neibor_num, device).to(device)
        user_model_anchor = AnchorKG(args, news_entity_dict, entity_news_dict, new_title_embedding, entity_embedding,
                                     relation_embedding, entity_adj, relation_adj, kg_env, entity_dict,
                                     neibor_embedding, neibor_num, device).to(device)
        model_recommender = Recommender(args, new_title_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, device).to(device)
        model_reasoner = Reasoner(args, kg_env, entity_embedding, relation_embedding, device).to(device)

        optimizer_news_anchor = torch.optim.Adam(news_model_anchor.parameters(), lr=0.1 * 0.00002)
        optimizer_user_anchor = torch.optim.Adam(user_model_anchor.parameters(), lr=0.1 * 0.00002)
        optimizer_recommender = torch.optim.Adam(model_recommender.parameters(), lr=0.00001)
        optimizer_reasoner = torch.optim.Adam(model_reasoner.parameters(), lr=0.00001)

        self.trainer = Trainer(args, news_model_anchor, user_model_anchor, model_recommender, model_reasoner,
                               optimizer_news_anchor, optimizer_user_anchor, optimizer_recommender, optimizer_reasoner, data)

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

