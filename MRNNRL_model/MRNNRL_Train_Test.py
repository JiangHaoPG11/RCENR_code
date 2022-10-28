from MRNNRL_model.MRNNRL import MRNNRL
from MRNNRL_model.Recommender import Recommender
from MRNNRL_model.Reasoner import Reasoner
from MRNNRL_model.MRNNRL_Trainer import Trainer
import torch

class MRNNRL_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        subgraph_model = MRNNRL(args, news_entity_dict, entity_news_dict, user_click_dict,
                                news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict,
                                news_title_embedding, entity_embedding, relation_embedding,
                                entity_adj, relation_adj, neibor_embedding, neibor_num, device).to(device)

        model_recommender = Recommender(args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                                        news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device).to(device)
        model_reasoner = Reasoner(args, kg_env, entity_embedding, relation_embedding, news_title_embedding, device).to(device)

        optimizer_subgraph = torch.optim.Adam(subgraph_model.parameters(), lr=0.1 * 0.00002)
        optimizer_recommender = torch.optim.Adam(model_recommender.parameters(), lr=0.001)
        optimizer_reasoner = torch.optim.Adam(model_reasoner.parameters(), lr=0.001)

        self.trainer = Trainer(args, subgraph_model,  model_recommender, model_reasoner,
                               optimizer_subgraph, optimizer_recommender, optimizer_reasoner, data)

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

