from RC4ERec_model.RC4ERec import RC4ERec
from RC4ERec_model.Recommender import Recommender
from RC4ERec_model.Reasoner import Reasoner
from RC4ERec_model.RC4ERec_Trainer import Trainer
import torch

class RC4ERec_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        subgraph_model = RC4ERec(args, news_entity_dict, entity_news_dict, user_click_dict,
                                news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict,
                                news_title_embedding, entity_embedding, relation_embedding,
                                entity_adj, relation_adj, neibor_embedding, neibor_num, device).to(device)

        model_recommender = Recommender(args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                                        news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device).to(device)
        model_reasoner = Reasoner(args, entity_embedding, relation_embedding, news_title_embedding, device).to(device)
        optimizer_subgraph = torch.optim.Adam(subgraph_model.parameters(), lr=0.000002, weight_decay=0.000001)

        optimizer_recommender = torch.optim.Adam([{'params': model_recommender.parameters(), 'lr': 0.0001}])
        for name, parameters in model_recommender.named_parameters():
            print(name, ':', parameters.size())
        optimizer_reasoner = torch.optim.Adam([{'params': model_reasoner.parameters(), 'lr': 0.0001}])
        for name, parameters in model_reasoner.named_parameters():
            print(name, ':', parameters.size())
        self.trainer = Trainer(args, subgraph_model,  model_recommender, model_reasoner,
                               optimizer_subgraph, optimizer_recommender, optimizer_reasoner, data)

    def Train(self):
        print('training begining ...')
        self.trainer.train()

    def Test(self):
        self.trainer.test()

