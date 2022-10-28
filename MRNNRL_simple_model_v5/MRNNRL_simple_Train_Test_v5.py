from MRNNRL_simple_model_v5.MRNNRL_simple_v5 import MRNNRL_simple_v5
from MRNNRL_simple_model_v5.Recommender_simple_v5 import Recommender_simple_v5
from MRNNRL_simple_model_v5.Reasoner_simple_v5 import Reasoner_simple_v5
from MRNNRL_simple_model_v5.MRNNRL_simple_Trainer_v5 import Trainer
import torch

class MRNNRL_simple_Train_Test_v5():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        subgraph_model = MRNNRL_simple_v5(args, news_entity_dict, entity_news_dict, user_click_dict,
                                news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict,
                                news_title_embedding, entity_embedding, relation_embedding,
                                entity_adj, relation_adj, neibor_embedding, neibor_num, device).to(device)

        model_recommender = Recommender_simple_v5(args, news_title_embedding, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                                        news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device).to(device)
        model_reasoner = Reasoner_simple_v5(args, entity_embedding, relation_embedding, news_title_embedding, device).to(device)
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
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

