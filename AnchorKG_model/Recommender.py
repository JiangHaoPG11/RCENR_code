import torch
import torch.nn as nn
import torch.nn.functional as F

class Recommender(torch.nn.Module):

    def __init__(self, args, news_title_embedding, entity_embedding, relation_embedding, entity_adj, relation_adj, device):
        super(Recommender, self).__init__()
        self.args = args
        self.device = device
        self.news_title_embedding = news_title_embedding
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj

        self.elu = nn.ELU(inplace=False)
        self.mlp_layer1 = nn.Linear(self.args.embedding_size * 2, self.args.embedding_size)
        self.mlp_layer2 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.news_compress1 = nn.Linear(self.args.title_size, self.args.embedding_size)
        self.news_compress2 = nn.Linear(self.args.embedding_size, self.args.embedding_size,)
        self.anchor_embedding_layer = nn.Linear(self.args.embedding_size * 2, self.args.embedding_size)
        self.anchor_weights_layer1 = nn.Linear(self.args.embedding_size, self.args.embedding_size)
        self.anchor_weights_layer2 = nn.Linear(self.args.embedding_size, 1)
        self.tanh = nn.Tanh()

    def get_news_embedding_batch(self, newsids):
        news_embeddings = []
        for newsid in newsids:
            news_embeddings.append(torch.FloatTensor(self.news_title_embedding[newsid]).to(self.device))
        return torch.stack(news_embeddings)

    def get_neighbors(self, entities):
        neighbor_entities = []
        neighbor_relations = []
        for entity_batch in entities: # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)
            neighbor_entities.append([])
            neighbor_relations.append([])
            for entity in entity_batch:
                if entity not in self.entity_adj.keys():
                    neighbor_entities[-1].append(self.entity_adj[0])
                    neighbor_relations[-1].append(self.relation_adj[0])
                else:
                    if type(entity) == int:
                        neighbor_entities[-1].append(self.entity_adj[entity])
                        neighbor_relations[-1].append(self.relation_adj[entity])
                    else:
                        neighbor_entities[-1].append([])
                        neighbor_relations[-1].append([])
                        for entity_i in entity:
                            neighbor_entities[-1][-1].append(self.entity_adj[entity_i])
                            neighbor_relations[-1][-1].append(self.relation_adj[entity_i])

        return torch.tensor(neighbor_entities).to(self.device), torch.tensor(neighbor_relations).to(self.device)

    def get_anchor_graph_embedding(self, anchor_graph): # [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        anchor_graph_nodes = []
        for i in range(len(anchor_graph[1])): # bz
            anchor_graph_nodes.append([])
            for j in range(len(anchor_graph)): # k-hop
                anchor_graph_nodes[-1].extend(anchor_graph[j][i].tolist()) # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop)

        anchor_graph_nodes_embedding = self.entity_embedding(torch.tensor(anchor_graph_nodes)) # bz, d(1-hop) +  d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), dim
        neibor_entities, neibor_relations = self.get_neighbors(anchor_graph_nodes) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20
        neibor_entities_embedding = self.entity_embedding(neibor_entities) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim
        neibor_relations_embedding = self.relation_embedding(neibor_relations) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 20, dim
        anchor_embedding = torch.cat([anchor_graph_nodes_embedding, torch.sum(neibor_entities_embedding+neibor_relations_embedding, dim=-2)], dim=-1) # bz, d(1-hop) + d(1-hop) * d(2-hop) + d(1-hop) * d(2-hop) * d(3-hop), 2*dim
        anchor_embedding = self.tanh(self.anchor_embedding_layer(anchor_embedding))
        anchor_embedding_weight = F.softmax(self.anchor_weights_layer2(self.elu(self.anchor_weights_layer1(anchor_embedding))), dim = -2)
        anchor_embedding = torch.sum(anchor_embedding * anchor_embedding_weight, dim=-2) # bz, dim (ut in equation)
        return anchor_embedding

    def forward(self, cand_news, clicked_news, cand_anchor_graph1, clicked_anchor_graph2):  # bz, 1; bz, 1; [[bz, d(1-hop) * 1], [bz, d(1-hop) * d(2-hop)], [bz, d(1-hop) * d(2-hop) * d(3-hop)]]
        cand_news = torch.flatten(cand_news, 0, 1)
        clicked_news = torch.flatten(clicked_news, 0, 1)
        cand_news_embedding1 = self.get_news_embedding_batch(cand_news)  # bz, news_dim
        clicked_news_embedding2 = self.get_news_embedding_batch(clicked_news)  # bz, news_dim
        cand_news_embedding1 = self.tanh(self.news_compress2(self.elu(self.news_compress1(cand_news_embedding1))))  # bz, news_dim
        clicked_news_embedding2 = self.tanh(self.news_compress2(self.elu(self.news_compress1(clicked_news_embedding2))))  # bz, news_dim
        anchor_embedding1 = self.get_anchor_graph_embedding(cand_anchor_graph1) # bz, news_dim
        anchor_embedding2 = self.get_anchor_graph_embedding(clicked_anchor_graph2) # bz, news_dim
        cand_news_embedding1 = torch.cat([cand_news_embedding1, anchor_embedding1], dim=-1) # bz, 2 * news_dim
        clicked_news_embedding2 = torch.cat([clicked_news_embedding2, anchor_embedding2], dim=-1) # bz, 2 * news_dim
        cand_news_embedding1 = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(cand_news_embedding1))))
        clicked_news_embedding2 = self.elu(self.mlp_layer2(self.elu(self.mlp_layer1(clicked_news_embedding2))))
        news_embedding = torch.reshape(cand_news_embedding1, (self.args.batch_size, self.args.sample_size, -1))
        user_embedding = torch.mean(torch.reshape(clicked_news_embedding2, (self.args.batch_size, self.args.user_clicked_num, -1)), dim=1)
        user_embedding = user_embedding.unsqueeze(1).repeat(1, 5, 1)
        score = torch.sum(news_embedding * user_embedding, dim = -1)
        return score
