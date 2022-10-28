from utils.TransE_pytoch import *
import pandas as pd
from tqdm import tqdm
news_entity_index_df = pd.read_csv('../Data/df/entity_index_df.csv')
news_entity_embedding = np.load('../Data/metadata/new_entity_embedding.npy')

### 获取KG
def get_KG_construct(news_entity_index_df):
    '''
    构建知识图谱
    :return: dict(kg)：kg的键是头部实体，kg的值是尾部实体和关系
    '''
    print('constructing adjacency matrix ...')
    # 加载图
    news_entity_id_list = news_entity_index_df['id'].tolist()
    graph_file_fp = open('../wikidata-graph/wikidata-graph.tsv', 'r', encoding='utf-8')
    graph = []
    entity_id_list = news_entity_id_list
    # relation_id_list = ['mention', 'clicked'] #加入了新闻->实体的关系"mention"和用户->新闻的关系'clicked'。
    # entity_id_list = []
    relation_id_list = []
    index = 0
    for line in graph_file_fp:
        index += 1
        linesplit = line.split('\n')[0].split('\t')
        # 保证只有新闻连接的实体加入到图中
        if len(linesplit) > 1:
            graph.append([linesplit[0], linesplit[1], linesplit[2]])
        if index > 100000:
            break
    print('三元组个数：{}'.format(len(graph)))
    kg = {}
    index = 0
    for triple in graph:
        index += 1
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return graph, kg

def get_onehop_neighbor( kg, news_entity_index_df):
    '''
    得到一跳邻居实体
    :param kg: 知识图谱字典
    :param news_entity_index_df: 实体id和index对应df
    :return: List(neigh_entity_adj): 对应新闻实体的邻居实体id列表
             List(neigh_relation_adj)：对应新闻实体的邻居实体id列表
             entity_index_df ：添加邻居实体之后的df
    '''
    print('get one_hop neighbor entities and select new graph......')
    entity_id_list = news_entity_index_df['id'].values.tolist()
    total_entity_list = []
    new_graph = []
    total_relation_list = []
    loop = tqdm(total = len(entity_id_list))
    index1 = 1
    for entity_id in entity_id_list:
        index1 += 1
        if entity_id in kg.keys():
            for index in range(len(kg[entity_id])):
                # print(index)
                total_entity_list.append(kg[entity_id][index][0])
                total_relation_list.append(kg[entity_id][index][1])
                new_graph.append([entity_id, kg[entity_id][index][1], kg[entity_id][index][0]])
        loop.update(1)
    loop.close()
    total_entity_list = entity_id_list + total_entity_list

    return new_graph, total_entity_list, total_relation_list

# def select_graph(graph, news_entity_index_df, total_entity_list):
#     print('select one hop graph......')
#     news_entity_id_list = news_entity_index_df['id'].tolist()
#     total_entity = news_entity_id_list + total_entity_list
#     entity_id_list = news_entity_id_list
#     relation_id_list = []
#     new_graph = []
#     for triplet in graph:
#         if (triplet[0] in total_entity) or (triplet[2] in total_entity):
#             new_graph.append([triplet[0], triplet[1], triplet[2]])
#             entity_id_list.append(triplet[0])
#             entity_id_list.append(triplet[2])
#             relation_id_list.append(triplet[1])
#     return new_graph, entity_id_list, relation_id_list


def map_graph_to_index(graph, entity_id_list, relation_id_list):
    print('map entity to index......')
    news_entity_id_list = news_entity_index_df['id'].tolist()
    news_entity_num = len(news_entity_id_list)
    # 映射实体
    entity_dict = {}
    index = 0
    for entity_id in entity_id_list:
        if entity_id not in entity_dict.keys():
            entity_dict[entity_id] = index
            index += 1
    entity_index_list = list(entity_dict.values())
    entityid2index = pd.DataFrame(pd.Series(entity_dict))
    entityid2index = entityid2index.reset_index()
    entityid2index.columns = ['entity_id', 'index']
    entityid2index.to_csv('../Data/KG/entityid2index.csv')
    print('总实体数：{} 新闻实体数：{}'.format(len(entity_dict), news_entity_num))
    # 映射关系
    relation_dict = {}
    index = 2
    relation_dict['mention'] = 0
    relation_dict['clicked'] = 1
    for relation_id in relation_id_list:
        if relation_id not in relation_dict.keys():
            relation_dict[relation_id] = index
            index += 1
    relation_index_list = list(relation_dict.values())
    relationid2index = pd.DataFrame(pd.Series(relation_dict))
    relationid2index = relationid2index.reset_index()
    relationid2index.columns = ['relation_id', 'index']
    relationid2index.to_csv('../Data/KG/relationid2index.csv')
    print('总关系数：{} 新闻关系数：{}'.format(len(relation_dict), 2))
    # 映射三元组
    graph_index = []
    for graph_one in graph:
        h = graph_one[0]
        r = graph_one[1]
        t = graph_one[2]
        h_idnex = entity_dict[h]
        r_index = relation_dict[r]
        t_idnex = entity_dict[t]
        graph_index.append([h_idnex, r_index, t_idnex])
    graph = pd.DataFrame(graph_index)
    graph.columns = ['h_index', 'r_index', 't_idnex']
    graph.to_csv('../Data/KG/graph_index.csv')
    return entity_index_list,  relation_index_list, graph_index, news_entity_num

### 获取transE嵌入
def get_transE_embedding(entity_index_list, relation_index_list, graph_index, news_entity_num, news_entity_embedding):
    print('start transE embedding......')
    transE = TransE(entity_index_list, relation_index_list, graph_index, epochs=1, batch_size=4, embedding_dim=100, lr=0.01, margin=1.0, norm=2)
    transE.data_initialise()
    entities_embedding, relations_embedding = transE.training_run()
    entities_embedding = list(entities_embedding)
    relations_embedding = list(relations_embedding)
    # 融合
    entities_embedding_fuse = entities_embedding[news_entity_num:]
    news_entity_embedding = list(news_entity_embedding)
    entities_embedding = news_entity_embedding + entities_embedding_fuse
    clicked_embedding = [np.random.rand(100)]
    mention_embedding = [np.random.rand(100)]
    relations_embedding = mention_embedding + clicked_embedding + relations_embedding
    np.save("../Data/KG/TransE_entity_embedding", np.array(entities_embedding))
    np.save("../Data/KG/TransE_relation_embedding", np.array(relations_embedding))

if __name__ == "__main__":
    graph, kg = get_KG_construct(news_entity_index_df)
    new_graph, total_entity_list, total_relation_list = get_onehop_neighbor(kg, news_entity_index_df)
    entity_index_list,  relation_index_list, graph_index, news_entity_num = map_graph_to_index(new_graph, total_entity_list, total_relation_list)
    get_transE_embedding (entity_index_list, relation_index_list, graph_index, news_entity_num, news_entity_embedding)

