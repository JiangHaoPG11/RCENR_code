import pandas as pd
import numpy as np
import random
from nltk import word_tokenize
from nltk.corpus import stopwords
# from transformers import BertTokenizer, BertModel
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#nltk.download('stopwords')
### 读取用户历史交互记录
def load_data_behaviors_New():
    '''
    读取用户行为信息，新闻信息和新闻实体向量
    :return: DataFrame(user_behaviors_df),DataFrame(new_Info_df),DataFrame(entity_embedding_df)
    '''
    user_behaviors_df = pd.read_table('../Data/MIND_small/MINDsmall_train/behaviors.tsv',
                                      header=None,
                                      names=['impression_id', 'user_id', 'time', 'history', 'impressions'],
                                      sep="\t", nrows = 100 )
    new_Info_df = pd.read_table('../Data/MIND_small/MINDsmall_train/news.tsv',
                                header=None,
                                names=['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                       'title_entities', 'abstract_entities'])
    entity_embedding_df = pd.read_table('../Data/MIND_small/MINDsmall_train/entity_embedding.vec', header=None)
    return user_behaviors_df, new_Info_df, entity_embedding_df

### 获取用户id和点击标签
def get_total_user_id_and_label(user_behaviors_df):
    '''
    针对用户印象日志，对于user_behaviors_df中的每一个用户提取正样本和负样本
    :param user_behaviors_df:
    :return: DataFrame(user_pnewid_nnew_id_df): 包含用户对应的正样本和负样本新闻id
             List(total_newid_list)：所有新闻id，用于构建index
    '''
    impress_log = user_behaviors_df['impressions'].values.tolist()
    user_list = user_behaviors_df['user_id'].values.tolist()
    user_id_list = []
    # 获得用户的正新闻和负新闻,用于负采样
    user_pnew = []
    user_nnew = []
    user_pnew_list = []
    user_nnew_list = []
    ## 拆解新闻ID，用于获得新闻的index
    total_newid_list = []
    ## 循环印象日志，得到正新闻id和负新闻id
    for i in range(len(impress_log)):
        line = impress_log[i]
        split = line.split(' ')
        for j in range(len(split)):
            temp = split[j]
            for z in range(len(temp)):
                if temp[z] != '-':
                    continue
                else:
                    break
            New_id = temp[0:z]
            Label = temp[-1]
            if Label == '1':
                user_pnew.append(New_id)
            if Label == '0':
                user_nnew.append(New_id)
            total_newid_list.append(New_id)
        user_pnew_list.append(user_pnew)
        user_nnew_list.append(user_nnew)
        user_id_list.append(user_list[i])
        # 清空
        user_pnew = []
        user_nnew = []
    user_pnewid_nnew_id_df = pd.DataFrame(data=None, columns= ['user_id', 'pnew_id', 'nnew_id'])
    user_pnewid_nnew_id_df['user_id'] = user_id_list
    user_pnewid_nnew_id_df['pnew_id'] = user_pnew_list
    user_pnewid_nnew_id_df['nnew_id'] = user_nnew_list
    # print('---------total_newid_list-------------')
    # print(total_newid_list)
    return user_pnewid_nnew_id_df, total_newid_list

### 获取用户index和点击新闻id
def get_user_index( user_behaviors_df, user_pnewid_nnew_id_df):
    '''
    构建用户字典，得到用户历史点击新闻id，并进行采样
    :param user_behaviors_df:
    :param user_pnewid_nnew_id_df:
    :return:  DataFrame(user_pnewid_nnew_id_df): 对应用户的正样本新闻和负样本新闻
              DataFrame(user_clicked_df)：对应用户的点击新闻
              List(user_clicked_newid_list)：用户点击新闻list
              Dict(user_dict):用户字典
    '''
    # 获取用户id，得到用户index
    user_id_list = user_pnewid_nnew_id_df['user_id'].values.tolist()
    user_index_list = []
    total_user_index_list = []
    temp = []
    user_clicked_newid_list = []
    # 创建用户字典
    user_dict = {}
    # 获取用户index和对应的点击新闻
    user_id_set_list = list(set(user_id_list))
    index = 0
    for i in range(len(user_id_set_list)):
        if (user_id_set_list[i] not in user_dict.keys()) == True :
            user_dict[user_id_set_list[i]] = index
            user_index_list.append(index)
            index += 1
        temp1 = user_behaviors_df[user_behaviors_df['user_id'] == user_id_set_list[i]]
        state = temp1['history'].values.tolist()[0]
        state = str(state)
        if state != 'nan':
            line = temp1['history'].values.tolist()[0]
        else:
            print(1)
        split = line.split(' ')
        for j in range(len(split)):
            temp.append(split[j])
        user_clicked_newid_list.append(temp)
        temp = []

    # 将映射字典存储在df中
    user_index_df = pd.DataFrame.from_dict(user_dict, orient='index', columns=['user_index'])
    user_index_df = user_index_df.reset_index()
    print('总共的用户数量{}'.format(user_index_df.shape[0]))
    # print(new_index_df)
    user_index_df.to_csv('../Data/df/user_index_df.csv', index=False)

    ## 获得用户点击新闻id和对应用户表示的df
    user_clicked_df = pd.DataFrame()
    user_clicked_df['user_id'] = user_id_set_list
    user_clicked_df['user_index'] = user_index_list
    user_clicked_df['user_clicked_newid'] = user_clicked_newid_list
    # print('--------------user_clicked_df-------------')
    # print(user_clicked_df)

    ## 映射用户index到预测df
    for i in range(len(user_id_list)):
        total_user_index_list.append(user_dict[user_id_list[i]])
    user_pnewid_nnew_id_df['user_index'] = total_user_index_list
    user_pnewid_nnew_id_df.to_csv('../Data/df/user_pnewid_nnew_id_df.csv')
    # print('--------------user_pnewid_nnew_id_df-------------')
    # print(user_pnewid_nnew_id_df)
    print('总的用户数{}'.format(len(user_index_list)))
    return user_pnewid_nnew_id_df, user_clicked_df, user_clicked_newid_list, user_dict

### 获取用户点击新闻id和候选新闻id映射得到index
def get_index_clicked_newid_cand_newid( user_clicked_newid_list, total_newid_list):
    '''
    合并用户点击新闻和正负样本新闻list，然后映射得到对应的新闻index
    :param user_clicked_newid_list:
    :param total_newid_list:
    :return: dict(new_dict): 新闻ID和新闻index对应字典
    '''
    temp1 = []
    temp2 = []
    for i in range(len(user_clicked_newid_list)):
        for item in user_clicked_newid_list[i]:
            temp1.append(item)
    for i in range(len(total_newid_list)):
        temp2.append(total_newid_list[i])
    new_id_list = temp1 + temp2

    # 获取候选新闻和用户点击新闻对应的映射字典
    new_dict = {}
    index = 0
    for new_id in new_id_list:
        if (new_id not in new_dict.keys()) == True:
            new_dict[new_id] = index
            index += 1
    new_dict['padding'] = index

    # 将映射字典存储在df中
    new_index_df = pd.DataFrame.from_dict(new_dict, orient='index', columns=['new_index'])
    new_index_df = new_index_df.reset_index()
    print('总共的新闻标题数量{}'.format(new_index_df.shape[0]))
    # print(new_index_df)
    new_index_df.to_csv('../Data/df/new_index_df.csv',index= False)
    return new_dict

### 获取测试集数据
def get_test_new_user(new_dict, user_pnewid_nnew_id_df):
    '''
    构建用于测试模型的数据：test_user_index，test_candidate_newindex，test_label，test_bound
    :param new_dict:
    :param user_pnewid_nnew_id_df:
    :return:
    '''
    pnewid_list = user_pnewid_nnew_id_df['pnew_id'].values.tolist()
    nnewid_list = user_pnewid_nnew_id_df['nnew_id'].values.tolist()
    userindex_list = user_pnewid_nnew_id_df['user_index'].values.tolist()

    newindex_list = []
    userindex = []
    label_list = []
    Bound = []
    index = 0
    for i in range(user_pnewid_nnew_id_df.shape[0]):
        start = index
        for pnewid in pnewid_list[i]:
            temp = [new_dict[pnewid], 0, 0, 0, 0]
            newindex_list.append(temp)
            userindex.append(userindex_list[i])
            label_list.append(1)
            index += 1
        for nnewid in nnewid_list[i]:
            temp = [new_dict[nnewid], 0, 0, 0, 0]
            newindex_list.append(temp)
            userindex.append(userindex_list[i])
            label_list.append(0)
            index += 1
        Bound.append([start,index])
    np.save('../Data/test/test_user_index.npy', np.array(userindex))
    np.save('../Data/test/test_candidate_newindex.npy', np.array(newindex_list))
    np.save('../Data/test/test_label.npy', np.array(label_list))
    np.save('../Data/test/test_bound.npy', np.array(Bound))

### 负采样
def negtivate_sample(user_pnewid_nnew_id_df, new_dict):
    '''
    进行负采样，采样印象日志当中出现但是用户没有点击的样本作为负样本
    :param user_pnewid_nnew_id_df:
    :param new_dict:
    :return: DataFrame(newindex_label_df)：用户对应的采样正负样本以及label
    '''
    ## 映射正样本新闻id
    pnewid_list = user_pnewid_nnew_id_df['pnew_id']
    pnew_index_list_one = []
    pnew_index_list = []
    for pnewid in pnewid_list:
        for pnew in pnewid:
            pnew_index = new_dict[pnew]
            pnew_index_list_one.append(pnew_index)
        pnew_index_list.append(pnew_index_list_one)
        pnew_index_list_one=[]

    ## 映射负样本新闻id
    nnew_index_list_one = []
    nnew_index_list = []
    nnewid_list = user_pnewid_nnew_id_df['nnew_id']
    for nnewid in nnewid_list:
        for nnew in nnewid:
            nnew_index = new_dict[nnew]
            nnew_index_list_one.append(nnew_index)
        nnew_index_list.append(nnew_index_list_one)
        nnew_index_list_one=[]

    ## 填充负样本
    for nnewindex in nnew_index_list:
        if len(nnewindex) < 4:
            need_len = 4 - len(nnewindex)
            for i in range(need_len):
                nnewindex.append(random.sample(nnewindex,1)[0])

    ## 选取用户
    user_index = []
    candidate_newindex_list = []
    label = [1, 0 , 0 , 0, 0]
    label_list = []

    user_index_list = user_pnewid_nnew_id_df['user_index'].values.tolist()
    for i in range(len(user_index_list)):
        user_index_temp = user_index_list[i]
        for pnew_index  in pnew_index_list[i]:
            candidate_newindex = []
            candidate_newindex.append(pnew_index)
            candidate_newindex = candidate_newindex + random.sample(nnew_index_list[i], 4)
            ## shuffle新闻集
            candidate_order = list(range(4 + 1))
            random.shuffle(candidate_order)
            ## shuffle列表
            candidate_shuffle = []
            label_list_shuffle = []
            for i in candidate_order:
                candidate_shuffle.append(candidate_newindex[i])
                label_list_shuffle.append(label[i])
            # 标签
            label_list.append(label_list_shuffle)
            # 候选新闻index
            candidate_newindex_list.append(candidate_shuffle)
            # 用户index
            user_index.append(user_index_temp)
    newindex_label_df = pd.DataFrame()
    newindex_label_df['user_index'] = user_index
    newindex_label_df['candidate_newindex'] = candidate_newindex_list
    newindex_label_df['label'] = label_list
    newindex_label_df.to_csv('../Data/df/newindex_label_df.csv')

    ## 保存用户index列表
    user_index = np.array(user_index)
    # print('-----------user_index.npy--------------')
    # print(user_index)
    np.save('../Data/metadata/user_index.npy', user_index)

    ## 保存候选新闻index列表
    candidate_newindex = np.array(candidate_newindex_list)
    # print('-----------candidate_newindex.npy--------------')
    # print(candidate_newindex)
    np.save('../Data/metadata/candidate_newindex.npy', candidate_newindex)

    ## 保存label列表
    label = np.array(label_list)
    # print('-----------label.npy--------------')
    # print(label)
    np.save('../Data/metadata/label.npy', label)

    return newindex_label_df

### 构建用户点击新闻和候选新闻index
def construct_clicked_newindex_cand_newindex( user_clicked_df,  user_clicked_newid_list, new_dict, maxlen_clicked ):
    '''
    根据新闻字典，将用户点击新闻ID映射成index
    :param user_clicked_df:
    :param user_clicked_newid_list:
    :param new_dict:
    :return: DataFrame(newid_Label_df):
    '''
    user_clicked_newindex_in = []
    user_clicked_newindex_list = []

    for i in range(len(user_clicked_newid_list)):
        user_clicked_newid_list[i] = list(set(user_clicked_newid_list[i]))
        for item in user_clicked_newid_list[i]:
            temp = new_dict[item]
            user_clicked_newindex_in.append(temp)
        user_clicked_newindex_list.append(user_clicked_newindex_in)
        user_clicked_newindex_in = []

    # 记录用户实际点击数
    temp = []
    for i in range(len(user_clicked_newindex_list)):
        if len(user_clicked_newindex_list[i]) >= maxlen_clicked:
            temp.append(maxlen_clicked)
        else:
            temp.append(len(user_clicked_newindex_list[i]))
    # 用户点击padding
    print('最大点击数{}'.format(maxlen_clicked))
    padding = new_dict['padding']
    for i in range(len(user_clicked_newindex_list)):
        if len(user_clicked_newindex_list[i]) >= maxlen_clicked:
            user_clicked_newindex_list[i] = random.sample(user_clicked_newindex_list[i] ,maxlen_clicked)
        if len(user_clicked_newindex_list[i]) < maxlen_clicked:
            need_len = maxlen_clicked - len(user_clicked_newindex_list[i])
            for j in range(need_len):
                user_clicked_newindex_list[i].append(padding)

    user_clicked_newindex = np.array(user_clicked_newindex_list)
    print(len(user_clicked_newindex_list))
    # print('--------------user_clicked_newindex---------------')
    # print(user_clicked_newindex)
    np.save('../Data/metadata/user_clicked_newindex', user_clicked_newindex)

    user_clicked_df['user_clicked_newindex'] = user_clicked_newindex_list
    user_clicked_df.to_csv('../Data/df/user_clicked_df.csv',index= False)

    # 构建用户邻居字典
    user_index_list = user_clicked_df['user_index'].tolist()
    user_clicked_newindex_list = user_clicked_df['user_clicked_newindex'].tolist()
    user_one_hop_dict = {}
    for i in range(len(user_index_list)):
        user_index = user_index_list[i]
        user_one_hop_dict[user_index] = user_clicked_newindex_list[i]
    user_one_hop_dict[len(user_index_list)] = [padding] * maxlen_clicked

    return user_clicked_df, user_one_hop_dict

### 获取标题文字
def get_title_word (new_dict, new_Info_df):
    '''
    获取新闻标题文字
    :param new_dict:
    :param new_Info_df:
    :return: DataFrame(new_Info_df_select):根据新闻id选取的新闻
             List(word_total_list_t):所有新闻标题的单词List
    '''
    New_id_list = list(new_dict.keys())
    new_Info_df_select = pd.DataFrame(data=None, columns=['id', 'category', 'subcategory', 'title', 'abstract', 'url',
                                                          'title_entities', 'abstract_entities'])
    for New_id in New_id_list:
        df_select = new_Info_df[new_Info_df['id'] == New_id]
        new_Info_df_select = pd.concat([new_Info_df_select, df_select], axis=0)
    word_list = []
    word_total_list_t = []
    title_word_list = new_Info_df_select['title'].tolist()

    for title_word in title_word_list:
        words = word_tokenize(title_word)
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', "'"]  # 定义符号列表
        cutwords2 = [word for word in words if word not in interpunctuations]  # 去除标点符号
        stops = set(stopwords.words("english"))
        cutwords3 = [word for word in cutwords2 if word not in stops]
        for word in cutwords3:
            word_list.append(word)
        word_total_list_t.append(word_list)
        word_list = []
    word_total_list_t.append(['padding'])

    # for title_word in title_word_list:
    #     split = title_word.split(' ')
    #     for word in split:
    #         word_list.append(word)
    #     word_total_list_t.append(word_list)
    #     word_list = []
    # word_total_list_t.append(['padding'])

    new_title = new_Info_df_select['title'].values.tolist()
    # new_abstract = new_Info_df_select['abstract'].values.tolist()
    # new_title.append('padding')
    # new_abstract.append('padding')
    # np.save('pretrain-data/new_title.npy',np.array(new_title))
    # np.save('pretrain-data/new_abstract.npy', np.array(new_abstract))
    return new_Info_df_select, word_total_list_t, np.array(new_title)

## 获取标题嵌入
def get_news_embedding (new_title):
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained("bert-base-uncased")
    title_text = new_title
    title_text = title_text.tolist()
    #
    # encoded_input = tokenizer(title_text, padding=True, truncation=True, return_tensors='tf')
    # output = model(encoded_input)
    # new_title_embedding = output[1].numpy()
    new_title_embedding = []
    for i in range(len(title_text)+1):
        new_title_embedding.append(np.random.rand(400))
    np.save('../Data/metadata/new_title_embedding.npy', np.array(new_title_embedding))

### 获取标题文字index
def get_title_word_index (word_total_list_t, new_Info_df_select):
    '''
    将新闻单词映射到index
    :param word_total_list_t:
    :param new_Info_df_select:
    :return:
            DataFrame(new_Info_df_select)：选择的新闻信息
            List(title_word_index):单词对应的index
            max_len：新闻标题的最大长度（用于对齐）
    '''
    word_dict = {}
    index = 0
    word_index_list_one = []
    word_index_list = []
    for word_total_list_t_one in word_total_list_t:
        for word in word_total_list_t_one:
            if (word not in word_dict.keys()) == True:
                word_dict[word] = index
                index += 1
    for word_total_list_t_one in word_total_list_t:
        for word in word_total_list_t_one:
            word_index_list_one.append(word_dict[word])
        word_index_list.append(word_index_list_one)
        word_index_list_one = []
    word_dict['padding'] = index
    ## 构建映射矩阵
    word_index_df = pd.DataFrame(pd.Series(word_dict), columns=['word_id'])
    word_index_df = word_index_df.reset_index().rename(columns={'index': 'word'})
    word_index_df.to_csv('../Data/df/word_index_df.csv', index= False)
    new_Info_df_select['title_word_id'] = word_index_list[:len(word_index_list)-1]
    title_word_index = word_index_list
    ## 记录单词长度
    new_title_length = []
    for i in range(len(title_word_index)):
        new_title_length.append(len(title_word_index[i]))
    new_title_length = np.array(new_title_length)
    np.save('../Data/metadata/new_title_length.npy', new_title_length)
    ## 求最大长度
    max_len = 0
    for i in range(len(title_word_index)):
        if len(title_word_index[i]) > max_len:
            max_len = len(title_word_index[i])

    ## 按照最大长度补全
    empty_word= [word_index_df.shape[0]-1] * max_len
    for i in range(len(title_word_index)):
        if len(title_word_index [i]) == 0:
            title_word_index [i].extend(empty_word)
        elif len(title_word_index[i]) < max_len:
            need_len = max_len - len(title_word_index[i])
            title_word_index[i].extend([word_index_df.shape[0]-1] * need_len)
    title_word_index = np.array(title_word_index)
    np.save('../Data/metadata/new_title_word_index.npy', title_word_index)

    # print('-----------word_index_df------------')
    # print(word_index_df)
    # print('------------title_word_index-------------')
    # print(title_word_index)
    print('词袋数 {}'.format(word_index_df.shape[0]))
    print('新闻标题最大长度 {}'.format(max_len))

    return new_Info_df_select, title_word_index, max_len

### 读取新闻信息，提取每一个新闻的标题实体ID和摘要实体ID
def get_title_and_abstract_entity_id(new_Info_df_select):
    '''
    读取新闻信息，提取每一个新闻的标题实体ID和摘要实体ID
    :param new_Info_df_select:
    :return: DataFrame(new_Info_df_select) :加入提取标题实体ID和摘要实体ID的新闻信息
             List(entity_id_total_list_a)：摘要实体ID
             List(entity_id_total_list_t)：标题实体ID
    '''
    title_entities_list = new_Info_df_select['title_entities'].values.tolist()
    abstract_entities_list = new_Info_df_select['abstract_entities'].values.tolist()
    ## 提取标题实体ID
    entity_id_total_list_t = []
    for title_entities in title_entities_list:
        entity_id_list_t = []
        if type(title_entities) == float:
            entity_id_list_t.append('padding')
        else:
            if len(title_entities) == 2:
                entity_id_list_t.append('padding')
            else:
                temp = eval(title_entities)
                for i in range(len(temp)):
                    entity_id = temp[i]['WikidataId']
                    entity_id_list_t.append(entity_id)

        entity_id_total_list_t.append(entity_id_list_t)

    entity_id_total_list_a = []
    for abstract_entities in abstract_entities_list:
        entity_id_list_a = []
        if type(abstract_entities) == float:
            entity_id_list_a.append('padding')
        else:
            if len(abstract_entities) == 2:
                entity_id_list_a.append('padding')
            else:
                temp = eval(abstract_entities)
                for i in range(len(temp)):
                    entity_id = temp[i]['WikidataId']
                    entity_id_list_a.append(entity_id)
        entity_id_total_list_a.append(entity_id_list_a)

    return new_Info_df_select, entity_id_total_list_t, entity_id_total_list_a

### 将标题实体ID和摘要实体ID列表合并
def merge_title_and_abstract_list(entity_information_t ,entity_information_a, new_Info_df_select):
    '''
    合并实体ID和摘要实体ID
    :param entity_id_total_list_a:
    :param entity_id_total_list_t:
    :param new_Info_df_select:
    :return:
        DataFrame(new_Info_df_select):加入合并实体的新闻信息
         List(entity_id_list)：合并后的实体IDlist
    '''
    entity_id_list = []
    for i in range(len(entity_information_a)):
        temp = entity_information_a[i]
        for entity_id in entity_information_t[i]:
            temp.append(entity_id)
        entity_id_list.append(temp)
    new_Info_df_select['entity_id'] = entity_id_list
    entity_index_df = pd.DataFrame()
    entity_index_df['entity_id'] = entity_id_list
    entity_index_df.to_csv('../Data/df/entity_index_df.csv')
    return new_Info_df_select, entity_id_list

### 获取实体id映射，方便提取实体嵌入
def map_entity_information(entity_id_list, new_Info_df_select, max_len):
    '''
    映射新闻实体ID
    :param entity_id_list:
    :param new_Info_df_select:
    :param max_len:
    :return: DataFrame(entity_index_df):实体ID和实体index df
    '''
    # 映射id
    id_dict = {}
    id_dict['padding'] = 0
    index = 1
    entity_index_list = []
    for entity_id_list_one in entity_id_list:
        entity_index_list_one = []
        for entity_id in entity_id_list_one:
            if (entity_id not in id_dict.keys()) == True :
                id_dict[entity_id] = index
                index = index + 1
            entity_index = id_dict[entity_id]
            entity_index_list_one.append(entity_index)
        entity_index_list.append(entity_index_list_one)
    new_Info_df_select['entity_index'] = entity_index_list
    entity_index_df = pd.DataFrame(pd.Series(id_dict))
    entity_index_df = entity_index_df.reset_index()
    entity_index_df.columns = ['id','index']
    entity_index_df.to_csv('../Data/df/entity_index_df.csv')
    entity_index = new_Info_df_select['entity_index'].values.tolist()
    print('单个新闻最大实体个数{}'.format(max_len))
    empty_entity = [id_dict['padding']] * max_len
    for i in range(len(entity_index)):
        if len(entity_index[i]) == 0:
            entity_index[i].extend(empty_entity)
        elif len(entity_index[i])  < max_len:
            need_len = max_len - len(entity_index[i])
            entity_index[i].extend([id_dict['padding']] * need_len)
        entity_index[i] = [int(entity_index[i][a]) for i in range(max_len)]
        entity_index[i] = np.array(entity_index[i])
        if len(entity_index[i]) != max_len:
            print('error')

    entity_index.append(empty_entity)
    entity_index = np.array(entity_index)
    np.save('../Data/metadata/new_entity_index.npy', entity_index)
    return entity_index_df, id_dict

### 提取实体嵌入向量
def extract_entity_vector(entity_index_df, entity_embedding_df):
    '''
    提取实体嵌入向量
    :param entity_index_df:
    :param entity_embedding_df:
    :return:
        DataFrame(entity_index_df):加入实体向量之后的entity_index_df
        List(embedding_list)
    '''
    ## 加载实体嵌入
    entity_embedding_df['vector'] = entity_embedding_df.iloc[:, 1:101].values.tolist()
    entity_embedding_df = entity_embedding_df[[0,'vector']].rename(columns={0: "entity"})
    ## 加载映射dataframe，按照映射id提取实体嵌入
    entity_id_list = entity_index_df['id'].values.tolist()
    entity_embedding_list = []
    empty_id = 100 * [0]
    for entity_id in entity_id_list:
        entity_embedding = entity_embedding_df[entity_embedding_df['entity'] == entity_id]['vector'].values
        if len(entity_embedding) == 0:
            entity_embedding_list.append(empty_id)
        else:
            entity_embedding_list.append(entity_embedding[0])
    entity_index_df ['entity_embedding'] = entity_embedding_list
    entity_index_df = entity_index_df.fillna(method='ffill')
    embedding_list = entity_index_df['entity_embedding'].values.tolist()
    # embedding_list.append(empty_id)
    entity_embedding_index = np.array(embedding_list)
    np.save('../Data/metadata/new_entity_embedding.npy', entity_embedding_index)

    return entity_index_df

### 获取主题
def get_category_index(new_Info_df_select):
    '''
    获取新闻主题
    :param new_Info_df_select:
    :return:
        DataFrame(new_Info_df_select):加入新闻主题index的新闻信息df
    '''
    index = 0
    category_dict = {}
    category_list = new_Info_df_select['category'].values.tolist()
    for category in category_list:
        if (category not in category_dict.keys()) == True:
            category_dict[category] = index
            index += 1
    category_dict['padding'] = index
    category_index_list = []
    for i in range(len(category_list)):
        category = category_list[i]
        category_index_list.append(category_dict[category])

    new_Info_df_select['category_index'] = category_index_list
    category_index_list.append(category_dict['padding'])
    np.save('../Data/metadata/new_category_index.npy', np.array(category_index_list))
    category_index_df = pd.DataFrame(pd.Series(category_dict), columns=['category_index'])
    category_index_df = category_index_df.reset_index().rename(columns={'index': 'category'})
    category_index_df.to_csv('../Data/df/category_index_df.csv', index=False)
    print('主题个数{}'.format(category_index_df.shape[0]))
    np.save('../Data/metadata/category_index.npy', np.array(category_index_df['category_index'].values.tolist()))
    return new_Info_df_select, category_dict

### 获取子主题
def get_subcategory_index(new_Info_df_select):
    '''
        获取新闻副主题
        :param new_Info_df_select:
        :return:
            DataFrame(new_Info_df_select):加入新闻副主题index的新闻信息df
    '''
    index = 0
    subcategory_dict = {}
    subcategory_list = new_Info_df_select['subcategory'].values.tolist()
    for subcategory in subcategory_list:
        if (subcategory not in subcategory_dict.keys()) == True:
            subcategory_dict[subcategory] = index
            index += 1
    subcategory_dict['padding'] = index
    subcategory_index_list = []

    for i in range(len(subcategory_list)):
        subcategory = subcategory_list[i]
        subcategory_index_list.append(subcategory_dict[subcategory])

    new_Info_df_select['subcategory_index'] = subcategory_index_list
    subcategory_index_list.append(subcategory_dict['padding'])
    np.save('../Data/metadata/new_subcategory_index.npy', np.array(subcategory_index_list))
    subcategory_index_df = pd.DataFrame(pd.Series(subcategory_dict), columns=['subcategory_index'])
    subcategory_index_df = subcategory_index_df.reset_index().rename(columns={'index': 'subcategory'})
    subcategory_index_df.to_csv('../Data/df/subcategory_index_df.csv', index=False)
    print('子主题个数{}'.format(subcategory_index_df.shape[0]))
    np.save('../Data/metadata/subcategory_index', np.array(subcategory_index_df['subcategory_index'].values.tolist()))
    new_Info_df_select.to_csv('../Data/df/new_Info_df.csv')
    return new_Info_df_select, subcategory_dict

if __name__ == "__main__":
    ### 读取用户历史交互记录
    user_behaviors_df, new_Info_df, entity_embedding_df = load_data_behaviors_New()
    ### 获取用户id和点击新闻id
    user_pnewid_nnew_id_df, total_newid_list = get_total_user_id_and_label(user_behaviors_df)
    ### 获取用户index
    newid_Label_df, user_clicked_df, user_clicked_newid_list, user_dict = get_user_index( user_behaviors_df, user_pnewid_nnew_id_df)
    ## 获取用户点击新闻id和候选新闻id映射
    new_dict = get_index_clicked_newid_cand_newid(user_clicked_newid_list, total_newid_list)
    ## 获取测试集
    get_test_new_user(new_dict, user_pnewid_nnew_id_df)
    ### 负采样
    negtivate_sample(user_pnewid_nnew_id_df, new_dict)
    ### 构建用户点击新闻和候选新闻index
    user_clicked_df, user_one_hop_dict = construct_clicked_newindex_cand_newindex( user_clicked_df,  user_clicked_newid_list, new_dict, maxlen_clicked = 10)
    ### 获取标题文字
    new_Info_df_select, word_total_list_t, new_title = get_title_word(new_dict, new_Info_df)
    ### 获取文章嵌入
    get_news_embedding(new_title)
    ### 获取文字index
    new_Info_df_select, word_id_list, max_len = get_title_word_index (word_total_list_t, new_Info_df_select)
    ### 读取新闻信息，提取每一个新闻的标题实体ID和摘要实体ID
    new_Info_df_select, entity_information_t, entity_information_a = get_title_and_abstract_entity_id( new_Info_df_select)
    ### 将标题实体ID和摘要实体ID列表合并
    new_Info_df_select, entity_id_list = merge_title_and_abstract_list( entity_information_a, entity_information_t, new_Info_df_select)
    ### 映射新闻实体index
    entity_index_df, id_dict = map_entity_information(entity_id_list, new_Info_df_select, max_len = 20)
    ### 获取新闻实体向量
    entity_index_df = extract_entity_vector(entity_index_df, entity_embedding_df)
    ### 获取新闻主题
    new_Info_df_select, category_dict = get_category_index(new_Info_df_select)
    ### 获取副新闻主题
    new_Info_df_select, subcategory_dict = get_subcategory_index(new_Info_df_select)
