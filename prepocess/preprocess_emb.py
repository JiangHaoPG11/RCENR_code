import gensim
import numpy as np
import pandas as pd

## 获取词嵌入
def get_word_embedding(model):
    '''
    根据预训练模型获得单词嵌入向量
    :param model: 预训练模型
    :return:
    '''
    word_index_df = pd.read_csv('../Data/df/word_index_df.csv')
    word_size = word_index_df.shape[0]
    word_list = word_index_df['word'].values.tolist()
    vector_empty = [0] * 300
    word_embedding_list = []
    for i in range(word_size):
        word = word_list[i]
        if word in model:
            vector = model[word]
            word_embedding_list.append(vector)
        else:
            word_embedding_list.append(vector_empty)
    word_embedding = np.array(word_embedding_list)
    np.save('../Data/metadata/new_title_word_embedding.npy', word_embedding)
    word_index_df['word_embedding'] = word_embedding_list
    word_index_df.to_csv('../Data/df/word_index_df.csv',index = False)

## 获取主题嵌入
def get_category_embedding(model):
    '''
    根据预训练模型获得主题嵌入向量
    :param model: 预训练模型
    :return:
    '''
    category_index_df = pd.read_csv('../Data/category_index_df.csv')
    category_size = category_index_df.shape[0]
    category_list = category_index_df['category'].values.tolist()
    # model = gensim.models.KeyedVectors.load_word2vec_format('../data2/GoogleNews-vectors-negative300.bin',binary=True)
    vector_empty = [0] * 300
    category_embedding_list = []
    for i in range(category_size):
        category = category_list[i]
        if category in model:
            vector = model[category]
            category_embedding_list.append(vector)
        else:
            category_embedding_list.append(vector_empty)
    category_embedding = np.array(category_embedding_list)
    np.save('../Data/metadata/new_category_embedding.npy', category_embedding)
    category_index_df['category_embedding'] = category_embedding_list
    category_index_df.to_csv('../Data/df/category_index_df.csv',index = False)

## 获取子主题嵌入
def get_subcategory_embedding(model):
    '''
    根据预训练模型获得副主题嵌入向量
    :param model: 预训练模型
    :return:
    '''
    subcategory_index_df = pd.read_csv('../Data/df/subcategory_index_df.csv')
    subcategory_size = subcategory_index_df.shape[0]
    subcategory_list = subcategory_index_df['subcategory'].values.tolist()
    # model = gensim.models.KeyedVectors.load_word2vec_format('../data2/GoogleNews-vectors-negative300.bin',binary=True)
    vector_empty = [0] * 300
    subcategory_embedding_list = []
    for i in range(subcategory_size):
        subcategory = subcategory_list[i]
        if subcategory in model:
            vector = model[subcategory]
            subcategory_embedding_list.append(vector)
        else:
            subcategory_embedding_list.append(vector_empty)
    subcategory_embedding = np.array(subcategory_embedding_list)
    np.save('../Data/metadata/new_subcategory_embedding.npy', subcategory_embedding)
    subcategory_index_df['subcategory_embedding'] = subcategory_embedding_list
    subcategory_index_df.to_csv('../Data/df/subcategory_index_df.csv',index = False)
    
if __name__ == "__main__":
    ## 加载模型
    model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz',binary=True)
    ##  获取词嵌入
    get_word_embedding(model)
    ## 获取主题嵌入
    get_category_embedding(model)
    ## 获取子主题嵌入
    get_subcategory_embedding(model)
