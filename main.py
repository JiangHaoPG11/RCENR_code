from RC4ERec_model.RC4ERec_Train_Test import *
from DataLoad import load_data
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='KPRN')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--user_size', type = int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--user_clicked_num', type=int, default=10)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--title_size', type=int, default=400, help='新闻初始向量维数')
    parser.add_argument('--embedding_size', type=int, default=100, help='新闻和用户向量')
    parser.add_argument('--depth', type=list, default=[5,3,2], help='K-跳深度')
    parser.add_argument('--checkpoint_dir', type=str, default='./out/save_model/', help='模型保留位置')
    parser.add_argument('--category_num', type=int, default=15, help='类别向量总数')
    parser.add_argument('--subcategory_num', type=int, default=170, help='子类别向量总数')
    parser.add_argument('--word_num', type=int, default=11969, help='单词总数')
    parser.add_argument('--title_num', type=int, default=4139, help='标题新闻总数')
    parser.add_argument('--user_clicked_new_num', type=int, default=50, help='单个用户点击的新闻个数')
    parser.add_argument('--total_word_size', type=int, default=11969, help='词袋中总的单词数量')
    parser.add_argument('--title_word_size', type=int, default=23, help='每个title中的单词数量')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='单词嵌入维数')
    parser.add_argument('--attention_heads', type=int, default=20, help='多头注意力的头数')
    parser.add_argument('--num_units', type=int, default=20, help='多头注意力输出维数')
    parser.add_argument('--attention_dim', type=int, default=20, help='注意力层的维数')
    parser.add_argument('--total_entity_size', type=int, default=3803, help='总实体特征个数')
    parser.add_argument('--new_entity_size', type=int, default=20, help='单个新闻最大实体个数')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='实体嵌入维数')
    parser.add_argument('--sample_num', type=int, default=5, help='采样点数')
    parser.add_argument('--neigh_num', type=int, default=5, help='邻居节点个数')
    parser.add_argument('--category_dim', type=int, default=100, help='主题总数')
    parser.add_argument('--subcategory_dim', type=int, default=100, help='自主题总数')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='询问向量维数')
    return parser.parse_args()

def main(path, device):
    args = parse_args()
    data = load_data(args,path)
    if args.mode == "RC4ERec":
        model = RC4ERec_Train_Test(args, data, device)
        model.Train()
        model.Test()
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.dirname(os.getcwd())
    main(path, device)
