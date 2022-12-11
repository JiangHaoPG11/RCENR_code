from AnchorKG_model.AnchorKG_Train_Test import *
from PGPR_model.PGPR_Train_Test import *
from RippleNet_model.RippleNet_Train_Test import *
from ADAC_model.ADAC_Train_Test import *
from KPRN_model.KPRN_Train_Test import *
from MRNN_model.MRNN_Train_Test import *
from MRNNRL_model.MRNNRL_Train_Test import *
from MRNNRL_simple_model.MRNNRL_simple_Train_Test import *
from MRNNRL_simple_model_v2.MRNNRL_simple_Train_Test_v2 import *
from MRNNRL_simple_model_v3.MRNNRL_simple_Train_Test_v3 import *
from MRNNRL_simple_model_v4.MRNNRL_simple_Train_Test_v4 import *
from MRNNRL_simple_model_v5.MRNNRL_simple_Train_Test_v5 import *
from Baseline_model.Baseline_Train_Test import *
from MCCLK_model.MCCLK_Train_Test import *
from KGIN_model.KGIN_Train_Test import *
from KGCN_model.KGCN_Train_Test import *
from KGAT_model.KGAT_Train_Test import *
from LightGCN_model.LightGCN_Train_Test import *
from KGAT_VAE_model.KGAT_VAE_Train_Test import *
from VAE_model.VAE_Train_Test import *
from LightGCN_VAE_model.LightGCN_VAE_Train_Test import *
from LightGCL_model.LightGCL_Train_Test import *
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

    # RippleNet
    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
    parser.add_argument('--update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,help='whether using outputs of all hops or just the last hop when making prediction')

    # KPRN
    parser.add_argument('--kprn_path_long', type=int, default=6, help='路径长度')
    parser.add_argument('--kprn_max_path', type=int, default=5, help='每个用户项目对的最大个数')

    # ADAC
    parser.add_argument('--ADAC_path_long', type=int, default=5, help='路径长度')
    parser.add_argument('--ADAC_max_path', type=int, default=5, help='每个用户项目对的最大个数')

    # MRNN
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

    # MCCLK
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    parser.add_argument('--MCCLK_lr', type=float, default=3e-3, help='learning rate')  # default = 1e-4
    parser.add_argument('--sim_regularity', type=float, default=1e-4, help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=bool, default=True, help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument("--ind", type=str, default='mi', help="Independence modeling: mi, distance, cosine")

    # KGIN
    parser.add_argument("--n_factors", type=int, default=4, help="number of latent factor for user favour")

    # MetaKG
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--meta_batch_size', type=int, default=10, help='meta batch size')
    # LightGCN
    parser.add_argument("--lgn_layers", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--keep_prob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--dropout", type=float, default=0, help="using the dropout or not")
    return parser.parse_args()

def main(path, device):
    args = parse_args()
    data = load_data(args,path)
    if args.mode == "AnchorKG":
        model = AnchorKG_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "PGPR":
        model = PGPR_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "RippleNet":
        model = RippleNet_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "ADAC":
        model = ADAC_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "KPRN":
        model = KPRN_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNN":
        model = MRNN_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNNRL":
        model = MRNNRL_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNNRL_simple":
        model = MRNNRL_simple_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNNRL_simple_v2":
        model = MRNNRL_simple_Train_Test_v2(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNNRL_simple_v3":
        model = MRNNRL_simple_Train_Test_v3(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNNRL_simple_v4":
        model = MRNNRL_simple_Train_Test_v4(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MRNNRL_simple_v5":
        model = MRNNRL_simple_Train_Test_v5(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "Baseline":
        model = Baseline_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "MCCLK":
        model = MCCLK_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "KGIN":
        model = KGIN_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "KGCN":
        model = KGCN_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "KGAT":
        model = KGAT_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "LightGCN":
        model = LightGCN_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "KGAT_velf":
        model = KGAT_VAE_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "VAE":
        model = VAE_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "LightGCN_VAE":
        model = LightGCN_VAE_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "LightGCL":
        model = LightGCL_Train_Test(args, data, device)
        model.Train()
        model.Test()
    if args.mode == "RC4ERec":
        model = RC4ERec_Train_Test(args, data, device)
        model.Train()
        model.Test()
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.dirname(os.getcwd())
    main(path, device)
