import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import os

def pklLoad(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)

def pklSave(fname, obj):
    with open(fname, 'wb') as f:
        pkl.dump(obj, f)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):    # train_mask/test_mask, label_num全部的label
    """Create mask."""
    mask = np.zeros(l)  # label_num的0数组，train_mask这个是train中使用的action lable
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sample_mask_sigmoid(idx, h, w):
    """Create mask."""
    mask = np.zeros((h, w))
    matrix_one = np.ones((h, w))
    mask[idx, :] = matrix_one[idx, :]
    return np.array(mask, dtype=np.bool)


def load_data_vis_multi(dataset_str, use_trainval, feat_suffix, label_suffix='ally_multi'):
    """Load data."""
    names = [feat_suffix, label_suffix, 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.NELL.{}".format(dataset_str, names[i]), 'rb') as f:
            print("{}/ind.NELL.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph = tuple(objects)
    train_test_mask = []
    with open("{}/ind.NELL.index".format(dataset_str), 'rb') as f:
        train_test_mask = pkl.load(f)

    features = allx  # .tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.array(ally)

    idx_test = []
    idx_train = []
    idx_trainval = []

    if use_trainval == True:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] == 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)
    else:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] >= 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)

    idx_val = idx_test

    train_mask = sample_mask_sigmoid(idx_train, labels.shape[0], labels.shape[1])
    val_mask = sample_mask_sigmoid(idx_val, labels.shape[0], labels.shape[1])
    trainval_mask = sample_mask_sigmoid(idx_trainval, labels.shape[0], labels.shape[1])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_trainval = np.zeros(labels.shape)

    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_trainval[trainval_mask] = labels[trainval_mask]

    return adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask


def load_data_action_zero_shot(dataset_str, w2v_type, split_ind, data_path = 'data'):
    """'ucf101', 'Dataset string，
       'Yahoo_100m', 'Word2Vec Type，
        0, 'current zero-shot split，
        'data_yahoo_100m_v2'，
        七个kpl文件，ind.ucf101.Yahoo_100m、ind.ucf101.labels、ind.ucf101.graph_all、
                    ind.ucf101.graph_att、ind.ucf101.split_train、ind.ucf101.split_test、ind.ucf101.lookup_table
    Load data."""
    names = [w2v_type, 'labels', 'graph_all', 'graph_att', 'split_train', 'split_test', 'lookup_table']
    objects = []
    for i in range(len(names)):
        with open(data_path+"/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:     # 打开文件夹，data_yahoo_100m_v2/ind.ucf101.name[i]
            print(data_path+"/ind.{}.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):   # 判断当前的python版本是否是python3
                objects.append(pkl.load(f, encoding='latin1'))  # 解压.pkl文件，数据形式读到到objects
            else:
                objects.append(pkl.load(f))

    allx, ally, graph_all, graph_att, split_train, split_test, lookup_table_act_att = tuple(objects)    # 7个元组，7个pkl文件内容
    zero_shot_train_classes = split_train[split_ind, :]  # ind.ucf101.split_train的pkl文件,(50,51),取第0行数据，(51,)
    zero_shot_test_classes = split_test[split_ind, :]    # .split_test的pkl文件,(50,50),得到(50,)
    # print("zero_shot_train_classes")    # 0  2  6 ....95 96 97 98 一串具体的数字
    # print(zero_shot_train_classes)
    # print("zero_shot_test_classes")     # 1 3 5 ...89  91  99 100
    # print(zero_shot_test_classes)

    features = allx         # .tolil()，ind.ucf101.Yahoo_100m的kpl文件的内容，(1689,500)
    adj_all = nx.adjacency_matrix(nx.from_dict_of_lists(graph_all))     # 先把ind.ucf101.graph_all构图，再构邻接矩阵,(1689,1689)
    adj_att = nx.adjacency_matrix(nx.from_dict_of_lists(graph_att))     # .graph_att构图，再构邻接矩阵,(1588,1588)
    labels = np.array(ally)     # Here the label is for each video      # ind.ucf101.labels是每个视频的 lable,(13320,)
    # print("graph_all")
    # print(adj_all)
    # print("graph_att")
    # print(adj_att)
    # print("labels")     # 13320个数，0，0，，，1，2，，，100，100
    # print(labels[:15])

    # Here, idx_xxx is for indicating video samples for training, test, and validation,
    # this is a little difference between the original GCN papaer since it conduct node classification.
    # y_xxx is also for each video sample.
    idx_test = []   # 视频样本，用于测试
    idx_train = []  # 视频样本，用于训练
    y_train = []    # 训练的每个视频样本
    y_test = []     # 测试的每个视频样本

    for i in range(len(labels)):    # 视频的label，(13320,)，ind.ucf101.labels
        if labels[i] in zero_shot_train_classes:    # 标签i 在train的lable数组
            idx_train.append(i)                     # idx_train 保存train中video，对应label的i，
            y_train.append(labels[i])               # y_train 保存train中video的label
        elif labels[i] in zero_shot_test_classes:
            idx_test.append(i)                      # idx_test保存test的video i，y_test 保存 action lable
            y_test.append(labels[i])
    idx_trainval = idx_train    # 保存train的video，对应labels的i
    idx_val = idx_test          # 保存test的video，对应label的i
    y_trainval = y_train    #
    y_val = y_test          #

    # Here, we use the xxx_mask to indicate which nodes (action labels) are used in traing and tesing
    # since this is a zero-shot setting
    train_mask = zero_shot_train_classes    # 保存train的 lable 数组
    test_mask = zero_shot_test_classes      # 保存test的 lable 数组
    label_num = len(train_mask)+len(test_mask)  # label的总数量
    train_mask = sample_mask(train_mask, label_num)     # label_num的0数组里面，train 用到的lable对应位置为1
    test_mask = sample_mask(test_mask, label_num)
    # print("train_mask")  # True False  True False False False
    # print(train_mask)
    # print("test_mask")   # False  True False  True  True  True
    # print(test_mask)

    return adj_all, adj_att, features, y_train, y_val, idx_train, idx_val, train_mask, test_mask, lookup_table_act_att


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):  # 判断sparse_mx是否是 list类型
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_features_dense(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_dense2(features):   # Yahoo_100m的kpl文件的内容，
    rowsum = np.array(features.sum(1))  # 把1689的行相加，  dense的思想
    r_inv = np.power(rowsum, -1).flatten()  # rowsum的-1次方，再把rowsum降到一维元组(1689,)，按行的方向降，matrix不会改变
    r_inv[np.isinf(r_inv)] = 0.     # 当-1次方时，为0结果会无穷大，需要改为0
    r_mat_inv = sp.diags(r_inv)     # 形成一个以一维元组为对角线元素的矩阵，对角线的元素(1689,1689)
    features = r_mat_inv.dot(features)  # (1689,1689)*(1689,500),每行的和，降为一维元组，求导，对角矩阵，乘以视频矩阵

    div_mat = sp.diags(rowsum)      # 1689行相加的和，为对角线的元素，(1689,1)

    return features, div_mat


def normalize_adj(adj):     # A+I 单位矩阵，自连接
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)        # 坐标格式的稀疏矩阵，(1689,1689)
    rowsum = np.array(adj.sum(1))       # adj的行求和,即每个节点的度
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()   # -1/2次方，再降为到一维元组
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)       # 一维元组为对角线元素的矩阵，得到度矩阵D^-1/2
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # (度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)


def preprocess_adj(adj):        # adj_all --graph_all.pkl的无向图
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))  # A+I 单位矩阵，邻接矩阵标准化,(度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)
    return sparse_to_tuple(adj_normalized)      # 稀疏矩阵转化为元组


def construct_feed_dict(features_all, features_att, support_all, support_att, label, train_mask, label_num, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: label})
    feed_dict.update({placeholders['train_mask']: train_mask})
    feed_dict.update({placeholders['features_all']: features_all})
    feed_dict.update({placeholders['features_att']: features_att})
    feed_dict.update({placeholders['support_all'][i]: support_all[i] for i in range(len(support_all))})
    feed_dict.update({placeholders['support_att'][i]: support_att[i] for i in range(len(support_att))})
    feed_dict.update({placeholders['num_features_nonzero']: features_all[1].shape})
    feed_dict.update({placeholders['label_num']: label_num})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def create_config_proto():
    """Reset tf default config proto"""
    config = tf.ConfigProto()   # tf.ConfigProto()主要的作用是配置tf.Session的运算方式
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True  # 配置GPU
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0
    config.gpu_options.force_gpu_compatible = True
    # config.operation_timeout_in_ms=8000
    config.log_device_placement = False
    return config


def get_imageNet_input_data(data_set, time_interval, ini_seg_num, num_class, root='data'):
    """ ucf101，UCF101是一个动作识别数据集，包含现实的101类别动作视频
        2、Number of time interval for a shot，
        32、Number of initial number of segments，
        1588、Number of chossen imageNet classes，
        'data_yahoo_100m_v2'
        imageNet_choosen_class_scores_for_ucf101.txt一行就是一个视频文件，
    preprocess the imageNet scores for all the data (train and test), merge
       the scores for a fixed time interval. The final segment length of a video
       is ini_seg_length/time_interval"""
    ini_file_name = root + '/imageNet_choosen_class_scores_for_' + data_set.lower() + '.txt'  # .lower()全变成小写，txt文件
    save_file_name = root + '/input_data_imageNet_scores_' + data_set.lower() + '.pkl'  # 生成.pkl文件

    tt = 0  # tt is used for judging whether the pre-saved file is correct for this setting by comparing with the given time_interval
    if not (ini_seg_num % time_interval) == 0:
        print('Error: The time_interval cannot be divided by ini_seg_length', time_interval, ini_seg_num)
        sys.exit()

    # We do not load the previous data for preventing errors

    if os.path.exists(save_file_name):
        saved_data = pklLoad(save_file_name)
        # print(saved_data) #2020.12.2
        all_inds = saved_data[0]
        all_scores = saved_data[1]
        # print("all_inds") #2021.01.07
        # print(all_inds[:1])
        # print("all_scores")
        # print(all_scores[:1])
        tt = saved_data[2]
    if not tt == time_interval:     # 把视频imageNet_choosen_class_scores_for_ucf101.txt  (6400,8192)分成16的片段
        count_sample = 0
        top_K = 0
        with open(ini_file_name) as f:  # 打开.txt文件
            all_inds = []   # Save the fianl index of all data
            all_scores = []  # Save the fianl scores of all data
            for line in f:  # Here, one line in f denotes one sample
                count_sample += 1
                datas = line.split(',')     # 保存.txt每一行的内容，即一个视频
                if top_K == 0:
                    top_K = int(len(datas) / 2)     # top_k 保存txt每一行长度的一半
                    top_K = int(top_K / ini_seg_num)  # Calculate the number of topK classes per initial segment，片段的长度
                inds_one_sample = []    # Save the fianl index of all segments in the current data
                scores_one_sample = []  # Save the fianl scores of all segments in the current data
                ind = datas[:int(len(datas) / 2)]   # .txt每一行，ind为视频前一半
                score = datas[int(len(datas) / 2):]  # score为每一行后一半，即视频后一半
                for i in range(int(ini_seg_num / time_interval)):   # 一个初始片段长度32/2，是片段为16
                    final_seg_scores = np.zeros(num_class)      # 长度为num_class 1588，值为0的列表
                    start1 = int(i*time_interval*top_K)         # 0,        2*top_K,,,,,15*2*top_K
                    end1 = start1 + int(time_interval*top_K)    # 2*top_K,  4*top_K,,,,,16*2*top_K
                    ind_tmp = ind[start1:end1]                  # ind_tmp为16片段之一，ind为总的一半3200，分16段200
                    ind_tmp = [int(nn) for nn in ind_tmp]       # 把ind_tmp 每个变成int，形成列表 ind_tmp
                    # print(ind_tmp)
                    score_tmp = score[start1:end1]              # score_tmp为score的16段之一
                    score_tmp = [float(nn) for nn in score_tmp]  # Note here，每一段里所有元素变成float类型，保存 score_tmp
                    # score_tmp = [float(nn)/float(nn) for nn in score_tmp]
                    for j in range(time_interval):  # time_interval为2，a shot的间隔
                        start2 = int(j*top_K)   # top_k为len(datas)/2/32=100
                        end2 = start2+top_K
                        ii_tmp = ind_tmp[start2:end2]           # 16段分两次打分，每次top_K
                        final_seg_scores[ii_tmp] += score_tmp[start2:end2]         # ii_tmp为编号，score_tmp为值
                    final_seg_scores /= time_interval                   # 把里面的值除以2，得到平均值，
                    current_seg_inds = np.argsort(-final_seg_scores)    # 按降序排列，返回值为序号，即类object名
                    current_seg_inds = current_seg_inds[:top_K]         # 前top_K=100个object
                    current_seg_inds = np.array(current_seg_inds)       # Convert list to numpy
                    current_seg_scores = final_seg_scores[current_seg_inds]     # top_K个object的 scores
                    current_seg_scores = np.array(current_seg_scores)  # Convert list to numpy
                    # current_seg_scores[:] = 1
                    #Note here, currently we do not adopt normalization
                    # current_seg_scores /= np.sum(current_seg_scores) # Normalization
                    inds_one_sample.append(current_seg_inds)        # 16个片段之一，前top_K的object
                    scores_one_sample.append(current_seg_scores)    # 前top_K个object的scores
                all_inds.append(inds_one_sample)        # 16段的，前top_K个object
                all_scores.append(scores_one_sample)    # 16段的，前top_K个object的scores
        pklSave(save_file_name, (all_inds, all_scores, time_interval))
        print(count_sample, 'samples are processed')

    return all_inds, all_scores


def get_att_input_activation(att_ind, att_score, num_att, w2v_dim):
    activation = np.zeros((num_att,1))
    att_score = np.array(att_score)
    activation[att_ind] = att_score.transpose()
    activation = np.tile(activation, w2v_dim)

    return activation
