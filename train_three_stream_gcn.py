from __future__ import division
from __future__ import print_function

import time
import datetime
import os
import tensorflow as tf

# 报错，tf.nn.softmax_cross_entropy_with_logits_v2

from utils import *
from models import GCN_dense_mse_2s, GCN_dense_mse_2s_little


# Settings
flags = tf.app.flags        # 添加参数，第一个是name，第二个是 默认参数，第三个是 输入参数
FLAGS = flags.FLAGS         # 取出参数


flags.DEFINE_string('dataset', 'ucf101', 'Dataset string.')     # ucf101, hmdb51, olympic_sports
flags.DEFINE_string('w2v_type', 'Yahoo_100m', 'Word2Vec Type.')     # Google_News_w2v, Yahoo_100m
flags.DEFINE_integer('w2v_dim', 500, 'dimension of the word2vec.')
flags.DEFINE_integer('time_interval', 2, 'Number of time interval for a shot.')     # 64,4,2
flags.DEFINE_integer('ini_seg_num', 32, 'Number of initial number of segments.')    # 64,32
flags.DEFINE_integer('num_class', 1588, 'Number of chossen imageNet classes.')  # 1588, 2414, 3714, 2271, 3653, 846
flags.DEFINE_integer('output_dim', 512, 'Number of units in the last layer (output the classifier).')   # 300, 500
flags.DEFINE_integer('split_ind', 0, 'current zero-shot split.')
flags.DEFINE_integer('topK', 50, 'we choose topK objects for each segment.')    # 40, 50, 100, 150, 200
flags.DEFINE_bool('use_normalization', 1, 'use_normalization for the classifiers.')
flags.DEFINE_bool('use_softmax', 1, 'use softmax or sigmoid for the classification.')
flags.DEFINE_bool('use_self_attention', 1, 'use self_attention or not.')
flags.DEFINE_integer('label_num', 101, 'number of actions.')
flags.DEFINE_integer('batch_size', 48, 'batch size.')
flags.DEFINE_string('use_little', 'no_use', 'whether use the little network')   # no_use, use_little, use_three_layer
flags.DEFINE_string('result_save_path', './results/', 'results save dir')


flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')   # 0.001, 0.0001
flags.DEFINE_string('save_path', './output_models/', 'save dir')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')     # 2048, 1024, 512, 300
flags.DEFINE_integer('hidden2', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


now_time = datetime.datetime.now().strftime('%Y-%m-%d-%T')

# Load data
data_path = 'data_yahoo_100m_v2'
all_att_inds, all_att_scores = get_imageNet_input_data(FLAGS.dataset, FLAGS.time_interval, FLAGS.ini_seg_num, FLAGS.num_class, root=data_path)  # 调用utils.py文件，N个视频，N个O×T的object矩阵，N个O×T的scores矩阵
adj_all, adj_att, features, y_train, y_val, idx_train, idx_val, train_mask, test_mask, lookup_table = \
        load_data_action_zero_shot(FLAGS.dataset, FLAGS.w2v_type, FLAGS.split_ind, data_path = data_path)# data_yahoo_100m

# adj_all --graph_all.pkl的图邻接矩阵，  adj_att --graph_att.pkld的图邻接矩阵，   features --Yahoo_100m.kpl文件的内容(1689,500)，
# y_train --label[i]在zero_shot_train_classes中的i，  y_val --label[i]在zero_shot_test_classes中的i，
# idx_train --zero_shot_train_classes保存train的video，对应label[i]，    idx_val ----zero_shot_test_classes保存test的video，对应label[i]
# train_mask --lable_num的101个值中，train用到节点的lable为1, test_mask --lable_num的101个值中，test用到节点的lable为1,     lookup_table --lookup_table.pkl文件内容

label_num = len(train_mask)     # 101包含51个1
FLAGS.label_num = label_num
if FLAGS.w2v_type == 'Yahoo_100m':  # 每个词向量的维度k = 500
    FLAGS.w2v_dim = 500

# Some preprocessing
features, div_mat = preprocess_features_dense2(features)    # (1689,1689)*(1689,500),对角矩阵值为每个类，即一行的sum，对角矩阵乘以原特征   (1689,1)行的和
features_all = features
features_att = features[label_num:, :]       # 从101行开始，到1689，元组切片，(1588,500)
# print("输出词向量")
# print(features)
# print("全部知识图谱")
# print(adj_all)
# print("att知识图谱")
# print(adj_att)

if FLAGS.model == 'dense':  # 两个邻接矩阵A的初始化，support_att_batch的[s][0]、[s][1]、[s][2]拼接batch_size次
    support_all = [preprocess_adj(adj_all)]     # A+I 单位矩阵，邻接矩阵标准化,(度矩阵^-1/2)*邻接矩阵*(度矩阵^-1/2)，转化为元组
    support_att = [preprocess_adj(adj_att)]     # preprocess_adj( )这个函数，就是邻接矩阵的标准化，两个分支的D-1/2 A+I D-1/2
    support_att_batch = [preprocess_adj(adj_att)]
    for s in range(len(support_att_batch)):     # support_att_batch的[s][0]、[s][1]、[s][2]拼接batch_size次
        support_att_batch[s] = list(support_att_batch[s])   # 变list
        for i in range(FLAGS.batch_size-1):     # 0-47，batch_size长度为48
            support_att_batch[s][0] = np.concatenate((support_att_batch[s][0], support_att[s][0]+(i+1)*FLAGS.num_class))    # 对于数组拼接support_att[s][0]+i×1588
            support_att_batch[s][1] = np.concatenate((support_att_batch[s][1], support_att[s][1]))  # 元组拼接support_att[s][1] 48次
        support_att_batch[s][2] = tuple(np.array(support_att[s][2])*FLAGS.batch_size)   # support_att[s][2]拼接48次，转换成元组
    num_supports = len(support_att)     # 保存support_att长度，为1
    if FLAGS.use_little == 'use_little':
        model_func = GCN_dense_mse_2s_little
    else:
        model_func = GCN_dense_mse_2s   # 表示model_func执行的是GCN_dense_mse_2s，父类为 Model_dense_2s 卷积操作

else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

print(features.shape)

# Define placeholders 占位符，得到两个块，tmp_row_index，tmp_batch_index，一个字典placeholders的占位符
# （48,16,50）,48片相同(16,50),第一行50个0，第二行50个1，，(48,16,50),48片不同，第一片全是0，第二片全是1
if FLAGS.use_self_attention:
    seg_number = int(FLAGS.ini_seg_num / FLAGS.time_interval)   # 16
    topK = FLAGS.topK       # 每个片段，50个object
    print('topK = %d' %topK)
    tmp_row_index = np.arange(0, seg_number)    # [0,1,,,,15]，
    tmp_row_index = np.expand_dims(tmp_row_index, 1)  # 扩展行，[[0]\n,[1]\n,,,,]变为16行,shape为(16,1)
    tmp_row_index = np.expand_dims(tmp_row_index, 0)  # 扩展列，(1,16,1),一片[[0]\n,[1]\n,,,[15]\n]
    tmp_row_index = np.tile(tmp_row_index,(FLAGS.batch_size, 1, topK))  # 重复，（48,16,50）,48片(16,50),第一行50个0，第二行50个1
    tmp_batch_index = np.arange(0, FLAGS.batch_size)    # [0,,,47]
    tmp_batch_index = np.expand_dims(tmp_batch_index, 1)    # (48，1)
    tmp_batch_index = np.expand_dims(tmp_batch_index, 1)    # (48,1,1)，[ [[0]]\n,[[1]]\n,,,,[[47]]\n],48个[[0]]、、[[47]]
    tmp_batch_index = np.tile(tmp_batch_index, (1, seg_number, topK))   # tile函数重复，(48,16,50),48片，第一片全是0，第二片全是1
    placeholders = {
        'support_all': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],    # 返回一个提供值的句柄的 SparseTensor，但不能直接计算
        'support_att': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],    # num_supports为1，
        'features_all': tf.placeholder(tf.float32, shape=(features_all.shape[0], features_all.shape[1])),
        'features_att': tf.placeholder(tf.float32, shape=(FLAGS.batch_size, seg_number, 1, features_att.shape[0])),     # （48，16，1，500）
        'labels': tf.placeholder(tf.int32, shape=(FLAGS.batch_size)),
        'train_mask': tf.placeholder(tf.int32, shape=(train_mask.shape[0])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32, shape=()),
        'label_num': tf.placeholder(tf.int32, shape=())
    }
else:
    placeholders = {
        'support_all': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support_att': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features_all': tf.placeholder(tf.float32, shape=(features_all.shape[0], features_all.shape[1])),
        'features_att': tf.placeholder(tf.float32, shape=(features_att.shape[0], features_att.shape[1])),
        'labels': tf.placeholder(tf.int32),
        'train_mask': tf.placeholder(tf.int32, shape=(train_mask.shape[0])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
        'learning_rate': tf.placeholder(tf.float32, shape=()),
        'label_num': tf.placeholder(tf.int32, shape=())
    }


# Create model   ，稀疏张量，indices表示值所在的各个位置，values表示值，dense_shape表示shape（101，50）,5050,lookup_table在pkl文件读出
lookup_table_act_att = tf.SparseTensor(indices=lookup_table[0], values=lookup_table[1], dense_shape=lookup_table[2])
# print(placeholders['features_all']) # 会读入字典的features_all，这里字典还没读入参数
model = model_func(placeholders, lookup_table_act_att, input_dim=features.shape[1], logging=True)   # 警告报错，GCN_dense_mse_2s

sess = tf.Session(config=create_config_proto())     # tf.ConfigProto()主要的作用是配置tf.Session的运算方式

# Init variables
sess.run(tf.global_variables_initializer())  # 初始化模型的参数
print(tf.global_variables_initializer())

savepath = FLAGS.save_path      # /output_models
exp_name = os.path.basename(FLAGS.dataset)  # exp_name为ucf101，basename返回的是文件名
savepath = os.path.join(savepath, exp_name)     # /output_models/ucf101
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('!!! Make directory %s' % savepath)
else:
    print('### save to: %s' % savepath)

result_save_path = FLAGS.result_save_path + FLAGS.dataset + '/'     # /results/ucf101/
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
    print('!!! Make directory %s' % result_save_path)
else:
    print('### save to: %s' % result_save_path)
result_file_name = result_save_path + FLAGS.dataset + '_' + FLAGS.w2v_type + '_' \
                    + str(FLAGS.time_interval) + '_' + str(FLAGS.ini_seg_num) \
                    + '_' + str(FLAGS.num_class) + '_' + FLAGS.use_little  \
                    + str(int(FLAGS.learning_rate *100000))+ '_' + str(FLAGS.hidden1) + '_' \
                    + str(FLAGS.output_dim) + '_' + str(FLAGS.use_normalization) + '_'\
                    + str(FLAGS.use_softmax) + '_' + str(FLAGS.split_ind) + '_' \
                    + str(FLAGS.use_self_attention) + '_' + str(FLAGS.batch_size) + '_' \
                    + str(FLAGS.hidden2) + '.txt'

# Train model
now_lr = FLAGS.learning_rate    # 学习率为0.0001
y_train = np.array(y_train)     # labels为0,0,0,,,1,1,1,,,100,100,100，y_train为labels[i]
idx_train = np.array(idx_train)     # idx_train为i
y_val = np.array(y_val)         # test里视频的lable名[0:6698]
idx_val = np.array(idx_val)     # zero_shot_test_classes对应lables数组的类别个数[0:6698]
all_att_inds = np.array(all_att_inds)   # N个O×T的object矩阵(13320,16,100)
all_att_scores = np.array(all_att_scores)   # N个O×T的scores矩阵(13320,16,100)
for epoch in range(FLAGS.epochs):
    count = 0
    rand_inds = np.random.permutation(len(y_train))     # 乱序的序列,y_train为label[i]内容，即label名
    rand_inds = rand_inds[:int(len(rand_inds)/FLAGS.batch_size)*FLAGS.batch_size]  # :int(6622/48)让他能整除48
    rand_inds = np.reshape(rand_inds, [-1, FLAGS.batch_size])   # （137，48），size = 6576
    for inds in rand_inds[:int(len(rand_inds)/5)]:  # inds为0-48，batch_size为48，？？rand_inds第一行，有48
        # Construct feed dictionary
        label = y_train[inds]       # (48,),label[i]在zero_shot_train_classes中的i，label为i
        video_idx = idx_train[inds]     # (48,)，保存train的video，对应label[i]名称，video_idx为动作名
        if FLAGS.use_self_attention:
            features_att_this_sample = np.zeros([FLAGS.batch_size, seg_number, FLAGS.num_class])    # (48,16,1588)的零矩阵
            att_ind = all_att_inds[video_idx]       # N个O×T的object矩阵，(13320,16,100)选48个train的视频i ->（48，16，100）
            att_score = all_att_scores[video_idx]   # N个O×T的scores矩阵，（48，16，100）
            att_ind = att_ind[:, :, :topK]          # （48，16，50），选取前50个
            att_score = att_score[:, :, :topK]      # （48，16，50），
            features_att_this_sample[tmp_batch_index, tmp_row_index, att_ind] = att_score   # 两个占位符，加前50个视频，[48,16,1588]
            # print(features_att_this_sample)
            features_att_this_sample = np.expand_dims(features_att_this_sample, 2)  # 在2位置添加数据，(48,16,1,1588)得到特征集
        else:
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :topK]
            att_score = att_score[:, :topK]
            att_activation = get_att_input_activation(att_ind, att_score, FLAGS.num_class, features_att.shape[1])
            features_att_this_sample = np.multiply(features_att, att_activation) # Note here we multiply the att scores and the att features
        feed_dict = construct_feed_dict(features_all, features_att_this_sample, support_all, support_att_batch, label, train_mask, label_num, placeholders) # 给字典跟新参数
        feed_dict.update({placeholders['learning_rate']: now_lr})   # 前一行是feed数据，给placeholders字典的值赋值

        outs = sess.run([model.opt_op, model.loss, model.optimizer._lr, model.accuracy, model.classifier, model.attend_feat], feed_dict=feed_dict)      # 输出6个结果
        # run():feed_dict将图形元素映射到值的字典
        # 报错，OP_REQUIRES failed at gather_op.cc:103、strided_slice_op.cc:248

        if count % 1 == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "sample_batch:", '%04d' % (count + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "lr=", "{:.5f}".format(float(outs[2])))
            count += 1
    # model.save(sess=sess, save_path=savepath)
    test_accuracy = 0
    test_inds = np.arange(len(y_val))   # test里视频的lable名[0:6698]
    test_inds = test_inds[:int(len(test_inds) / FLAGS.batch_size) * FLAGS.batch_size]
    test_inds = np.reshape(test_inds, [-1, FLAGS.batch_size])
    count_test = 0
    for inds in test_inds:
        # Construct feed dictionary
        label = y_val[inds]
        video_idx = idx_val[inds]
        if FLAGS.use_self_attention:
            features_att_this_sample = np.zeros([FLAGS.batch_size, seg_number, FLAGS.num_class])
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :, :topK]
            att_score = att_score[:, :, :topK]
            features_att_this_sample[tmp_batch_index, tmp_row_index, att_ind] = att_score
            features_att_this_sample = np.expand_dims(features_att_this_sample, 2)
        else:
            att_ind = all_att_inds[video_idx]
            att_score = all_att_scores[video_idx]
            att_ind = att_ind[:, :topK]
            att_score = att_score[:, :topK]
            att_activation = get_att_input_activation(att_ind, att_score, FLAGS.num_class, features_att.shape[1])
            features_att_this_sample = np.multiply(features_att, att_activation) # Note here we multiply the att scores and the att features
        feed_dict = construct_feed_dict(features_all, features_att_this_sample, support_all, support_att_batch, label,
                                        test_mask, label_num, placeholders)

        # Test step
        out = sess.run(model.accuracy, feed_dict=feed_dict)
        test_accuracy += np.sum(np.array(out[0]))
        count_test += 1
        if count_test % 10 == 0:
            print('%04d baches are processed for testing' % (count_test ))
    test_accuracy /= len(y_val)
    print("Epoch:", '%04d' % (epoch + 1),
          "accuracy=", "{:.5f}".format(float(test_accuracy)),
          )
    with open(result_file_name, 'a') as f:
        f.write(str(test_accuracy)+'\n')


print("Optimization Finished!")

sess.close()
