#!/usr/bin/env python

"""
Usage:
nohup python codes/get_oneil_mgaedc_representation.py  >> logs/log_oneil_mgdc_merge_reg_loewe.txt 2>&1 &
"""

import time
import os
import numpy as np
import pandas as pd
import networkx as nx
# from rdkit import Chem
# from rdkit.Chem import AllChem
import scipy.sparse as sp
from itertools import islice, combinations
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
# Train on CPU (hide GPU) due to memory constraints
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
tf.compat.v1.disable_v2_behavior() 

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('embedding_dim', 320, 'Number of the dim of embedding')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('val_test_size', 0.1, 'the rate of validation and test samples.')

#some usefull funs
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def get_classification_stats(net_labels, net_preds):
    net_preds_binary = [ 1 if x >= 0.5 else 0 for x in net_preds ]
    net_auc = roc_auc_score(net_labels, net_preds)
    precision, recall, _ = precision_recall_curve(net_labels, net_preds)
    net_aupr = auc(recall, precision)
    net_acc = accuracy_score(net_preds_binary, net_labels)
    net_f1 = f1_score(net_labels, net_preds_binary)
    net_precision = precision_score(net_labels, net_preds_binary, zero_division=0)
    net_recall = recall_score(net_labels, net_preds_binary)
    fnoutput = [net_auc,net_acc, net_aupr, net_f1, net_precision, net_recall]
    return fnoutput

# 1. load the data   
import numpy as np
import pandas as pd
data = pd.read_csv('rawdata/oneil_loewe_cutoff30.txt', sep='\t', header=0)
data.columns = ['drugname1','drugname2','cell_line','synergy','fold']
drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
drugscount = len(drugslist)
cellslist = sorted(list(set(data['cell_line']))) 
cellscount = len(cellslist)

# drugs_data = pd.read_csv('rawdata/drug2id.tsv', sep='\t', header=0)
# drugslist = list(drugs_data['drug'])
# drugscount = len(drugslist)


# get the features
# get the features
drug_feat = pd.read_csv('rawdata/oneil_drug_informax_feat.txt',sep='\t', header=None)
drug_feat = sp.csr_matrix( drug_feat )
drug_feat = sparse_to_tuple(drug_feat.tocoo())
num_drug_feat = drug_feat[2][1]
num_drug_nonzeros = drug_feat[1].shape[0]

if not os.path.isdir('results'):
    os.makedirs('results')

if not os.path.isdir('logs'):
    os.makedirs('logs')

resultspath = 'results/results_loewe/results_mgaedc_representation'
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)

all_drug_indexs = []
for idx1 in range(drugscount):
    for idx2 in range(drugscount):
        drugname1 = drugslist[idx1]
        drugname2 = drugslist[idx2]
        all_drug_indexs.append(drugname1 + '&' +drugname2)

indexs_all = []
indexs_all_triu = []
for idx1 in range(drugscount):
    for idx2 in range(drugscount):
        indexs_all.append([idx1, idx2])
        if idx1 < idx2:
            indexs_all_triu.append([idx1, idx2])


stats_loss = np.zeros((10,1))
# stats_auc = np.zeros((10,1))
# for test_fold in range(10):
test_fold = 0
valid_fold = list(range(10))[test_fold-1]
train_fold = [ x for x in list(range(10)) if x != test_fold and x != valid_fold ]
print(train_fold, valid_fold, test_fold)

test_data = data[data['fold']== test_fold]
valid_data = data[data['fold']==valid_fold]
train_data = data[(data['fold']!=test_fold) & (data['fold']!=valid_fold) ]
print('processing test fold {0} train folds {1} valid folds{2}.'.format(test_fold, train_fold, valid_fold))
print('test shape{0} train shape{1} valid shape {2}'.format(test_data.shape, train_data.shape, valid_data.shape))

d_net1_norm_train = {}
d_net2_norm_train = {}
d_net3_norm_train = {}
d_net1_norm_valid = {}
d_net2_norm_valid = {}
d_net3_norm_valid = {}
d_net1_index_train = {}
d_net2_index_train = {}
d_net3_index_train = {}
d_net1_index_valid = {}
d_net2_index_valid = {}
d_net3_index_valid = {}
d_net1_labels_train = {}
d_net2_labels_train = {}
d_net3_labels_train = {}
d_net1_pos_weight = {}
d_net2_pos_weight = {}
d_net3_pos_weight = {}
for cellidx in range(cellscount):
    # cellidx = 0
    cellname = cellslist[cellidx]
    print('processing ', cellname)
    each_data = data[data['cell_line']==cellname]
    net1_adj_mat_train = np.zeros((drugscount, drugscount))
    net2_adj_mat_train = np.zeros((drugscount, drugscount))
    net3_adj_mat_train = np.zeros((drugscount, drugscount))
    net1_adj_mat_valid = np.zeros((drugscount, drugscount))
    net2_adj_mat_valid = np.zeros((drugscount, drugscount))
    net3_adj_mat_valid = np.zeros((drugscount, drugscount))
    net1_train_pos = []
    net2_train_pos = []
    net3_train_pos = []
    net1_valid_pos = []
    net2_valid_pos = []
    net3_valid_pos = []
    net1_test_pos = []
    net2_test_pos = []
    net3_test_pos = []
    for each in each_data.values:
        drugname1, drugname2, cell_line, synergy, fold = each
        drugidx1 = drugslist.index(drugname1)
        drugidx2 = drugslist.index(drugname2)
        if drugidx2 < drugidx1:
            drugidx1, drugidx2 = drugidx2, drugidx1
        #net1
        if float(synergy) >= 30:
            # net1_pos.append([drugidx1, drugidx2])
            #train
            if fold in train_fold:
                net1_adj_mat_train[drugidx1, drugidx2] = 1
                net1_train_pos.append([drugidx1, drugidx2])
                net1_train_pos.append([drugidx2, drugidx1])
            elif fold == valid_fold:
                net1_valid_pos.append([drugidx1, drugidx2])
                net1_valid_pos.append([drugidx2, drugidx1])
            # #test
            elif fold == test_fold:
                net1_test_pos.append([drugidx1, drugidx2])
                net1_test_pos.append([drugidx2, drugidx1])
        #net2
        elif (float(synergy) < 30) and (float(synergy) > 0):
            # net2_pos.append([drugidx1, drugidx2])
        #train
            if fold in train_fold:
                net2_adj_mat_train[drugidx1, drugidx2] = 1
                net2_train_pos.append([drugidx1, drugidx2])
                net2_train_pos.append([drugidx2, drugidx1])
            elif fold == valid_fold:
                net2_valid_pos.append([drugidx1, drugidx2])
                net2_valid_pos.append([drugidx2, drugidx1])
            #test
            elif fold == test_fold:
                net2_test_pos.append([drugidx1, drugidx2])
                net2_test_pos.append([drugidx2, drugidx1])
        #net3
        elif float(synergy) < 0:
            # net3_pos.append([drugidx1, drugidx2])
        #train
            if fold in train_fold:
                net3_adj_mat_train[drugidx1, drugidx2] = 1
                net3_train_pos.append([drugidx1, drugidx2])
                net3_train_pos.append([drugidx2, drugidx1])
            elif fold == valid_fold:
                net3_valid_pos.append([drugidx1, drugidx2])
                net3_valid_pos.append([drugidx2, drugidx1])
            #test
            elif fold == test_fold:
                net3_test_pos.append([drugidx1, drugidx2])
                net3_test_pos.append([drugidx2, drugidx1])
    #net1
    net1_adj_mat_train = sp.csr_matrix(net1_adj_mat_train)
    net1_adj_mat_train = net1_adj_mat_train + net1_adj_mat_train.T
    net1_adj_norm_train = preprocess_graph(net1_adj_mat_train)
    d_net1_norm_train[cellidx] = net1_adj_norm_train

    net1_labels_train = net1_adj_mat_train + sp.eye(net1_adj_mat_train.shape[0])
    net1_labels_train = sparse_to_tuple(net1_labels_train)
    net1_labels_train_pos = [indexs_all.index(x) for x in net1_labels_train[0].tolist() ]
    net1_labels_train_neg = [ x for x in list(range(len(indexs_all))) if x not in net1_labels_train_pos ]
    d_net1_labels_train[cellidx] = [net1_labels_train_pos, net1_labels_train_neg]
    # d_net1_labels_train[cellidx] = net1_labels_train
    net1_pos_weight = float(net1_adj_mat_train.shape[0] * net1_adj_mat_train.shape[0] - net1_adj_mat_train.sum() ) / net1_adj_mat_train.sum()
    d_net1_pos_weight[cellidx] = net1_pos_weight

    # net1_index_train_neg = [ indexs_all.index(x) for x in indexs_all_triu if x not in  net1_train_pos and x not in net1_valid_pos and x not in net1_test_pos]
    net1_index_train_pos = [ indexs_all.index(x) for x in net1_train_pos ]
    net1_index_valid_pos = [ indexs_all.index(x) for x in net1_valid_pos ]
    net1_index_train_neg = [ indexs_all.index(x) for x in net2_train_pos + net3_train_pos ]
    net1_index_valid_neg = [ indexs_all.index(x) for x in net2_valid_pos + net3_valid_pos ]
    d_net1_index_train[cellidx] = [ net1_index_train_pos, net1_index_train_neg ]
    d_net1_index_valid[cellidx] = [net1_index_valid_pos, net1_index_valid_neg ]


    #net2
    net2_adj_mat_train = sp.csr_matrix(net2_adj_mat_train)
    net2_adj_mat_train = net2_adj_mat_train + net2_adj_mat_train.T
    net2_adj_norm_train = preprocess_graph(net2_adj_mat_train)
    d_net2_norm_train[cellidx] = net2_adj_norm_train

    net2_labels_train = net2_adj_mat_train + sp.eye(net2_adj_mat_train.shape[0])
    net2_labels_train = sparse_to_tuple(net2_labels_train)
    net2_labels_train_pos = [indexs_all.index(x) for x in net2_labels_train[0].tolist() ]
    net2_labels_train_neg = [ x for x in list(range(len(indexs_all))) if x not in net2_labels_train_pos ]
    d_net2_labels_train[cellidx] = [net2_labels_train_pos, net2_labels_train_neg]
    # d_net2_labels_train[cellidx] = net2_labels_train
    net2_pos_weight = float(net2_adj_mat_train.shape[0] * net2_adj_mat_train.shape[0] - net2_adj_mat_train.sum() ) / net2_adj_mat_train.sum()
    d_net2_pos_weight[cellidx] = net2_pos_weight

    net2_index_train_pos = [ indexs_all.index(x) for x in net2_train_pos ]
    net2_index_valid_pos = [ indexs_all.index(x) for x in net2_valid_pos ]
    net2_index_train_neg = [ indexs_all.index(x) for x in net1_train_pos + net3_train_pos ]
    net2_index_valid_neg = [ indexs_all.index(x) for x in net1_valid_pos + net3_valid_pos ]
    d_net2_index_train[cellidx] = [ net2_index_train_pos, net2_index_train_neg ]
    d_net2_index_valid[cellidx] =  [ net2_index_valid_pos, net2_index_valid_neg ]

    #net3
    net3_adj_mat_train = sp.csr_matrix(net3_adj_mat_train)
    net3_adj_mat_train = net3_adj_mat_train + net3_adj_mat_train.T
    net3_adj_norm_train = preprocess_graph(net3_adj_mat_train)
    d_net3_norm_train[cellidx] = net3_adj_norm_train

    net3_labels_train = net3_adj_mat_train + sp.eye(net3_adj_mat_train.shape[0])
    net3_labels_train = sparse_to_tuple(net3_labels_train)
    net3_labels_train_pos = [indexs_all.index(x) for x in net3_labels_train[0].tolist() ]
    net3_labels_train_neg = [ x for x in list(range(len(indexs_all))) if x not in net3_labels_train_pos ]
    d_net3_labels_train[cellidx] = [net3_labels_train_pos, net3_labels_train_neg]
    # d_net3_labels_train[cellidx] = net3_labels_train
    net3_pos_weight = float(net3_adj_mat_train.shape[0] * net3_adj_mat_train.shape[0] - net3_adj_mat_train.sum() ) / net3_adj_mat_train.sum()
    d_net3_pos_weight[cellidx] = net3_pos_weight

    net3_index_train_pos = [ indexs_all.index(x) for x in net3_train_pos ]
    net3_index_valid_pos = [ indexs_all.index(x) for x in net3_valid_pos ]
    net3_index_train_neg = [ indexs_all.index(x) for x in net1_train_pos + net2_train_pos ]
    net3_index_valid_neg = [ indexs_all.index(x) for x in net1_valid_pos + net2_valid_pos ]
    d_net3_index_train[cellidx] = [ net3_index_train_pos, net3_index_train_neg ]
    d_net3_index_valid[cellidx] =  [net3_index_valid_pos, net3_index_valid_neg ]


placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
}
placeholders.update({'net1_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
# placeholders.update({'net1_labels_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
placeholders.update({'net2_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
# placeholders.update({'net1_labels_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
placeholders.update({'net3_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
# placeholders.update({'net1_labels_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})

# Create model
from models.model_merge_simple_reg_3decoders import mgdc
model = mgdc(placeholders, num_drug_feat, num_drug_nonzeros, FLAGS.embedding_dim, fncellscount=cellscount , name='mgdc')

# d_net1_labels_train = {cellidx: tf.sparse_tensor_to_dense(placeholders['net1_labels_'+str(cellidx)], validate_indices=False) for cellidx in range(cellscount)}

from models.optimizer_merge_reg_3decoders import optimizer
with tf.name_scope('optimizer'):
    opt = optimizer(model=model, preds_specific=model.reconstructions_specific, preds_common=model.reconstructions_common, lr=FLAGS.learning_rate, d_net1_indexs = d_net1_index_train,d_net2_indexs = d_net1_index_train, d_net3_indexs = d_net1_index_train, fncellscount=cellscount)


# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# FLAGS.epochs = 10
best_auc = 0
min_loss = float('inf')
best_cell_stats = []

saver = tf.train.Saver(max_to_keep=1)

for epoch in range(FLAGS.epochs):
    # epoch =  0
    feed_dict = dict()
    feed_dict.update({placeholders['features']: drug_feat})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['net1_adj_norm_'+str(cellidx)] : d_net1_norm_train[cellidx] for cellidx in range(cellscount)})
    feed_dict.update({placeholders['net2_adj_norm_'+str(cellidx)] : d_net2_norm_train[cellidx] for cellidx in range(cellscount)})
    feed_dict.update({placeholders['net3_adj_norm_'+str(cellidx)] : d_net3_norm_train[cellidx] for cellidx in range(cellscount)})
    _, train_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    if epoch % 100 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{}".format(train_loss))

    feed_dict.update({placeholders['dropout']: 0})
    res1, res2  = sess.run( [model.reconstructions_common, model.reconstructions_specific], feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    #get the valid loss
    valid_loss = 0
    for cellidx in range(cellscount):
        # cellidx = 0
        #net1
        net1_index_pos, net1_index_neg = d_net1_index_valid[cellidx]
        pos_neg_weight = len(net1_index_neg) / len(net1_index_pos)
        net1_labels_pos = [1] * len(net1_index_pos)
        net1_labels_neg = [0] * len(net1_index_neg)
        #common reconstruct
        net1_preds_common_pos = [ sigmoid(res1[cellidx][0].reshape(-1,1)[x][0]) for x in net1_index_pos]
        net1_preds_common_neg = [ sigmoid(res1[cellidx][0].reshape(-1,1)[x][0]) for x in net1_index_neg]
        net1_loss_common =  pos_neg_weight * mean_squared_error(net1_preds_common_pos, net1_labels_pos) +  mean_squared_error(net1_preds_common_neg, net1_labels_neg)
        #sepcific 
        net1_preds_specific_pos = [ sigmoid(res2[cellidx][0].reshape(-1,1)[x][0]) for x in net1_index_pos]
        net1_preds_specific_neg = [ sigmoid(res2[cellidx][0].reshape(-1,1)[x][0]) for x in net1_index_neg]
        net1_loss_specific =  pos_neg_weight * mean_squared_error(net1_preds_specific_pos, net1_labels_pos) +  mean_squared_error(net1_preds_specific_neg, net1_labels_neg)
        net1_loss = net1_loss_common + net1_loss_specific
        valid_loss += net1_loss

        #net2
        net2_index_pos, net2_index_neg = d_net2_index_valid[cellidx]
        pos_neg_weight = len(net2_index_neg) / len(net2_index_pos)
        net2_labels_pos = [1] * len(net2_index_pos)
        net2_labels_neg = [0] * len(net2_index_neg)
        #common reconstruct
        net2_preds_common_pos = [ sigmoid(res1[cellidx][1].reshape(-1,1)[x][0]) for x in net2_index_pos]
        net2_preds_common_neg = [ sigmoid(res1[cellidx][1].reshape(-1,1)[x][0]) for x in net2_index_neg]
        net2_loss_common =  pos_neg_weight * mean_squared_error(net2_preds_common_pos, net2_labels_pos) +  mean_squared_error(net2_preds_common_neg, net2_labels_neg)
        #sepcific 
        net2_preds_specific_pos = [ sigmoid(res2[cellidx][1].reshape(-1,1)[x][0]) for x in net2_index_pos]
        net2_preds_specific_neg = [ sigmoid(res2[cellidx][1].reshape(-1,1)[x][0]) for x in net2_index_neg]
        net2_loss_specific =  pos_neg_weight * mean_squared_error(net2_preds_specific_pos, net2_labels_pos) +  mean_squared_error(net2_preds_specific_neg, net2_labels_neg)
        net2_loss = net2_loss_common + net2_loss_specific
        valid_loss += net2_loss

        #net3
        net3_index_pos, net3_index_neg = d_net3_index_valid[cellidx]
        pos_neg_weight = len(net3_index_neg) / len(net3_index_pos)
        net3_labels_pos = [1] * len(net3_index_pos)
        net3_labels_neg = [0] * len(net3_index_neg)
        #common reconstruct
        net3_preds_common_pos = [ sigmoid(res1[cellidx][2].reshape(-1,1)[x][0]) for x in net3_index_pos]
        net3_preds_common_neg = [ sigmoid(res1[cellidx][2].reshape(-1,1)[x][0]) for x in net3_index_neg]
        net3_loss_common =  pos_neg_weight * mean_squared_error(net3_preds_common_pos, net3_labels_pos) +  mean_squared_error(net3_preds_common_neg, net3_labels_neg)
        #sepcific 
        net3_preds_specific_pos = [ sigmoid(res2[cellidx][2].reshape(-1,1)[x][0]) for x in net3_index_pos]
        net3_preds_specific_neg = [ sigmoid(res2[cellidx][2].reshape(-1,1)[x][0]) for x in net3_index_neg]
        net3_loss_specific =  pos_neg_weight * mean_squared_error(net3_preds_specific_pos, net3_labels_pos) +  mean_squared_error(net3_preds_specific_neg, net3_labels_neg)
        net3_loss = net3_loss_common + net3_loss_specific
        valid_loss += net3_loss

    if valid_loss < min_loss:
        min_loss = valid_loss
        # best_cell_stats = cells_stats
        saver.save(sess, resultspath + '/best_model.ckpt')

print('Optimization Finished!')
saver.restore(sess, resultspath + '/best_model.ckpt')

embeddings_common, embeddings_specific = sess.run( [model.embeddings_common, model.embeddings_specific], feed_dict=feed_dict)
#save embeddings specific
for cellidx in range(cellscount):
    embeddings = pd.DataFrame(embeddings_specific[cellidx])
    embeddings.index = drugslist
    embeddings.to_csv(resultspath+ '/results_embeddings_specific_' + str(cellidx) +'_' +str(test_fold)+'.txt', sep='\t',header=None, index=True)
#save embeddings common
embeddings_common = pd.DataFrame(embeddings_common)
embeddings_common.index = drugslist
embeddings_common.to_csv(resultspath+ '/results_embeddings_common_' +str(test_fold)+'.txt', sep='\t',header=None, index=True)

    # model.save_weights( resultspath + '/best_model_weights_' +str(test_fold))
    # model.save(resultspath + '/best_model.h5')
    # model.load_weights(resultspath + '/best_model_weights_' + str(test_fold) )
print('Optimization Finished!')

# stats_auc[test_fold] = best_auc
stats_loss[test_fold] = min_loss

pd.DataFrame(stats_loss).to_csv(resultspath+ '/stats_loss.txt',sep='\t',header=None,index=None)
# pd.DataFrame(stats_auc).to_csv(resultspath+ '/stats_auc.txt',sep='\t',header=None,index=None)


