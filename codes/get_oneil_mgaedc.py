
"""
Usage:
nohup python codes/get_oneil_mgaedc.py  >> logs/log_oneil_mgaedc100_folds.txt 2>&1 &

"""

import numpy as np
import pandas as pd
import argparse
import os
import time
import torch
import torch.nn as nn
import pickle
import logging
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, cohen_kappa_score

from models.PRODeepSyn_datasets import FastTensorDataLoader
from models.PRODeepSyn_utils import save_args, arg_min, conf_inv, calc_stat, save_best_model, find_best_model, random_split_indices
time_str = str(datetime.now().strftime('%y%m%d%H%M'))

#files
OUTPUT_DIR = 'results/results_loewe/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# SYNERGY_FILE = os.path.join( 'oneil_synergy_10folds_cutoff30.txt')
# DRUG_FEAT_FILE = os.path.join( 'rawdata/drug_feat.npy')
# # DRUG2ID_FILE = os.path.join('rawdata/drug2id.tsv')
# # CELL_FEAT_FILE = os.path.join( 'rawdata/cell_feat.npy')
# # CELL_FEAT_FILE = os.path.join( 'target_mut.npy')
# CELL_FEAT_FILE = os.path.join( 'rawdata/target_ge.npy')
# CELL2ID_FILE = os.path.join('rawdata/cell2id.tsv')

SYNERGY_FILE = 'rawdata/oneil_synergy_loewe_cutoff30.txt'
DRUG_FEAT_FILE = 'rawdata/oneil_drug_feat.npy'
CELL_FEAT_FILE = 'rawdata/oneil_cell_feat.npy'

import numpy as np
import pandas as pd
data = pd.read_csv(SYNERGY_FILE, sep='\t', header=0)
data.columns = ['drugname1','drugname2','cell_line','synergy','fold']
drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
drugscount = len(drugslist)
cellslist = sorted(list(set(data['cell_line']))) 
cellscount = len(cellslist)


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help="n epoch")
parser.add_argument('--batch', type=int, default=256, help="batch size")
parser.add_argument('--gpu', type=int, default=1, help="cuda device")
parser.add_argument('--patience', type=int, default=100, help='patience for early stop')
parser.add_argument('--suffix', type=str, default='results_oneil_mgaedc100_folds', help="model dir suffix")
parser.add_argument('--hidden', type=int, nargs='+', default=[2048, 4096, 8192], help="hidden size")
parser.add_argument('--lr', type=float, nargs='+', default=[1e-3, 1e-4, 1e-5], help="learning rate")

args = parser.parse_args()
out_dir = os.path.join(OUTPUT_DIR, '{}'.format(args.suffix))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


log_file = os.path.join(out_dir, 'cv.log')
logging.basicConfig(filename=log_file,
                    format='%(asctime)s %(message)s',
                    datefmt='[%Y-%m-%d %H:%M:%S]',
                    level=logging.INFO)

save_args(args, os.path.join(out_dir, 'args.json'))
test_loss_file = os.path.join(out_dir, 'test_loss.pkl')

if torch.cuda.is_available() and (args.gpu is not None):
    gpu_id = args.gpu
else:
    gpu_id = None

##read in the topology data
from sklearn.preprocessing import StandardScaler

# cells_data = pd.read_csv('rawdata/cell2id.tsv', sep='\t', header=0)
# cellslist = list(cells_data['cell'])
# # cellslist_sorted = sorted(cellslist) 
# cellscount = len(cellslist)


###read the dataset
from torch.utils.data import Dataset
def read_map(map_file):
    d = {}
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split('\t')
            d[k] = int(v)
    return d

class FastSynergyDataset(Dataset):
    def __init__(self, drug_feat_file, drug_feat_topology_common, drug_feat_topology_specifics, cell_feat_file, synergy_score_file, use_folds, train=True):
        # self.drug2id = read_map(drug2id_file)
        # self.cell2id = read_map(cell2id_file)
        self.drug_feat1 = np.load(drug_feat_file)
        self.drug_feat2 = drug_feat_topology_common
        self.drug_feat3 = drug_feat_topology_specifics
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(drugslist)
        valid_cells = set(cellslist)
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        #drug1-drug2-cell
                        sample = [
                            torch.from_numpy(self.drug_feat1[drugslist.index(drug1)]).float(),
                            torch.from_numpy(self.drug_feat2[drugslist.index(drug1)]).float(),
                            torch.from_numpy(self.drug_feat3[cellname][drugslist.index(drug1)]).float(),
                            torch.from_numpy(self.drug_feat1[drugslist.index(drug2)]).float(),
                            torch.from_numpy(self.drug_feat2[drugslist.index(drug2)]).float(),
                            torch.from_numpy(self.drug_feat3[cellname][drugslist.index(drug2)]).float(),
                            torch.from_numpy(self.cell_feat[cellslist.index(cellname)]).float(),
                            torch.FloatTensor([float(score)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [drugslist.index(drug1), drugslist.index(drug2), cellslist.index(cellname), score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            ###drug2-drug1-cell
                            sample = [
                                torch.from_numpy(self.drug_feat1[drugslist.index(drug2)]).float(),
                                torch.from_numpy(self.drug_feat2[drugslist.index(drug2)]).float(),
                                torch.from_numpy(self.drug_feat3[cellname][drugslist.index(drug2)]).float(),
                                torch.from_numpy(self.drug_feat1[drugslist.index(drug1)]).float(),
                                torch.from_numpy(self.drug_feat2[drugslist.index(drug1)]).float(),
                                torch.from_numpy(self.drug_feat3[cellname][drugslist.index(drug1)]).float(),
                                torch.from_numpy(self.cell_feat[cellslist.index(cellname)]).float(),
                                torch.FloatTensor([float(score)]),
                            ]
                            self.samples.append(sample)
                            raw_sample = [drugslist.index(drug2), drugslist.index(drug1), cellslist.index(cellname), score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat1_len(self):
        return self.drug_feat1.shape[-1]

    def drug_feat2_len(self):
        return self.drug_feat2.shape[-1]

    def drug_feat3_len(self):
        return self.drug_feat2.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1_f1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d1_f2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        d1_f3 = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        d2_f1 = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        d2_f2 = torch.cat([torch.unsqueeze(self.samples[i][4], 0) for i in indices], dim=0)
        d2_f3 = torch.cat([torch.unsqueeze(self.samples[i][5], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][6], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][7], 0) for i in indices], dim=0)
        return d1_f1, d1_f2, d1_f3, d2_f1, d2_f2, d2_f3, c, y


##create model
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, drug_feat3_len:int, cell_feat_len:int, hidden_size: int):
        super(DNN, self).__init__()

        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, drug_feat2_len),
        )

        self.drug_network3 = nn.Sequential(
            nn.Linear(drug_feat3_len, drug_feat3_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat3_len*2),
            nn.Linear(drug_feat3_len*2, drug_feat3_len),
        )

        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len ),
            nn.Linear(cell_feat_len, 768),
        )

        self.fc_network = nn.Sequential(
            nn.BatchNorm1d(2*(drug_feat1_len + drug_feat2_len + drug_feat3_len)+ 768),
            nn.Linear(2*(drug_feat1_len + drug_feat2_len + drug_feat3_len)+ 768, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug1_feat3: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, drug2_feat3: torch.Tensor, cell_feat: torch.Tensor):
        drug1_feat1_vector = self.drug_network1( drug1_feat1 ) 
        drug1_feat2_vector = self.drug_network2( drug1_feat2 )
        drug1_feat3_vector = self.drug_network3( drug1_feat3 )
        drug2_feat1_vector = self.drug_network1( drug2_feat1 ) 
        drug2_feat2_vector = self.drug_network2( drug2_feat2 )
        drug2_feat3_vector = self.drug_network3( drug2_feat3 )
        cell_feat_vector = self.cell_network(cell_feat)
        # cell_feat_vector = cell_feat
        feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector,drug1_feat3_vector , drug2_feat1_vector, drug2_feat2_vector, drug2_feat3_vector, cell_feat_vector], 1)
        out = self.fc_network(feat)
        return out

class DNN_orig(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DNN_orig, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, cell_feat: torch.Tensor):
        feat = torch.cat([drug1_feat1, drug1_feat2, drug2_feat1, drug2_feat2, cell_feat], 1)
        out = self.network(feat)
        return out

#useful functions
def create_model(data, hidden_size, gpu_id=None):
    # model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    model = DNN(data.drug_feat1_len(), data.drug_feat2_len(), data.drug_feat2_len(), data.cell_feat_len(), hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    return model

def step_batch(model, batch, loss_func, gpu_id=None, train=True):
    if gpu_id is not None:
        batch = [x.cuda(gpu_id) for x in batch]
    drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = batch
    # if gpu_id is not None:
        # drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = drug1_feats1.cuda(gpu_id), drug1_feats2.cuda(gpu_id),drug1_feats3.cuda(gpu_id), drug2_feats1.cuda(gpu_id), drug2_feats2.cuda(gpu_id), drug2_feats3.cuda(gpu_id), cell_feats.cuda(gpu_id), y_true.cuda(gpu_id)
    if train:
        y_pred = model(drug1_feats1, drug1_feats2, drug1_feats3,drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
    else:
        yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2,drug2_feats3, cell_feats)
        yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2,drug1_feats3, cell_feats)
        y_pred = (yp1 + yp2) / 2
    loss = loss_func(y_pred, y_true)
    return loss


def train_epoch(model, loader, loss_func, optimizer, gpu_id=None):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, gpu_id)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def eval_epoch(model, loader, loss_func, gpu_id=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            loss = step_batch(model, batch, loss_func, gpu_id, train=False)
            epoch_loss += loss.item()
    return epoch_loss


def train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=False, mdl_dir=None):
    min_loss = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
        val_loss /= valid_loader.dataset_len
        if val_loss < min_loss:
            angry = 0
            min_loss = val_loss
            if sl:
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= patience:
                break
    if sl:
        model.load_state_dict(torch.load(find_best_model(mdl_dir)))
    return min_loss


def eval_model(model, optimizer, loss_func, train_data, test_data,
               batch_size, n_epoch, patience, gpu_id, mdl_dir):
    tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(tr_indices), batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(*train_data.tensor_samples(es_indices), batch_size=len(es_indices) // 4)
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data) // 4)
    train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, gpu_id,
                sl=True, mdl_dir=mdl_dir)
    test_loss = eval_epoch(model, test_loader, loss_func, gpu_id)
    test_loss /= len(test_data)
    return test_loss


#begin cross validation
topology_folder = 'results/results_loewe/results_mgaedc_representation'

n_folds = 1
n_delimiter = 60
test_losses = []
test_pccs = []
class_stats = np.zeros((n_folds, 7))

# for test_fold in range(n_folds):
test_fold = 0
valid_fold = list(range(10))[test_fold-1]
train_fold = [ x for x in list(range(10)) if x != test_fold and x != valid_fold ]
print(train_fold, valid_fold, test_fold)

mdl_dir = os.path.join(out_dir, str(test_fold))
if not os.path.exists(mdl_dir):
    os.makedirs(mdl_dir)

#common topology
drug_topology_common = pd.read_csv(topology_folder + '/results_embeddings_common_'+str(test_fold)+'.txt', sep='\t', header=None, index_col=0)
scaler = StandardScaler().fit(drug_topology_common.values)
drug_topology_common = scaler.transform(drug_topology_common.values)

#specific topology
d_topology_specifics = {}
for cellidx in range(cellscount):
    # cellidx = 0
    cellname = cellslist[cellidx]
    drug_topology = pd.read_csv(topology_folder + '/results_embeddings_specific_'+ str(cellidx) +'_'+str(test_fold)+'.txt', sep='\t', header=None, index_col=0)
    scaler = StandardScaler().fit(drug_topology.values)
    drug_topology_specific = scaler.transform(drug_topology.values)
    # np.save('results/drug_feat_topology_norm.npy', drug_feat_topology_norm)
    d_topology_specifics[cellname] = drug_topology_specific

logging.info("Outer: train folds {}, valid folds {} ,test folds {}".format(train_fold, valid_fold, test_fold))
logging.info("-" * n_delimiter)

best_hs, best_lr = args.hidden[2], args.lr[1]
logging.info("Best hidden size: {} | Best learning rate: {}".format(best_hs, best_lr))

##preprocess data
##preprocess data
train_data = FastSynergyDataset( DRUG_FEAT_FILE, drug_topology_common, d_topology_specifics, CELL_FEAT_FILE, SYNERGY_FILE, use_folds=train_fold)
valid_data = FastSynergyDataset( DRUG_FEAT_FILE, drug_topology_common, d_topology_specifics, CELL_FEAT_FILE, SYNERGY_FILE, use_folds=[valid_fold], train=False)
test_data = FastSynergyDataset( DRUG_FEAT_FILE, drug_topology_common, d_topology_specifics, CELL_FEAT_FILE, SYNERGY_FILE, use_folds=[test_fold], train=False)

# tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
train_loader = FastTensorDataLoader(*train_data.tensor_samples(), batch_size=args.batch, shuffle=True)
valid_loader = FastTensorDataLoader(*valid_data.tensor_samples(), batch_size=len(valid_data))
test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))

model = create_model(train_data, best_hs, gpu_id)
optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
loss_func = nn.MSELoss(reduction='sum')

##train
#train
min_loss = float('inf')
for epoch in range(1, args.epoch + 1):
    trn_loss = train_epoch(model, train_loader, loss_func, optimizer, gpu_id)
    trn_loss /= train_loader.dataset_len
    val_loss = eval_epoch(model, valid_loader, loss_func, gpu_id)
    val_loss /= valid_loader.dataset_len
    if epoch % 100 == 0: 
        print("epoch: {} | train loss: {} valid loss {}".format(epoch, trn_loss, val_loss))
    if val_loss < min_loss:
        min_loss = val_loss
        save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)

model.load_state_dict(torch.load(find_best_model(mdl_dir)))

##test predict
##test predict
##test predict
with torch.no_grad():
    for test_each in test_loader:
        test_each = [x.cuda(gpu_id) for x in test_each]
        drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = test_each
        yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2,drug2_feats3, cell_feats)
        yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2,drug1_feats3, cell_feats)
        y_pred = (yp1 + yp2) / 2
        test_loss = loss_func(y_pred, y_true).item()
        y_pred = y_pred.cpu().numpy().flatten()
        y_true = y_true.cpu().numpy().flatten()
        test_pcc = np.corrcoef(y_pred, y_true)[0, 1]
        test_loss /= len(y_true)
        y_pred_binary = [ 1 if x >= 30 else 0 for x in y_pred ]
        y_true_binary = [ 1 if x >= 30 else 0 for x in y_true ]
        roc_score = roc_auc_score(y_true_binary, y_pred)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
        auprc_score = auc(recall, precision)
        accuracy = accuracy_score( y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary)
        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)

class_stat = [roc_score, auprc_score, accuracy, f1, precision, recall, kappa]
class_stats[test_fold] = class_stat
test_losses.append(test_loss)
test_pccs.append(test_pcc)
logging.info("Test loss: {:.4f}".format(test_loss))
logging.info("Test pcc: {:.4f}".format(test_pcc))
logging.info("*" * n_delimiter + '\n')

##cal the stats in each cell line mse
from sklearn.metrics import mean_squared_error

all_data = pd.read_csv(SYNERGY_FILE, sep='\t', header=0)
test_data_orig = all_data[all_data['fold']==test_fold]
test_data_orig['pred'] = y_pred
test_data_orig.to_csv(out_dir + '/test_data_' + str(test_fold) + '.txt' ,sep='\t',header=True, index=False)
cells_stats = np.zeros((cellscount, 9))
for cellidx in range(cellscount):
    # cellidx = 0
    cellname = cellslist[cellidx]
    each_data = test_data_orig[test_data_orig['cell_line']== cellname ]
    each_true = each_data['synergy'].tolist()
    each_pred = each_data['pred'].tolist()
    each_loss = mean_squared_error(each_true, each_pred)
    each_pcc = np.corrcoef(each_pred, each_true)[0, 1]
    #class
    each_pred_binary = [ 1 if x >= 30 else 0 for x in each_pred ]
    each_true_binary = [ 1 if x >= 30 else 0 for x in each_true ]
    roc_score_each = roc_auc_score(each_true_binary, each_pred)
    precision, recall, _ = precision_recall_curve(each_true_binary, each_pred_binary)
    auprc_score_each = auc(recall, precision)
    accuracy_each = accuracy_score(each_true_binary, each_pred_binary)
    f1_each = f1_score(each_true_binary, each_pred_binary)
    precision_each = precision_score(each_true_binary, each_pred_binary, zero_division=0)
    recall_each = recall_score(each_true_binary, each_pred_binary)
    kappa_each = cohen_kappa_score(each_true_binary, each_pred_binary)
    t = [each_loss, each_pcc, roc_score_each, auprc_score_each, accuracy_each, f1_each, precision_each, recall_each, kappa_each]
    cells_stats[cellidx] = t

pd.DataFrame(cells_stats).to_csv(out_dir+'/test_data_cells_stats_'+str(test_fold)+'.txt', sep='\t', header=None, index=None)


logging.info("CV completed")
with open(test_loss_file, 'wb') as f:
    pickle.dump(test_losses, f)
mu, sigma = calc_stat(test_losses)
logging.info("MSE: {:.4f} ± {:.4f}".format(mu, sigma))
lo, hi = conf_inv(mu, sigma, len(test_losses))
logging.info("Confidence interval: [{:.4f}, {:.4f}]".format(lo, hi))
rmse_loss = [x ** 0.5 for x in test_losses]
mu, sigma = calc_stat(rmse_loss)
logging.info("RMSE: {:.4f} ± {:.4f}".format(mu, sigma))
pcc_mean, pcc_std = calc_stat(test_pccs)
logging.info("pcc: {:.4f} ± {:.4f}".format(pcc_mean, pcc_std))

class_stats = np.concatenate([class_stats, class_stats.mean(axis=0, keepdims=True), class_stats.std(axis=0, keepdims=True)], axis=0)
pd.DataFrame(class_stats).to_csv(out_dir + '/class_stats.txt', sep='\t', header=None, index=None)


