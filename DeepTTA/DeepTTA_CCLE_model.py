import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import codecs
from subword_nmt.apply_bpe import BPE
import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
import collections
import os
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import time
import pickle
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler
from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE

sys.path.append('D:/moon/DeepTTC/')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import model_helper
import Step1_getData
#import Step2_DataEncoding
#import Step3_model2
from Step2_DataEncoding import DataEncoding
from Step3_model2 import DeepTTC


def drug2emb_encoder(smile):
    vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


os.chdir('D:/moon/DeepTTC/')

edk11 = pd.read_csv("EDC-11K/EDC11K_DRUG.csv")
edk11_rna = pd.read_csv("EDC-11K/edc11k_rna.csv")

vocab_dir = 'D:/moon/DeepTTC/'
obj = DataEncoding(vocab_dir=vocab_dir)

edk11['drug_encoding'] = [drug2emb_encoder(i) for i in edk11["smiles"]]
edk11_rna2 = edk11_rna[edk11['CellName']]

a = np.load("EDC-11K/EDC11K_datase_r2_9_1.npz")
at = a['train']
ate = a['test']


train_data = edk11.iloc[at,:]
test_data = edk11.iloc[ate,:]
train_data['Label'] = train_data['LN_IC50']
test_data['Label'] = test_data['LN_IC50']
train_data.index = range(train_data.shape[0])
test_data.index = range(test_data.shape[0])

train_rna = edk11_rna2.iloc[:,at]
test_rna = edk11_rna2.iloc[:,ate]
train_rnadata = train_rna.T
test_rnadata = test_rna.T
train_rnadata.index = range(train_rnadata.shape[0])
test_rnadata.index = range(test_rnadata.shape[0])

print(train_data)
print(train_rnadata)
print(test_data)
print(test_rnadata)


print("Data generator")

torch.cuda.synchronize()
model = DeepTTC('D:/moon/DeepTTC/')

model.train(train_drug=train_data, train_rna=train_rnadata,
          val_drug=train_data, val_rna=train_rnadata)

y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, Cl = model.predict(test_data, test_rnadata)


print('mse : ', mse)
print('rmse : ', rmse)
print('person : ', person)
print('p_val : ', p_val)
print('spearman : ', spearman)
print('s_p_val : ', s_p_val)
print('Cl : ', Cl)

result_df = pd.DataFrame(y_label, y_pred)

result_df.to_csv("result.csv")

modeldir = 'D:/moon/DeepTTC/Model_80'
modelfile = modeldir + '/model.pt'
if not os.path.exists(modeldir):
    os.mkdir(modeldir)

model.save_model()
print("Model Saveed :{}".format(modelfile))

























