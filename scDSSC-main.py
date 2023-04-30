import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.autograd import Variable
import h5py
import numpy as np
import sys
sys.path.append('./scDSSC_main')
from layers import ZINBLoss, MeanAct, DispAct
from evaluation import eva
from AutoEncoder_ZINB import AE
from preprocess import read_dataset, normalize
import scanpy as sc
from utils import best_map, thrC, post_proC
from scDSSC_ZINB import Deep_Sparse_Subspace_Clustering


def h5_helper(file): #takes .h5 file and outputs adata object (as opposed to taking an .h5ad file and outputting an adata object)
  data_mat = h5py.File(file)
  x = np.array(data_mat['X'])
  y = np.array(data_mat['Y'])
  data_mat.close()
  print(x.shape)
  print(y.shape)

  adata = sc.AnnData(x)
  adata.obs['Group'] = y
  
  return adata
  

def scDSSC(adata): #this function takes an anndata object (so need to convert from h5 if have h5)
  #preprocessing
  x = adata.X
  y = adata.obs['Group'].values
  print(adata)
  adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
  adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True, select_hvg=True)
  print(adata.X.shape)
  print(y.shape)
  x_sd = adata.X.std(0)
  x_sd_median = np.median(x_sd)
  print("median of gene sd: %.5f" % x_sd_median)
  sd = 2.5
  
  #clustering
  net = Deep_Sparse_Subspace_Clustering(n_enc_1=256, n_enc_2=32, n_dec_1=32, n_dec_2=256, n_input=2000,
                                      n_z=10, denoise=False, sigma=2.0, pre_lr=0.002, alt_lr=0.001,
                                      adata=adata, pre_epoches=200, alt_epoches=100, lambda_1=1.0, lambda_2=0.5)
  
  Coef_1 = net.pre_train() #learns a set of weights for the network that can be used to initialize the weights for fine-tuning
  Coef_2 = net.alt_train() #performs fine-tuning on the input data using the learned weights from pre-training
  Coef = thrC(Coef_2, ro=1.0) 
  pred_label, _ = post_proC(Coef, 14, 11, 7.0) #takes the thresholded weight matrix Coef and a set of hyperparameters as inputs 
                                               #and produces a set of predicted labels for the input data (all hyperparameters are set as in the paper)
  y = y.astype(np.int64)
  pred_label = pred_label.astype(np.int64)
  
  eva(y, pred_label) #evaluate metrics (nmi, ari) using true labels y and predicted labels pred_label
  


pb3k = sc.read_h5ad("data/pbmc3k_raw.h5ad") #read in test dataset
liver5k = h5_helper("data/HumanLiver_counts_top5000.h5") #read in h5 test set

scDSSC(liver5k)
