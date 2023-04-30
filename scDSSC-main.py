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
sys.path.append("./scDSSC-main")
from layers import ZINBLoss, MeanAct, DispAct
from evaluation import eva
from AutoEncoder_ZINB import AE
from preprocess import read_dataset, normalize
import scanpy as sc
from utils import best_map, thrC, post_proC
from scDSSC_ZINB import Deep_Sparse_Subspace_Clustering

pb3k = sc.read_h5ad("data/pbmc3k_raw.h5ad") #read in test dataset

def scDSSClustering(adata, results_file="results.h5ad"):
  adata.obs['Group'] = y
  print(y)
  #adata = read_dataset(adata, #functions from scDSSC
  #                transpose=False,
  #                test_split=False,
  #                copy=True)
  
  adata = adata.copy()
  
  adata = normalize(adata,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True,
                  select_hvg=True)
  
  print(adata.X.shape)
  print(y.shape)
  x_sd = adata.X.std(0)
  x_sd_median = np.median(x_sd)
  print("median of gene sd: %.5f" % x_sd_median)
  sd = 2.5
  

  net = Deep_Sparse_Subspace_Clustering(n_enc_1=256, n_enc_2=32, n_dec_1=32, n_dec_2=256, n_input=2000,
                                        n_z=10, denoise=False, sigma=2.0, pre_lr=0.002, alt_lr=0.001,
                                        adata=adata, pre_epoches=200, alt_epoches=100, lambda_1=1.0, lambda_2=0.5)
  Coef_1 = net.pre_train()
  Coef_2 = net.alt_train()
  Coef = thrC(Coef_2, ro=1.0)
  pred_label, _ = post_proC(Coef, 14, 11, 7.0)
  y = y.astype(np.int64)
  pred_label = pred_label.astype(np.int64)
  acc, nmi, ari = eva(y, pred_label) #evaluate acc, nmi, ari
  
  return adata, nmi, ari

pb3k_adj, pb3k_nmi, pb3k_ari = scDSSClustering(pb3k)


  
