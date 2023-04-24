import numpy as np
import pandas as pd
import scanpy as sc

pb3k = sc.read_h5ad("data/pbmc3k_raw.h5ad") #read dataset into anndata object
#adata = sc.read_h5ad("path/to/file.h5ad")

#step 1: preprocess data
sc.pl.highest_expr_genes(pb3k, n_top=25) #plots genes with highest mean expression across all cells (top 20)
sc.pp.filter_cells(pb3k, min_genes=200)#basic filtering
sc.pp.filter_genes(p3bk, min_cells=5)
sc.pl.highest_expr_genes(pb3k, n_top=25)
adata.var['mt'] = p3bk.var_names.str.startswith('MT-')  #annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(p3bk, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) #quality control metrics

#look for outliers in the following scatter plots:
sc.pl.scatter(p3bk, x='total_counts', y='pct_counts_mt') #x axis
sc.pl.scatter(p3bk, x='total_counts', y='n_genes_by_counts') #y axis


#step 2: cluster data
#step 3: visualize clusters
#step 4: evaluate data
