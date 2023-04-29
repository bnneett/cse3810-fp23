import numpy as np
import pandas as pd
import scanpy as sc

#This scanpy base implementation follows the official tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
sc.settings.verbosity = 3 #hints
sc.logging.print_header() #print stats
sc.settings.set_figure_params(dpi=80, facecolor='white')

pb3k = sc.read_h5ad("data/pbmc3k_raw.h5ad") #read tutorial dataset into anndata object
#adata = sc.read_h5ad("path/to/file.h5ad")

def scanpycluster(adata, mode=0, results_file="results.h5ad"):
  #preprocess data
  #sc.pl.highest_expr_genes(adata, n_top=25) #plots genes with highest mean expression across all cells (top 20)
  sc.pp.filter_cells(adata, min_genes=200) #basic filtering
  sc.pp.filter_genes(adata, min_cells=5)
  adata.var['mt'] = adata.var_names.str.startswith('MT-')  #annotate the group of mitochondrial genes as 'mt'
  sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) #quality control metrics
  #sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True) #violin plot of qc metrics
  #sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt') #scatter plots of metrics
  #sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')#used to visually evaluate parameters for slicing
  
  adata = adata[adata.obs.n_genes_by_counts < 2500, :]#slice data object
  adata = adata[adata.obs.pct_counts_mt < 5, :]#remove cells that have too many mitochondrial genes expressed or too many total counts
  
  sc.pp.normalize_total(adata, target_sum=1e4) #total-count normalize (library-size correct) the data matrix X to 10,000 reads per cell
  sc.pp.log1p(adata) #logarithmize the data
  sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5) #identify highly variable genes
  #sc.pl.highly_variable_genes(adata)
  
  adata.raw = adata #freeze state of data object
  adata = adata[:, adata.var.highly_variable] #filter highly variable genes
  sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt']) #regress out effects of total counts per cell and percentage of mitochondrial genes expressed, scale data to unit variance
  sc.pp.scale(adata, max_value=10) #scale each gene to unit variance, clip values exceeding standard deviation 10
  
  #principal component analysis
  sc.tl.pca(adata, svd_solver='arpack') #run pca
  #sc.pl.pca_variance_ratio(adata, log=True) #this gives us information about how many PCs we should consider in order to compute the neighborhood relations of cells
  adata.write(results_file) #save result(?)
  
  #computing the neighborhood graph
  sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40) #parameter values taken from tutorial page
  
  #embedding the neighborhood graph
  sc.tl.umap(adata)
  sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP']) #plot raw (uncorrected) gene expression
  
  #clustering the neighborhood graph (leiden/louvain)
  if mode==0:
    sc.tl.leiden(adata) #cluster using leiden method, which takes the neighborhood graph we just calculated
    sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7']) #plot results
    adata.write(results_file)
  elif mode==1:
    sc.tl.louvain(adata) #cluster using louvain method (requires having run neighbors first)
    sc.pl.umap(adata, color=['louvain', 'CST3', 'NKG7']) #plot results
    adata.write(results_file)
  else:
    raise ValueError("Mode must be set to 0 for Leiden clustering or 1 for Louvain clustering")
    
  #return updated data object
  return adata

#call the function as such:
#pb3k_adj = scanpycluster(pb3k, mode=0, results_file="pb3k_results.h5ad")
#pb3k_adj_lvn = scanpycluster(pb3k, mode=1, results_file="pb3k_lvn_results.h5ad")
  
  
