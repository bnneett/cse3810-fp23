import numpy as np
import pandas as pd
import scanpy as sc

#This scanpy base implementation follows the official tutorial: https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
sc.settings.verbosity = 3 #hints
sc.logging.print_header() #print stats
sc.settings.set_figure_params(dpi=80, facecolor='white')

pb3k = sc.read_h5ad("data/pbmc3k_raw.h5ad") #read tutorial dataset into anndata object
hliver = sc.read_10x_h5("data/HumanLiver_counts_top5000.h5") #extra test dataset
#adata = sc.read_h5ad("path/to/file.h5ad")

def scanpycluster(adata, mode=0):
  #preprocess data
  sc.pl.highest_expr_genes(pb3k, n_top=25).savefig("{}-f1.pdf".format(adata)) #plots genes with highest mean expression across all cells (top 20)
  sc.pp.filter_cells(pb3k, min_genes=200) #basic filtering
  sc.pp.filter_genes(p3bk, min_cells=5)
  adata.var['mt'] = adata.var_names.str.startswith('MT-')  #annotate the group of mitochondrial genes as 'mt'
  sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True) #quality control metrics
  #sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True).savefig("pb3k-f2.pdf") #violin plot of qc metrics
  sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt').savefig("{}-f2.pdf".format(adata)) #scatter plots of metrics
  sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts').savefig("{}-f3.pdf".format(adata))#used to visually evaluate parameters for slicing
  
  adata = adata[adata.obs.n_genes_by_counts < 2500, :]#slice data object
  adata = adata[adata.obs.pct_counts_mt < 5, :]#remove cells that have too many mitochondrial genes expressed or too many total counts
  
  sc.pp.normalize_total(adata, target_sum=1e4) #total-count normalize (library-size correct) the data matrix X to 10,000 reads per cell
  sc.pp.log1p(adata) #logarithmize the data
  sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5) #identify highly variable genes
  sc.pl.highly_variable_genes(adata).savefig("{}-f4.pdf".format(adata))
  
  adata.raw = adata #freeze state of data object
  adata = adata[:, adata.var.highly_variable] #filter highly variable genes
  sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt']) #regress out effects of total counts per cell and percentage of mitochondrial genes expressed, scale data to unit variance
  sc.pp.scale(adata, max_value=10) #scale each gene to unit variance, clip values exceeding standard deviation 10
  
  #principal component analysis
  sc.tl.pca(adata, svd_solver='arpack') #run pca
  sc.pl.pca_variance_ratio(adata, log=True).savefig("{}-f5.pdf".format(adata)) #this gives us information about how many PCs we should consider in order to compute the neighborhood relations of cells
  adata.write(results_file) #save result(?)
  
  #computing the neighborhood graph
  sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40) #parameter values taken from tutorial page
  
  #embedding the neighborhood graph
  sc.tl.umap(adata)
  sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP']).savefig("{}-f6.pdf".format(adata)) #plot raw (uncorrected) gene expression
  
  #clustering the neighborhood graph (leiden/louvain)
  if mode==0:
    sc.tl.leiden(adata) #cluster using leiden method, which takes the neighborhood graph we just calculated
    sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7']).savefig("{}-f7.pdf".format(adata)) #plot results
    adata.write(results_file)
  elif mode==1:
    sc.tl.louvain(adata) #cluster using louvain method (requires having run neighbors first)
    sc.pl.umap(adata, color=['louvain', 'CST3', 'NKG7']).savefig("{}-f7.pdf".format(adata)) #plot results
    adata.write(results_file)
  else:
    raise ValueError("Mode must be set to 0 for Leiden clustering or 1 for Louvain clustering")
    
  #return updated data object
  return adata

pb3k_adj = scanpycluster(pb3k, mode=0)
hliver_adj = scanpycluster(hliver, mode=0)
  
  
