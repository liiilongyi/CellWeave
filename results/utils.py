import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from preprocess import pca
import matplotlib.pyplot as plt
import torch
from sklearn.mixture import GaussianMixture

#os.environ['R_HOME'] = '/scbio4/tools/R/R-4.0.3_openblas/R-4.0.3'    

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    X = np.asarray(adata.obsm[used_obsm], dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{used_obsm} should be a 2D array, got shape {X.shape}")

    # 简单把 mclust 的 EEE 映射为 sklearn 的 'full' 协方差
    cov_type = "full"
    # 你如果以后要区分不同 modelNames，可以在这里扩展映射关系

    gmm = GaussianMixture(
        n_components=num_cluster,
        covariance_type=cov_type,
        random_state=random_seed
    )

    labels = gmm.fit_predict(X)

    adata.obs['mclust'] = labels.astype(int)
    # 如果后面要当作类别用，也可以：
    # adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='spaLLM', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']
       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()


def add_noise_by_zeroing(matrix, zero_prob):
    """
    给定矩阵，将一定比例的元素置零作为噪声。

    参数:
    - matrix (torch.Tensor): 输入矩阵。
    - zero_prob (float): 每个元素置零的概率，范围为 [0, 1]。

    返回:
    - 加噪后的矩阵。
    """
    # 生成与输入矩阵相同形状的掩码，按概率zero_prob确定哪些元素置零
    noise_mask = torch.bernoulli((1 - zero_prob) * torch.ones_like(matrix)).to(matrix.device)
    # 将掩码应用于矩阵，将随机元素置零
    noisy_matrix = matrix * noise_mask
    return noisy_matrix


def add_noise_by_zeroing_columns(matrix, zero_prob=0.1):
    """
    给定矩阵，随机选择一定比例的列，并将整列置零作为噪声。

    参数:
    - matrix (torch.Tensor): 输入矩阵。
    - zero_prob (float): 将列置零的概率，范围为 [0, 1]。

    返回:
    - 加噪后的矩阵。
    """
    # 获取矩阵的形状
    num_columns = matrix.size(1)  # 获取列数

    # 生成随机列索引，确定哪些列置零
    zero_mask = torch.bernoulli((1 - zero_prob) * torch.ones(num_columns)).to(matrix.device)

    # 创建与矩阵同形状的掩码
    column_mask = zero_mask.unsqueeze(0).expand_as(matrix)  # 将列掩码扩展到与矩阵同形状

    # 将选中的列置零
    noisy_matrix = matrix * column_mask
    return noisy_matrix


def add_gaussian_noise(matrix, mean=0.0, std=0.001):
    """
    给定矩阵，按元素添加高斯噪声。

    参数:
    - matrix (torch.Tensor): 输入矩阵。
    - mean (float): 高斯噪声的均值。
    - std (float): 高斯噪声的标准差。

    返回:
    - 加噪后的矩阵。
    """
    # 生成与输入矩阵相同形状的噪声
    noise = torch.normal(mean=mean, std=std, size=matrix.size()).to(matrix.device)
    # 将噪声添加到矩阵上
    noisy_matrix = matrix + noise
    return noisy_matrix