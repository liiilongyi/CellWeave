import numpy as np
import scanpy as sc
import re

adata = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")
#adata = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")


adata2 = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")


adata.var['gene'] = adata.var['gene'].astype(str)
adata.var_names = adata.var['gene']
adata.var_names_make_unique()

adata2.var['gene'] = adata.var['gene'].astype(str)
adata2.var_names = adata.var['gene']
adata2.var_names_make_unique()

# 确保都是字符串
adata2.var_names = adata2.var_names.astype(str)
# 直接把 adata2 的基因名复制给 adata
adata.var_names = adata2.var_names.copy()

#adata.var.index.name = None
adata.var = adata.var.drop(columns=["gene"])
print(adata)

adata.write_h5ad("<SEGMENTED_BIN_H5AD>")

#adata2 = sc.read_h5ad("<CELL_LEVEL_EXPRESSION_H5AD>")
#print(adata2.var_names)