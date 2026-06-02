# CellWeave: Reconstructing Cells in Visium HD by Weaving Spatial Transcriptomics and Morphology


## Overview

Visium HD enables near single-cell molecular mapping, yet measurements remain fundamentally bin-based and lack explicit cell boundaries, limiting cell-resolved analysis. Accurate reconstruction of single cells from such data is challenging because complete cellular contours are not directly observed in any modality alone: histological images provide structural context but incomplete boundaries, whereas transcriptomic bins are sparse, locally mixed and not intrinsically organized into cells. Here we present CellWeave, a multimodal graph framework that reframes cell reconstruction from image segmentation or nucleus-centered expansion into bin-level cellular co-membership inference. CellWeave learns local multimodal affinities between neighboring bins from histological and transcriptomic features, and assembles bins into cell instances through seed-guided graph partitioning. A mixture-of-experts module adaptively balances modality contributions across tissue contexts, enabling flexible integration of heterogeneous signals. To address the lack of ground-truth cell boundaries in native Visium HD data, we establish a multi-perspective evaluation framework combining geometric fidelity, transcriptomic consistency and relative agreement analysis, and further introduce a Xenium-derived simulated Visium HD benchmark with boundary-resolved labels for direct quantitative assessment. Across Xenium-derived benchmarks and native Visium HD datasets from mouse brain and human cancers, CellWeave consistently improved boundary-level accuracy, cell-level molecular concordance and downstream cell-state resolution compared with existing segmentation and bin-to-cell reconstruction methods. CellWeave enables robust cell-resolved reconstruction from high-resolution spatial transcriptomics, supporting downstream analyses including fine-grained cell annotation, spatial organization and tissue microenvironment characterization. 

## Main Workflow

### 1. Preprocess Visium HD bin expression

Use [`preprocess.py`](preprocess.py) to read 10x Visium HD bin outputs, align the 10x tissue-position parquet file with the expression matrix, keep in-tissue bins, and save a bin-level `.h5ad` file.

Edit the data root and output file in:

```python
root = Path(r"<LOCAL_DATA_OR_OUTPUT_PATH>")
out_path = "<BIN_EXPRESSION_H5AD>"
```

Run:

```bash
python preprocess.py
```

Expected output:

- bin-level expression `.h5ad`
- optional QC figures for bin positions and zero-expression bins

### 2. Assign nucleus-derived seed labels

Use [`seed.py`](seed.py) to assign seed labels to bins from a Cellpose or StarDist nucleus segmentation mask.

Edit:

```python
path = "<NUCLEUS_SEGMENTATION_MASK_NPY>"
adata = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")
scale = 1
diameter = 16
```

Run:

```bash
python seed.py
```

Expected output:

- bin-level `.h5ad` with `adata.obs["seed"]`

Notes:

- Cellpose output is expected as a dictionary-like `.npy` containing `masks`.
- StarDist output can be a plain 2D label array.
- Bins not confidently assigned to a nucleus are labeled `-1`.

### 3. Build neighboring-bin edge labels

Use [`label.py`](label.py) to generate 8-neighborhood bin-bin edge labels from seed labels.

Edit:

```python
adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
edge_df.to_csv("<EDGE_LABELS_CSV>", index=False)
```

Run:

```bash
python label.py
```

Edge labels:

- `1`: two neighboring bins have the same seed label
- `-1`: two neighboring bins have different seed labels

Use [`cutlabel.py`](cutlabel.py) to downsample positive edges while keeping all negative edges:

```bash
python cutlabel.py
```

Edit the input and output CSV names at the top of the file:

```python
input_file = "<EDGE_LABELS_CSV>"
output_file = "<EDGE_LABELS_CSV>"
```

### 4. Train MoE + GCN edge classifier

The model architecture is defined in [`model.py`](model.py). The training and full-graph inference workflow is in [`train_MoE_ear.py`](train_MoE_ear.py).

The model combines:

- gene embeddings
- image embeddings
- a modality Mixture-of-Experts module
- a GCN edge classifier

Before running, edit the input paths and output names near the top and bottom of [`train_MoE_ear.py`](train_MoE_ear.py), especially:

```python
IMG_BARCODE_PATH = "<IMAGE_EMBEDDING_BARCODES_NPY>"
IMG_MATRIX_PATH = "<IMAGE_EMBEDDING_MATRIX_NPY>"
adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
emb_path = "<EMBEDDING_NPY>"
edge_df = pd.read_csv("<EDGE_LABELS_CSV>")
```

Run:

```bash
python train_MoE_ear.py
```

Expected outputs:

- trained model weights, e.g. `<MODEL_WEIGHTS_PT>`
- edge probabilities `.npz`
- MoE weights `.npy`
- training curve `.csv` / `.pdf`

### 5. Segment bins by seeded MSF / graph growing

Use [`seg.py`](seg.py) to convert predicted edge probabilities into final cell labels.

Edit:

```python
data = np.load("<EDGE_PROBABILITY_NPZ>", allow_pickle=True)
adata = sc.read_h5ad("<BIN_EXPRESSION_WITH_SEEDS_H5AD>")
alpha_thresh = 0.3
k_radius = 2.5
```

Run:

```bash
python seg.py
```

Expected output:

- final segmentation labels written to `adata.obs`
- updated `.h5ad`
- optional segmentation mask visualization

### 6. Evaluate segmentation

Use [`metric.py`](metric.py) to evaluate predicted cell labels against ground-truth labels from a parquet file.

Edit:

```python
adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>", backed="r")
pred_col = "<PREDICTED_SEGMENTATION_COLUMN>"
gt_path = r"<GROUND_TRUTH_CELL_LABELS_PARQUET>"
```

Run:

```bash
python metric.py
```

Reported metrics include:

- IoU matching
- precision / recall / F1
- SQ / RQ / PQ

### 7. Aggregate to cell-level AnnData

Use [`ds.py`](ds.py) to aggregate bin-level expression into cell-level expression according to a segmentation label column.

Edit:

```python
adata = sc.read_h5ad("<SEGMENTED_BIN_H5AD>")
adata2 = sc.read_h5ad("<BIN_EXPRESSION_H5AD>")
label_col = "CellWeave"
output = "<CELL_LEVEL_EXPRESSION_H5AD>"
```

Run:

```bash
python ds.py
```

Expected output:

- cell-level expression `.h5ad`
- cell centroid coordinates in `obs["px"]` and `obs["py"]`

## Downstream Analysis

The [`results/`](results) folder contains scripts used for downstream analysis and visualization, including:

- cell-level spatial embedding and clustering
- UMAP and spatial visualization
- heatmaps and marker dot plots
- neighborhood enrichment analysis
- ligand-receptor / CellChat-like analysis
- MapMyCells-related comparison

These scripts are kept close to the original experiment code. Many of them contain dataset-specific paths and should be edited before use.

## Dependencies

The original experiments used Python 3.10 with CUDA-enabled PyTorch. A lightweight dependency list is provided in [`requirements.txt`](requirements.txt).

Install the core Python dependencies with:

```bash
pip install -r requirements.txt
```

## Acknowledgements

We thank the developers of [Cellpose](https://www.cellpose.org/), [StarDist](https://github.com/stardist/stardist), [scGPT](https://github.com/bowang-lab/scGPT), [UNI](https://github.com/mahmoodlab/UNI), [SpatialGlue](https://github.com/JinmiaoChenLab/SpatialGlue), and [CellChat](https://github.com/jinworks/CellChat) for their valuable open-source contributions.
