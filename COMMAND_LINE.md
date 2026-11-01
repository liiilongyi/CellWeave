## Command-line workflow

CellWeave can also be run step by step from a terminal.

### 1. Prepare the input files and model weights

Follow the **Installation**, **Pretrained Models**, and **Input Data**
sections above. Edit `pipeline_config.json` in the repository root and
set the following fields to your input paths:

- `SAMPLE_ID`: sample name used in output filenames.
- `IMAGE_PATH`: path to the H&E image.
- `POS_PATH`: path to `tissue_positions.parquet`.
- `EXPR_FILTERED_H5`: path to `filtered_feature_bc_matrix.h5`.
- `EXPR_RAW_H5`: path to `raw_feature_bc_matrix.h5`.

Paths should be specified relative to the repository root or as absolute
paths.

### 2. Run the pipeline

Run the following commands from the repository root:

```bash
(
    set -e

    mkdir -p curve figure stardist_npy weight/MoE

    # Preprocess the Visium HD data
    python preprocess.py

    # Extract transcriptomic features with scGPT
    (cd scGPT && python example.py)

    # Extract morphology features with UNI
    (cd UNI && python uni.py)

    # Detect nuclei with StarDist
    python run_stardist.py

    # Generate nucleus seeds and edge labels
    python seed.py
    python label.py
    python cutlabel.py

    # Train the model
    python train.py

    # Assemble cell instances
    python seg.py
)
