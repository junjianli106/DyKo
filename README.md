
---

# DyKo: Dynamic Knowledge-guided Vision-Language Model for WSI Classification

**DyKo** is a **Dy**namic **K**n**o**wledge-guided Multiple Instance Learning (MIL) framework designed for gigapixel-scale Whole Slide Image (WSI) analysis. It adapts universal pathological knowledge to slide-specific visual evidence, enabling superior performance in few-shot WSI classification tasks.

> **Abstract:**
> Multiple Instance Learning (MIL) has emerged as the dominant paradigm for the analysis of gigapixel-scale Whole Slide Images (WSIs). However, recent methods leveraging guidance from Vision-Language Models often rely on static and universal pathological descriptions. This one-size-fits-all strategy fails to account for the vast morphological heterogeneity within individual WSIs. To address this, we propose **DyKo**, a Dynamic Knowledge-guided MIL framework.
> The core of DyKo is the **WSI-Adaptive Knowledge Instantiation (WAKI)** module. WAKI identifies key visual prototypes within a specific WSI's histology and uses them as queries to retrieve relevant concepts from a pathology knowledge base. To ensure fidelity, we introduce a **Structural Consistency loss** that enforces alignment between knowledge-instantiated and visual features. Comprehensive experiments on four public cancer datasets demonstrate that DyKo achieves superior performance over state-of-the-art methods in few-shot pathology diagnosis.

---

[//]: # ()
[//]: # (## 📰 News)

[//]: # ()
[//]: # (* **[2024.XX.XX]** The code and configuration files for DyKo are released.)

[//]: # (* **[2024.XX.XX]** Paper accepted to [Conference Name]! 🎉)

[//]: # (---)

[//]: # ()
[//]: # (## 🏗️ Framework Overview)

[//]: # ()
[//]: # (DyKo addresses the limitations of static text prompts in VLM-based pathology by introducing:)

[//]: # ()
[//]: # (1. **Visual-Language Alignment:** Leveraging pre-trained VLMs &#40;TITAN/CONCH&#41;.)

[//]: # (2. **Dynamic Concept Retrieval &#40;WAKI&#41;:** Retrieving slide-specific text concepts based on visual prototypes.)

[//]: # (3. **Dual-Stream Attention:** Fusing visual features and instantiated knowledge features.)

[//]: # (4. **Structural Consistency:** Preventing semantic drift during feature synthesis.)

[//]: # ()
[//]: # (---)

## 🛠️ Installation

### Prerequisites

* Linux
* Python 3.11+
* NVIDIA GPU 

### Step-by-Step Setup

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/DyKo.git
cd DyKo

```


2. **Create Conda Environment**
```bash
conda create -n DyKo python==3.11
conda activate DyKo

```


3. **Install Dependencies**
```bash
# Basic requirements
for req in $(cat requirements_few_shot.txt); do pip install "$req"; done

# Install PyTorch (adjust cuda version if necessary)
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

```


4. **Download Pre-trained VLM Weights**
* **TITAN:** Automatically downloaded via HuggingFace (`MahmoodLab/TITAN`).
* **CONCH:** Please download `conch.pth` manually from [MahmoodLab/CONCH](https://github.com/mahmoodlab/CONCH) and place it in the `ckpts/` folder.



---

## 📂 Data Preparation

DyKo requires two streams of data: **Visual Features** (from WSIs) and **Knowledge Features** (from text).

### 1. Dataset Downloads

Please download the raw WSIs from their respective sources:

* **TCGA-NSCLC / RCC:** [NIH Genomic Data Commons](https://www.google.com/search?q=https://portal.gdc.cancer.gov/)
* **CAMELYON16:** [CAMELYON16 Challenge](https://camelyon16.grand-challenge.org/)
* **UBC-OCEAN:** [Kaggle](https://www.google.com/search?q=https://www.kaggle.com/competitions/UBC-OCEAN)

### 2. Concept Knowledge Base (Text)

We use Quilt1M to build a pathology concept bank.

```bash
python process_quilt1m.py
# Output: concepts/text_embeddings_<DATASET>_1000.pt (Shape: [n_concepts, feature_dim])

```

### 3. Visual Feature Extraction

We recommend using [CLAM](https://www.google.com/search?q=https://github.com/mahmoodlab/CLAM) for tissue segmentation and patching.

* **Patch Size:** 448
* **Magnification:** 20x 
* **Encoder:** CONCH or TITAN

**Directory Structure for Features:**

```
data/
└── wsi_features/
    ├── CAMELYON16/
    │   └── feats-m20-s448-conch_v1_5/
    │       └── pt_files/
    │           ├── slide_1.pt  # Contains 'features' [N, 512]
    │           └── ...
    └── TCGA-NSCLC/
        └── ...

```

### 4. Split Files

Prepare CSV splits for few-shot learning in `splits/4foldcls/`.
**Format:** `train_slide_id, train_label, val_slide_id, val_label, test_slide_id, test_label`

---

## 🚀 Usage

### Training

You can run training using the provided bash scripts or directly via Python.

**Option 1: Using Scripts (Recommended)**

```bash
cd scripts/Cls
bash run_fold0.sh

```

**Option 2: Direct Command**

```bash
python train.py \
  --stage='train' \
  --config="Cls/CAMELYON16/DyKo/DyKo.yaml" \
  --gpus=0 \
  --fold=0 \
  --task='cls' \
  --seed=2025 \
  --n_shot=16

```

### Evaluation (Testing)

To test a trained model:

```bash
python train.py \
  --stage='test' \
  --config="Cls/CAMELYON16/DyKo/DyKo.yaml" \
  --gpus=0 \
  --fold=0 \
  --task='cls' \
  --n_shot=16

```

[//]: # ()
[//]: # (### Key Arguments)

[//]: # ()
[//]: # (| Argument | Description | Default |)

[//]: # (| --- | --- | --- |)

[//]: # (| `--config` | Path to the YAML configuration file | Required |)

[//]: # (| `--n_shot` | Number of shots for few-shot learning | `16` |)

[//]: # (| `--fold` | Cross-validation fold &#40;0-3&#41; | `0` |)

[//]: # (| `--task` | Task type &#40;`cls` or `prog`&#41; | `cls` |)

[//]: # (| `--grad_acc` | Gradient accumulation steps | `4` |)

[//]: # ()
[//]: # (> **Note on Reproducibility:** Due to the nature of few-shot sampling, different random seeds &#40;`--seed`&#41; will select different support sets, leading to performance variance. This is expected behavior.)

---

## 📊 Results & Logs

Training logs, checkpoints, and result CSVs are saved automatically:

```
logs/
└── Cls/<DATASET>/DyKo/DyKo/
    └── Shot<N_SHOT>/
        └── fold<FOLD>/
            ├── test_metrics.csv   # Accuracy, AUC, F1, etc.
            ├── all_probs.csv      # Model predictions
            └── version_<N>/       # TensorBoard logs

```

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 🖊️ Citation)

[//]: # ()
[//]: # (If you find **DyKo** useful for your research, please consider citing our paper:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{dyko2024,)

[//]: # (  title={DyKo: Dynamic Knowledge-guided Vision-Language Model for WSI Classification},)

[//]: # (  author={Your Name and Collaborators},)

[//]: # (  journal={Conference/Journal Name},)

[//]: # (  year={2024})

[//]: # (})

[//]: # ()
[//]: # (```)

## 🙏 Acknowledgements

This project is built upon the following excellent open-source works:

* [TITAN](https://github.com/MahmoodLab/TITAN) & [CONCH](https://github.com/mahmoodlab/CONCH) (Encoders)
* [CLAM](https://www.google.com/search?q=https://github.com/mahmoodlab/CLAM) (Preprocessing)
* [ViLa-MIL](https://www.google.com/search?q=https://github.com/BioMedical-Imaging-Laboratory/ViLa-MIL) & [FOCUS](https://www.google.com/search?q=https://github.com/HKU-MedAI/FOCUS) (References)
