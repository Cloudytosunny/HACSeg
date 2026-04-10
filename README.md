# HACSeg: Hierarchical Adaptive Clustering for Learning-free Instance Extraction in LiDAR Panoptic Segmentation

[![Paper](https://img.shields.io/badge/Paper)](你的论文链接) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is the official implementation of **HACSeg**, a novel hierarchical adaptive clustering framework designed for learning-free LiDAR panoptic segmentation.

---

## 🚀 Visualizations

<p align="center">
  <img src="https://github.com/Cloudytosunny/HACSeg/blob/main/pic/HACSeg_Mink_1.gif" width="48%" />
  <img src="https://github.com/Cloudytosunny/HACSeg/blob/main/pic/HACSeg_Mink_2.gif" width="48%" />
</p>

---

## 🏆 Leaderboard Results

To demonstrate the effectiveness of **HACSeg**, we present our performance on the SemanticKITTI Panoptic Segmentation leaderboards. (The data uploaded to the new website is different from the data uploaded to the old website, but the test metrics and test methods are the same.)

### 🆕 Latest Results (New Website)
*Results on the current benchmarking server.*

<p align="center">
  <a href="https://www.codabench.org/competitions/13666/#/results-tab">
    <img src="https://raw.githubusercontent.com/Cloudytosunny/HACSeg/main/pic/new_results.png" width="90%" alt="Latest Results" />
  </a>
</p>

---

### 📜 Historical Results (Old Website)
*Previous results (Note: Uploaded data differs from the new website).*

<p align="center">
  <a href="https://codalab.lisn.upsaclay.fr/competitions/7092#results">
    <img src="https://raw.githubusercontent.com/Cloudytosunny/HACSeg/main/pic/old_results.png" width="90%" alt="Old Results" />
  </a>
</p>


---

## 📝 Introduction
HACSeg focuses on the intrinsic geometric properties of LiDAR point clouds to achieve robust panoptic segmentation without the need for extensive training. 



## 🛠️ News
- **[2026/04]**: 🚀 Released the unified Decoupled-PQ evaluation suite! The scripts now support both IPQ and S_PQ metrics across SemanticKITTI and nuScenes datasets.
- **[2026/02]**: Code cleanup is in progress.
- **[Coming Soon]**: We will release the full HACSeg source code, including the core clustering algorithm and a comprehensive environment setup guide.

## 📅 Roadmap
- [x] Release unified evaluation scripts for Decoupled PQ (IPQ & S_PQ) on SemanticKITTI / nuScenes.
- [ ] Release core hierarchical adaptive clustering algorithm (C++/Python).
- [ ] Provide comprehensive environment setup and running instructions.
- [ ] Release full panoptic segmentation pipeline and inference examples.

---

## 📊 Evaluation & Metrics

HACSeg includes a unified evaluation suite to facilitate deep-dive analysis of panoptic segmentation performance. These scripts support both **SemanticKITTI** and **nuScenes** datasets and implement decoupled metrics to isolate instance extraction quality from semantic labeling quality.

### ⚙️ Environment Setup

The evaluation suite requires the following dependencies:

```bash
# Install basic requirements
pip install numpy pyyaml tqdm

# Required for nuScenes evaluation
pip install nuscenes-devkit
```

### 📈 Decoupled PQ Evaluation

We provide two core scripts to analyze the performance of HACSeg by decoupling different aspects of the panoptic task:

#### 1. Instance-Decoupled PQ (IPQ)
The `unified_evaluate_ins_decoupled_pq.py` script computes the **IPQ**, which measures the theoretical upper bound of instance extraction by assuming perfect semantic alignment. This metric is crucial for validating the geometric robustness of our **Hierarchical Adaptive Clustering** independently from the semantic backbone.

**SemanticKITTI Usage:**
```bash
python unified_evaluate_ins_decoupled_pq.py \
    --dataset semantickitti \
    --pred-dir /path/to/predictions \
    --gt-dir /path/to/labels \
    --output ./results_ipq_sem
```

**nuScenes Usage:**
```bash
python unified_evaluate_ins_decoupled_pq.py \
    --dataset nuscenes \
    --result-path /path/to/panoptic_results \
    --dataroot /data/sets/nuscenes \
    --version v1.0-trainval \
    --eval-set val \
    --output ./results_ipq_nus
```

#### 2. Semantic-Decoupled PQ (S_PQ)
The `unified_evaluate_semantic_decoupled_pq.py` script evaluates the **S_PQ** ($S\_PQ = PQ_{pre} / PQ_{csi}$). It provides insight into how much the final panoptic quality is constrained by semantic labeling errors versus instance clustering errors.

**SemanticKITTI Usage:**
```bash
python unified_evaluate_semantic_decoupled_pq.py \
    --dataset semantickitti \
    --pred-dir /path/to/predictions \
    --gt-dir /path/to/labels \
    --output ./results_spq_sem
```

**nuScenes Usage:**
```bash
python unified_evaluate_semantic_decoupled_pq.py \
    --dataset nuscenes \
    --result-path /path/to/panoptic_results \
    --dataroot /data/sets/nuscenes \
    --version v1.0-trainval \
    --eval-set val \
    --output ./results_spq_nus
```

### 📑 Metric Descriptions

Our evaluation framework provides a comprehensive set of metrics:

* **PQ (Panoptic Quality):** Standard metric for panoptic segmentation.
* **IPQ (Instance-centric Panoptic Quality):** Measures instance extraction performance.
* **S_PQ (Saturation PQ):** Ratio of actual PQ to the CSI-GT (theoretical bound).
* **PQ† (PQ Dagger):** A variation averaging PQ for "Things" and IoU for "Stuff".

> **Interactive Reports:** Both scripts automatically generate a `scores.txt` (YAML format) and a `detailed_results.html` file. The HTML report features an interactive, sortable table for efficient per-class metric analysis.




## 📧 Contact
**Wang Shaohu** (Southeast University) - [wangsh@seu.edu.cn]
