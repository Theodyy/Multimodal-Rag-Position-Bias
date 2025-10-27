# Multimodal RAG Position Bias

Official code for our EMNLP 2025 paper:  
**"Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation"**  
📄 [Paper (arXiv)](https://arxiv.org/pdf/2506.11063.pdf)

---

## 🔍 Overview

This project investigates **position bias** in Multimodal Retrieval-Augmented Generation (RAG) systems.  
We find that model accuracy follows a **U-shaped curve** with respect to evidence order —  
high at the beginning and end, but lowest in the middle.  

We propose the **Position Sensitivity Index (PSIₚ)** to quantify this bias and visualize its attention-level causes.

---

## 📦 Setup

```bash
git clone https://github.com/Theodyy/Multimodal-RAG-Position-Bias.git
cd Multimodal-RAG-Position-Bias
pip install -r requirements.txt
```

------

## 🚀 Run

### 🧩 Text-only (MS MARCO)

Evaluate position bias in text retrieval tasks.

```bash
cd exp/text-only
python ms-mini.py --output_dir ../../results/ms_mini
```

### 🖼️ Image-only (ChartQA)

Evaluate position bias in chart reasoning.

```bash
cd exp/image-only
python chart-mini.py --dataset_name chart --output_dir ../../results/chart_mini
```

Or batch run 10 trials:

```bash
bash run_mini5.sh
```

------

## 📊 Visualization

Reproduce Figure 4 (attention heatmaps):

```bash
cd vis
jupyter notebook single_3_diff.ipynb
```

------

## 📚 Citation

```bibtex
@article{yao2025spotlight,
  title={Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation},
  author={Yao, Jiayu and Liu, Shenghua and Wang, Yiwei and Mei, Lingrui and Bi, Baolong and Ge, Yuyao and Li, Zhecheng and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2506.11063},
  year={2025}
}
```

------

📧 Contact: **[yaojiayu25@mails.ucas.ac.cn](mailto:yaojiayu25@mails.ucas.ac.cn)**
