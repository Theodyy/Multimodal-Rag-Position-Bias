# Multimodal RAG Position Bias

This repository contains the code and resources for the paper:

**"Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation"**  
*(EMNLP 2025, Findings)*  
📄 Paper link: [arXiv / Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation](https://arxiv.org/pdf/2506.11063.pdf)

---

## 🔍 Overview
Multimodal Retrieval-Augmented Generation (RAG) systems are increasingly applied to knowledge-intensive tasks.  
However, we show that these systems are systematically sensitive to the **position of retrieved evidence**, leading to a consistent **U-shaped accuracy curve** and unstable reasoning.

Key contributions:
- Introduce the **Position Sensitivity Index ($PSI_p$)** to quantify positional bias.  
- Conduct large-scale controlled experiments across text, image, and multimodal RAG tasks.  
- Reveal that cross-modal interactions amplify positional bias compared to unimodal settings.  

---

## 📊 Contents
- `src/` – Core implementation of PSI$_p$ computation and evaluation pipeline.  
- `experiments/` – Example training and evaluation scripts.  

---

## 📈 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{yao2025spotlight,
  title={Who is in the Spotlight: The Hidden Bias Undermining Multimodal Retrieval-Augmented Generation},
  author={Yao, Jiayu and Liu, Shenghua and Wang, Yiwei and Mei, Lingrui and Bi, Baolong and Ge, Yuyao and Li, Zhecheng and Cheng, Xueqi},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2025},
  year={2025}
}
```

---

## 📬 Contact

For questions or feedback, please contact:
📧 `yaojiayu25@mails.ucas.ac.cn`


