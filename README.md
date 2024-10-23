# CaNet
The official implementation for WWW2024 Oral paper "Graph Out-of-Distribution Generalization via Causal Intervention"

Related material: [[Paper](https://arxiv.org/pdf/2402.11494.pdf)], [[Blog (Chinese)](https://zhuanlan.zhihu.com/p/709125359)], [[Blog (English)](https://medium.com/towards-data-science/towards-generalization-on-graphs-from-invariance-to-causality-c81a174ac37b)], [[Slides](https://qitianwu.github.io/assets/publications/www2024-canet/slides.pdf)]

## What's news

[2024.02.08] We release the code for the model on six datasets. More detailed info will be updated soon.

## Model and Results

Our model coordinates two key components 1) an environment estimator that infers pseudo environment labels, and 2) a mixture-of-expert GNN predictor with feature propagation
units conditioned on the pseudo environments. 

<img src="https://github.com/fannie1208/CaNet/assets/89764090/04603d2b-4d1d-4a1b-a2c0-6110c325e84d" alt="image" width="800">

## Dataset

One can download the datasets Planetoid (Cora, Citeseer, Pubmed), Arxiv, Twitch, and Elliptic from the google drive link below:

https://drive.google.com/drive/folders/1FAPWghoyGp9vzr1xmnndpmLLFS1OgBDa?usp=sharing

## Dependence

Python 3.8, PyTorch 1.13.0, PyTorch Geometric 2.1.0, NumPy 1.23.4

## Run the codes

Please refer to the bash script `run.sh` in each folder for running the training and evaluation pipeline on six datasets.

## Acknowledgement

The implementation of training pipeline and evaluation is based on [EERM](https://github.com/qitianwu/GraphOOD-EERM).

### Citation

If you find our code and model useful, please cite our work. Thank you!

```bibtex
      @inproceedings{wu2024canet,
      title = {Graph Out-of-Distribution Generalization via Causal Intervention},
      author = {Qitian Wu and Nie Fan and Chenxiao Yang and Tianyi Bao and Junchi Yan},
      booktitle = {The Web Conference},
      year = {2024}
      }
```
