# Learning Accurate, Efficient, and Interpretable MLPs on Multiplex Graphs via Node-wise Multi-View Ensemble Distillation

Official Implementation of Learning Accurate, Efficient, and Interpretable MLPs on Multiplex Graphs via Node-wise Multi-View Ensemble Distillation.

- Dataset Loader (ACM, IMDB, IMDB5K, DBLP, ArXiv, MAG)
- Two evaluation settings: transductive and inductive
- Various teacher MGNN architectures (RSAGE, RGCN, RGAT, HAN) and student MLPs
- Training paradigm for teacher MGNNs and student MLPs



## Getting Started

### Setup Environment

To run the code, please install the following libraries: dgl==2.4.0+cu124, torch==2.4.0, numpy==2.1.3, scipy==1.14.1



### Preparing Datasets

All datasets are available under `data/`. 

- `ACM`, `IMDB`, `IMDB5K`, and `DBLP` have already been well-organized.
- Before using `ArXiv` and `MAG`, please run `reorganize_arxiv_mag_datasets.py` to reorganize them.
- Your favourite datasets: download and add to the `load_data` function in `dataloader.py`.



### Training and Evaluation

To quickly train a teacher model you can run `train_teacher.py` by specifying the experiment setting, i.e. transductive (`tran`) or inductive (`ind`), teacher model, e.g. `RSAGE`, and dataset, e.g. `ACM`, as per the example below.

```
python train_teacher.py --exp_setting tran --teacher RSAGE --dataset ACM
```

To quickly train a student model with a pretrained teacher you can run `train_student.py` or `train_stud_ensemble.py` by specifying the experiment setting, teacher model, student model, and dataset like the example below. Make sure you train the teacher using the `train_teacher.py` first and have its result stored in the correct path.

```
python train_student.py --exp_setting tran --teacher RSAGE --student MLP --dataset ACM

python train_stud_ensemble.py --exp_setting tran --teacher RSAGE --student MyMLP --dataset ACM
```



## Acknowledgements

The code is implemented based on [GLNN](https://github.com/snap-research/graphless-neural-networks).
