# Multi-Label Graph Learning for Metabolic Pathway (MLGL-MP) Prediction framework
Multi-Label Graph Learning
Framework enhanced by Pathway Inter-dependence for Metabolic Pathway Prediction is a novel framework to make fully use of available metabolic pathway data, which can learn the task for  Metabolic Pathway prediction. Here is the overview of the MLGL-MP framework.


# Step-by-step running:

## 1. Requirements
+ Python == 3.8
+ PyTorch == 1.8.0
+ rdikt == 2019.09.3
+ scikit-learn == 0.24.2
+ pandas == 1.2.0
+ numpy == 1.20.2
+ Jupyter Notebook
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric

## 2. Dataset
A dataset of 6669 compounds named "kegg_classes.txt" in dataset belonging to one or more of these 11 constituent pathway types was adopted from [1].


## 3. Create data 
3.1 Pathway embeddings

Running Pathway_Embedding.ipynb in Pre-training and obtain Pathway_Embedding.pkl to be as node features in Pathway Dependence Graph

3.2 Pathway Dependence Matrix 

Running adj_file.ipynb in Dataset / Data_10-fold cross validation / fold x and obtain adj.pkl to be as edges in Pathway Dependence Graph (codes are in fold 1)

3.3 Obtain pytorch format data

Running
```sh
python create_data.py
```
train.csv and test.csv in data are input to create data in pytorch format,
stored at data/processed/, consisting of  train.pt and test.pt.
# Usage
```sh
python Training.py
```
# Acknowledgements
Part of the code was adopted from [2] and [3].
# References
[1] Baranwal, M., et al. A deep learning architecture for metabolic pathway prediction. Bioinformatics 2020;36(8):2547-2553.

[2] Nguyen, T., et al. GraphDTA: predicting drug–target binding affinity with graph neural networks. Bioinformatics 2021;37(8):1140-1147.

[3] Chen, Z.M., et al. Multi-Label Image Recognition with Graph Convolutional Networks. In, 2019 Ieee/Cvf Conference on Computer Vision and Pattern Recognition. Los Alamitos: Ieee Computer Soc; 2019. p. 5172-5181.
