import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score,hamming_loss,label_ranking_loss

def accuracy_(label,output):

    zs = torch.sigmoid(output).to('cpu').data.numpy()
    ts = label.to('cpu').data.numpy()
    preds = list(map(lambda x: (x >= 0.5).astype(int), zs))

    preds_list, t_list = [], []
    preds_list = np.append(preds_list, preds)
    t_list = np.append(t_list, ts)

    acc = accuracy_score(t_list, preds_list)
    precision = precision_score(t_list, preds_list)
    recall = recall_score(t_list, preds_list)
    f1_scroe = (2 * precision * recall) / (recall + precision)
    ham_l = hamming_loss(t_list, preds_list)
    return acc,precision,recall,f1_scroe,ham_l

def Coverage(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
    D = len(label[0])
    N = len(label)
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        label_index.append(index)
    cover = 0
    for i in range(N):
        # Sorted from largest to smallest
        index = np.argsort(-output[i]).tolist()
        tmp = 0
        for item in label_index[i]:
            tmp = max(tmp, index.index(item))
        cover += tmp
    coverage = cover * 1.0 / N
    return coverage

def One_error(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
    N = len(label)
    for i in range(N):
        if max(label[i]) == 0:
            print("This data is not in either category")
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]
        label_index.append(index)
    OneError = 0
    for i in range(N):
        if np.argmax(output[i]) not in label_index[i]:
            OneError += 1
    OneError = OneError * 1.0 / N
    return OneError

def Ranking_loss(label, output):
    label = label.to('cpu').data.numpy()
    output = output.to('cpu').data.numpy()
    RL = label_ranking_loss(label, output)
    return RL



