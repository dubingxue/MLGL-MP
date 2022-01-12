from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric import data as DATA
import torch
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

compound_iso_smiles = []
opts = ['train','test']
for opt in opts:
    df = pd.read_csv('data/' + opt + '.csv')
    compound_iso_smiles += list( df['SMILES'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

# convert to PyTorch data format
processed_train = 'data/processed/' + 'train.pt'
processed_test = 'data/processed/' + 'test.pt'
if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_test))):

    df = pd.read_csv('data/' + 'train.csv')
    train_compounds = list(df['SMILES'])
    train_compounds = np.asarray(train_compounds)

    train_Y = np.array(pd.DataFrame(df['Label'])).tolist()
    list_1 = []
    list_2 = []
    for i in train_Y:
        list_2 = [int(i) for i in i[0].split(',')]
        list_1.append(list_2)
    Train_Y = MultiLabelBinarizer().fit_transform(list_1)
    train_Y = np.asarray(Train_Y.tolist())

    df = pd.read_csv('data/' + 'test.csv')
    test_compounds, test_Y = list(df['SMILES']), list(df['Label'])
    test_compounds= np.asarray(test_compounds)

    test_Y = np.array(pd.DataFrame(df['Label'])).tolist()
    list_3 = []
    list_4 = []
    for i in test_Y:
        list_4 = [int(i) for i in i[0].split(',')]
        list_3.append(list_4)
    Test_Y = MultiLabelBinarizer().fit_transform(list_3)
    test_Y = np.asarray(Test_Y.tolist())

    # make data PyTorch Geometric ready
    print('preparing,' + 'train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='train', xd=train_compounds, y=train_Y,
                                smile_graph=smile_graph)
    print('preparing,' + 'test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset= 'test', xd=test_compounds, y=test_Y,
                               smile_graph=smile_graph)
    print(processed_train, ' and ', processed_test, ' have been created')
else:
    print(processed_train, ' and ', processed_test, ' are already created')
