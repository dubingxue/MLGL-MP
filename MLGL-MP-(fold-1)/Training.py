import sys
from model import *
from utils import *
from evalution import *
import pickle

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data,inp)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data,inp)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    return total_labels,total_preds

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

inp_1 = pickle.load(open('Pathway_Embedding.pkl','rb'))
inp_2 = inp_1.astype(np.float32)
inp = torch.from_numpy(inp_2)

processed_train = 'data/processed/' + 'train.pt'
processed_test = 'data/processed/' + 'test.pt'
if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_test))):
        print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='train')
    test_data = TestbedDataset(root='data', dataset='test')

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    num_classes = 11
    model = MLGL_MP(num_classes=num_classes, t=0.5, adj_file='adj.pkl').to(device)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    max_acc = 0

    model_file_name = 'model_' + '.pt'
    result_file_name = 'result_' + '.csv'
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        G, P = predicting(model, device, test_loader)
        acc,precision,recall,f1_scroe,ham_l= accuracy_(G,P)
        coverage = Coverage(G,P)
        one_error = One_error(G,P)
        RL = Ranking_loss(G,P)
        ret = [acc, precision, recall, f1_scroe,ham_l, coverage, one_error, RL]
        if acc > max_acc:
            max_acc = acc
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
        print('%.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f' % (acc, precision, recall, f1_scroe,ham_l,coverage,one_error,RL))

