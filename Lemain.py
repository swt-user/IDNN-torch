import torch
import torch.nn as nn
import numpy as np
from Lenetwork import Lenet
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils import get_MNIST
from ComputeMI import *

def get_cfg():

    cfg = {}
    cfg['SGD_BATCHSIZE']    = 256
    cfg['SGD_LEARNINGRATE'] = 0.0004
    cfg['NUM_EPOCHS']       = 70
    cfg['FULL_MI']          = True
    cfg['NAME']             = 'MNIST'
    cfg['EPOCHS']           = []
    cfg['NUM_CLASSES']      = 10
    cfg['NUM_BINS']         = 30
    cfg['MAX_ELEMENT']      = torch.Tensor([0])

    #cfg['ACTIVATION'] = 'tanh'
    cfg['ACTIVATION'] = 'relu'
    #cfg['ACTIVATION'] = 'softsign'
    # cfg['ACTIVATION'] = 'softplus'

    # How many hidden neurons to put into each of the layers
    cfg['layersizes'] = [1440, 2000, 500 ] # original IB network
    ARCH_NAME =  '-'.join(map(str,cfg['layersizes']))

    # Where to save activation and weights data
    cfg['SAVE_DIR'] = './rawdata/' + cfg['ACTIVATION'] + '_' + ARCH_NAME
    '''
    for epoch in range(cfg['NUM_EPOCHS']):
        if epoch < 20 and epoch%1==0:       # Log for all first 20 epochs
            cfg['EPOCHS'].append(epoch)
        elif epoch < 100 and epoch%5==0:    # Then for every 5th epoch
            cfg['EPOCHS'].append(epoch)
        elif epoch < 2000 and epoch%10==0:    # Then every 10th
            cfg['EPOCHS'].append(epoch)
        elif epoch%100==0:                # Then every 100th
            cfg['EPOCHS'].append(epoch)
    '''
    cfg['EPOCHS'] = np.unique(np.rint(np.logspace(0, np.log10(cfg['NUM_EPOCHS']), num=100, endpoint=True, base=10.0, dtype=None)))

    return cfg

def train(cfg, train_loader, test_loader):


    model = Lenet(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['SGD_LEARNINGRATE'])
    Loss = nn.CrossEntropyLoss()

    #full_data = get_MNIST(train_loader= train_loader)

    epoch = 0
    while(epoch < cfg['NUM_EPOCHS']):
        for batch_idx, (data, target) in enumerate(train_loader):
            epoch += 1
            optimizer.zero_grad()
            output = model(data)
            loss = Loss(output, target)
            loss.backward()
    
            max_element = model.report(epoch, test_loader)
            if max_element > cfg['MAX_ELEMENT']:
                cfg['MAX_ELEMENT'] = max_element

            optimizer.step()

            if (batch_idx + 1) % 30 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                model.test(test_loader)
        

if __name__ == "__main__":

    cfg = get_cfg()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train = True, download = True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1037,), (0.3081,))
                ])),
    batch_size = cfg['SGD_BATCHSIZE'], shuffle = False)

    # 测试集
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size = cfg['SGD_BATCHSIZE'], shuffle = False)

    train(cfg, train_loader, test_loader)

    print('--------max_elements:{}---------------'.format(cfg['MAX_ELEMENT']))
    
    full_data = get_MNIST(test_loader = test_loader)

    measures = MI(cfg, infoplane_measure='bin', full_dataset= full_data)

    plot_infoplane(cfg, infoplane_measure='bin', measures= measures, PLOT_LAYERS=[0,1,2])