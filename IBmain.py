import torch
from IBnetwork import IBnet
import numpy as np
from utils import IBDataset, construct_full_dataset
from torch.utils.data import DataLoader
from ComputeMI import MI,plot_infoplane, plot_SNR

def get_cfg():

    cfg = {}
    cfg['SGD_BATCHSIZE']    = 256
    cfg['SGD_LEARNINGRATE'] = 0.0004
    cfg['NUM_EPOCHS']       = 8000
    cfg['FULL_MI']          = True
    cfg['NAME']             = '2017_12_21_16_51_3_275766'
    cfg['EPOCHS']           = []
    cfg['NUM_CLASSES']      = 2
    cfg['NUM_BINS']         = 100
    cfg['MAX_ELEMENT']      = torch.Tensor([33.5246])

    cfg['ACTIVATION'] = 'tanh'
    # cfg['ACTIVATION'] = 'relu'
    # cfg['ACTIVATION'] = 'softsign'
    # cfg['ACTIVATION'] = 'softplus'

    # How many hidden neurons to put into each of the layers
    cfg['layersizes'] = [10,7,5,4,3] # original IB network
    ARCH_NAME =  '-'.join(map(str,cfg['layersizes']))

    # Where to save activation and weights data
    cfg['SAVE_DIR'] = 'rawdata/' + cfg['ACTIVATION'] + '_' + ARCH_NAME
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
    cfg['EPOCHS'] = np.unique(np.rint(np.logspace(0, np.log10(cfg['NUM_EPOCHS']), num=200, endpoint=True, base=10.0, dtype=None)))
    return cfg

def train(cfg):


    model = IBnet(cfg)

    dataset = IBDataset(cfg['NAME'])

    dataloader = DataLoader(dataset, batch_size=cfg['SGD_BATCHSIZE'], shuffle=False, num_workers=0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['SGD_LEARNINGRATE'])
    Loss = torch.nn.CrossEntropyLoss()

    full = construct_full_dataset(dataset.trn, dataset.tst)

    for epoch in range(cfg['NUM_EPOCHS']):

        for batch_id, (data , label) in enumerate(dataloader):
            output = model(data)
            loss = Loss(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id == 0:  # every epoch report once
                max_element = model.report(full.X, epoch)
                if max_element > cfg['MAX_ELEMENT']:
                    cfg['MAX_ELEMENT'] = max_element

        if epoch in cfg['EPOCHS']:
            test_accuracy = model.get_accuracy(dataset.tst.X, dataset.tst.y)
            print('epoch:{}, loss:{}, test_accuracy:{}'.format(epoch, loss, test_accuracy))


if __name__ == "__main__":
    cfg = get_cfg()
    
    #train(cfg)
    print('--------max_elements:{}---------------'.format(cfg['MAX_ELEMENT']))
    dataset = IBDataset(cfg['NAME'])

    full = construct_full_dataset(dataset.trn, dataset.tst)
    measures = MI(cfg, infoplane_measure='HSIC', full_dataset=full)
    plot_infoplane(cfg, infoplane_measure='HSIC', measures= measures, PLOT_LAYERS=[0,1,2,3,4])
    
    #plot_SNR(cfg, PLOT_LAYERS=[0,1,2,3,4])