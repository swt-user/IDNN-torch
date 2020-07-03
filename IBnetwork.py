import torch
import os
import numpy as np
import torch.nn as nn
import pickle


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=1/np.sqrt(float(m.weight.shape[0])))
        #torch.nn.init.normal_(m.bias, mean=0.0, std=1/np.sqrt(float(m.weight.shape[0])))
        m.bias.data.fill_(0.0)

class IBnet(nn.Module):

    def __init__(self, cfg):
        super(IBnet,self).__init__()

        self.cfg = cfg

        if cfg['ACTIVATION'] == 'relu':
            act_fun = nn.ReLU()
        elif cfg['ACTIVATION'] == 'tanh':
            act_fun = nn.Tanh()
        elif cfg['ACTIVATION'] == 'softsign':
            act_fun = nn.Softsign()
        elif cfg['ACTIVATION'] == 'softplus':
            act_fun = nn.Softplus()
        
        self.layer1 = nn.Sequential(
            nn.Linear(12, cfg['layersizes'][0]), 
            act_fun)

        self.layer2 = nn.Sequential(
            nn.Linear(cfg['layersizes'][0], cfg['layersizes'][1]),
            act_fun )

        self.layer3 = nn.Sequential(
            nn.Linear(cfg['layersizes'][1], cfg['layersizes'][2]),
            act_fun )

        self.layer4 = nn.Sequential(
            nn.Linear(cfg['layersizes'][2], cfg['layersizes'][3]),
            act_fun )

        self.layer5 = nn.Sequential(
            nn.Linear(cfg['layersizes'][3], cfg['layersizes'][4]),
            act_fun )

        self.layer6 = nn.Sequential(
            nn.Linear(cfg['layersizes'][4], cfg['NUM_CLASSES']),
            act_fun
        )

        self.apply(init_weights)

    def forward(self, input):
        ''' input (batch_size, 12)  '''
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        return x


    def report(self, input, epoch):
        ''' input (full_batch_size, 12) '''

        max_element = torch.Tensor([0])

        if not os.path.exists(self.cfg['SAVE_DIR']):
            print("Making directory", self.cfg['SAVE_DIR'])
            os.makedirs(self.cfg['SAVE_DIR'])

        if epoch not in self.cfg['EPOCHS']: # if don't need report
            return max_element

        data = {
            'weights_norm' : [],   # L2 norm of weights
            'gradmean'     : [],   # Mean of gradients
            'gradstd'      : [],   # Std of gradients
            'activity_tst' : []    # Activity in each layer for full dataset
        }

        with torch.no_grad():
            # record the activity_output
            x = self.layer1(input)
            data['activity_tst'].append(x)

            x = self.layer2(x)
            data['activity_tst'].append(x)

            x = self.layer3(x)
            data['activity_tst'].append(x)

            x = self.layer4(x)
            data['activity_tst'].append(x)

            x = self.layer5(x)
            data['activity_tst'].append(x)

        for param in self.parameters():
            data['weights_norm'].append(torch.sum(param*param))
            data['gradmean'].append(torch.mean(param.grad))
            data['gradstd'].append(torch.std(param.grad))

        for i in range(len(data['activity_tst'])):
            if torch.max(data['activity_tst'][i]) > max_element:
                max_element = torch.max(data['activity_tst'][i])

        fname = self.cfg['SAVE_DIR'] + "/epoch%08d"%epoch
        print("Saving", fname)
        with open(fname, 'wb') as f:
            pickle.dump({'ACTIVATION':self.cfg['ACTIVATION'], 'epoch':epoch, 'data':data}, f, pickle.HIGHEST_PROTOCOL)

        return max_element

    def get_accuracy(self, data, label):

        with torch.no_grad():
            # (0.2*4096, 2)
            output = self.forward(data)
            pre = torch.zeros(output.shape[0], dtype=torch.float32)
            for i in range(output.shape[0]):
                if output[i][0] < output[i][1]:
                    pre[i] = 1.0
            accuracy = torch.sum(pre == label.float()).item()/output.shape[0]
        
        return accuracy

