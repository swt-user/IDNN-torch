import torch
import os
import numpy as np
import torch.nn as nn
import pickle


class Lenet(nn.Module):

    def __init__(self, cfg):
        super(Lenet, self).__init__()

        self.cfg = cfg

        if cfg['ACTIVATION'] == 'relu':
            act_fun = nn.ReLU()
        elif cfg['ACTIVATION'] == 'tanh':
            act_fun = nn.Tanh()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            act_fun
            )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(10, 20, 3), 
            act_fun
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(20 * 10 * 10, 500),
            act_fun
        )

        self.layer4 = nn.Sequential(
            nn.Linear(500, 10),
            act_fun
        )

    def forward(self, input):

        out = self.layer1(input)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

    def report(self, epoch, test_loader):

        max_elements = 0

        if not os.path.exists(self.cfg['SAVE_DIR']):
            print("Making directory", self.cfg['SAVE_DIR'])
            os.makedirs(self.cfg['SAVE_DIR'])

        if epoch not in self.cfg['EPOCHS']: # if don't need report
            return max_elements

        data = {       
            'weights_norm' : [],   # L2 norm of weights
            'gradmean'     : [],   # Mean of gradients
            'gradstd'      : [],   # Std of gradients
            'activity_tst' : [],   # Activity in each layer for full dataset
        }

        with torch.no_grad():
            # record the activity_output
            for batch_id, (inpu, target) in enumerate(test_loader):
                if batch_id == 0:
                    
                    x = self.layer1(inpu)
                    data['activity_tst'].append(x.view(x.shape[0], -1))

                    x = self.layer2(x)
                    data['activity_tst'].append(x.view(x.shape[0], -1))

                    x = x.view(x.shape[0], -1)

                    x = self.layer3(x)
                    data['activity_tst'].append(x)
                
                else:

                    x = self.layer1(inpu)
                    data['activity_tst'][0] = torch.cat([data['activity_tst'][0], x.view(x.shape[0], -1)], dim = 0)

                    x = self.layer2(x)
                    data['activity_tst'][1] = torch.cat([data['activity_tst'][1], x.view(x.shape[0], -1)], dim = 0)

                    x = x.view(x.shape[0], -1)

                    x = self.layer3(x)
                    data['activity_tst'][2] = torch.cat([data['activity_tst'][2], x], dim = 0)


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

    def test(self, test_loader):
    
        self.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                
                output = self(data)
                pred = output.max(1, keepdim = True)[1] # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        print("\nTest set: , Accuracy: {}/{} ({:.0f}%) \n".format(
             correct, len(test_loader.dataset),
            100.* correct / len(test_loader.dataset)
                ))