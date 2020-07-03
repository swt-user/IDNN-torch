import numpy as np
import scipy.io as sio
from pathlib2 import Path
from collections import namedtuple
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class IBDataset(Dataset):

    def __init__(self, ID):
        super(IBDataset).__init__()
        self.trn, self.tst = get_IB_data(ID)
        self.mode = 'train'

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.trn.X[index], self.trn.y[index]
        else:
            return self.tst.X[index], self.tst.y[index]

    def __len__(self):
        if self.mode == 'train':
            return self.trn.X.shape[0]
        else:
            return self.tst.X.shape[0]
    
    def change_mode(self, mode):
        if mode == 'train':
            self.mode = mode
        elif mode == 'test':
            self.mode = mode


def get_IB_data(ID):

    # Returns two namedtuples, with IB training and testing data
    #   trn.X is training data
    #   trn.y is trainiing class, with numbers from 0 to 1
    #   trn.Y is training class, but coded as a 2-dim vector with one entry set to 1
    # similarly for tst

    nb_classes = 2
    data_file = Path('datasets/IB_data_'+str(ID)+'.npz')
    if data_file.is_file():
        data = np.load('datasets/IB_data_'+str(ID)+'.npz')
    else:
        create_IB_data(ID)
        data = np.load('datasets/IB_data_'+str(ID)+'.npz')
        

    X_train, y_train= torch.FloatTensor(data['X_train']), torch.LongTensor(data['y_train'].squeeze())
    X_test, y_test = torch.FloatTensor(data['X_test']), torch.LongTensor(data['y_test'].squeeze())
    # y_train (0.8*4096, 1)  y_test (0.2*4096，1)
    Y_train = one_hot_embedding(y_train, nb_classes)
    Y_test  = one_hot_embedding(y_test, nb_classes)


    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)
    del X_train, X_test, Y_train, Y_test, y_train, y_test
    return trn, tst



def create_IB_data(idx):

    # data_sets_org.data (4096,12) 
    # data_sets_org.labels (4096,2)
    data_sets_org = load_data()

    data_sets = data_shuffle(data_sets_org, [80], shuffle_data=True)

    X_train, y_train, X_test, y_test = data_sets.train.data, data_sets.train.labels[:,0], data_sets.test.data, data_sets.test.labels[:,0]

    np.savez_compressed('datasets/IB_data_'+str(idx), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)



def construct_full_dataset(trn, tst):

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])

    X = np.concatenate((trn.X,tst.X))

    y = np.concatenate((trn.y,tst.y))

    Y = np.concatenate((trn.Y,tst.Y))

    return Dataset(torch.FloatTensor(X), torch.FloatTensor(Y), torch.LongTensor(y), trn.nb_classes)

 

def load_data():

    """Load the data
    name - the name of the dataset
    return object with data and labels"""

    print ('Loading Data...')
    C = type('type_C', (object,), {})
    data_sets = C()
    d = sio.loadmat('datasets/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets = C()

    # datasets.data (4096,12)
    data_sets.data = F
    # datasets.labels (4096,2)
    data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)

    return data_sets



def shuffle_in_unison_inplace(a, b):

    """Shuffle the arrays randomly"""

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]



def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):

    """Divided the data to train and test and shuffle it"""

    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)

    C = type('type_C', (object,), {})
    data_sets = C()

    stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    if percent_of_train[0] > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])

    data_sets.train = C()
    data_sets.test = C()

    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels

    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]

    return data_sets

def one_hot_embedding(labels, num_classes):

    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels]

def get_MNIST(test_loader=None):

    nb_classes = 10
    
    if test_loader == None:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train = False, transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1037,), (0.3081,))
            ])),
            batch_size = cfg['SGD_BATCHSIZE'], shuffle = False)
    
    for batch, (data, target) in enumerate(test_loader):
        if batch == 0:
            X = data
            y = target
        else:
            X = torch.cat([X, data], dim=0)
            y = torch.cat([y, target])
    

    y = y.long()

    Y = one_hot_embedding(y, nb_classes)

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])

    full = Dataset(X, Y, y, nb_classes)
 
    del X, Y, y

    return full