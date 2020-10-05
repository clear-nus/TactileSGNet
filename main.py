# -*- coding: utf-8 -*`-
"""
Created on Sat August 1, 2020

@author: Fuqiang

This is the code for TactileSGNet. You may cite it as follows:
Gu, F., Sng, W., Taunyazov, T., & Soh, H. (2020). TactileSGNet: A Spiking Graph Neural Network for Event-based Tactile Object Recognition,IROS 2020.

"""

from __future__ import print_function
import os
import time
from datetime import date
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tactileSGNet import*
from torch.autograd import Variable
import random
import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

# model name
model_name = '_TactileSGNet_' # tactile 


# hyperparameter setting
num_classes = 36
k = 3 # number of nodes to be connected for constructing graph
num_run = 1
learning_rate = 1e-3 #1e-3
num_epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Event-based tactile dataset
class tactileDataset(Dataset):
    def __init__(self, data_path, train=True):
        if train:
            self.files = os.listdir(data_path + '/train')
            self.file_path = data_path + '/train/'
        else:
            self.files = os.listdir(data_path + '/test')
            self.file_path = data_path + '/test/'

    def __getitem__(self, index):
        fileName = self.files[index]
        nameStr = fileName.split('_label_')
        label = int(nameStr[-1].split('.')[0])
        data = torch.from_numpy(np.load(self.file_path + fileName)) #torch.FloatTensor(np.load(self.file_path + fileName))
        label = torch.LongTensor([label])                      
        return data, label                                                                                                                 
    def __len__(self):
        return len(self.files)

# Decay learning rate
def lr_scheduler(optimizer, epoch, init_lr = 0.01, lr_decay_epoch=30):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


# Tactile dataset
data_path = 'Ev-Objects/'
trainDataset = tactileDataset(data_path, train=True)
testDataset = tactileDataset(data_path, train=False)

# run for num_run times
best_acc = torch.zeros(num_run)
acc_list = list([])
training_loss_list = list([])
test_loss_list = list([])
net_list = list([])

for run in range(num_run):
    model = TactileSGNet(num_classes, k, device=device)
    model.to(device)
    criterion = nn.MSELoss() #nn.MSELoss(reduction='sum') #nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    acc = torch.zeros(num_epochs)
    training_loss = torch.zeros(num_epochs)
    test_loss = torch.zeros(num_epochs)
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0
        for trainData, trainLabel in tqdm.tqdm(trainDataset): 
            model.zero_grad()
            optimizer.zero_grad()
            trainData = trainData.to(device) 
            outputs = model(trainData)
            labels_ = torch.zeros(1, num_classes).scatter_(1, trainLabel.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_) 
              
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        training_loss[epoch] = running_loss

         # testing
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
        running_loss = 0
        with torch.no_grad():
            for testData, testLabel in testDataset:
                optimizer.zero_grad()
                outputs = model(testData, False)
                labels_ = torch.zeros(1, num_classes).scatter_(1, testLabel.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_) 
                running_loss += loss.item()
                _, predicted = outputs.cpu().max(0)
                total += float(testLabel.size(0))
                correct += float(predicted.eq(testLabel).sum().item())
                
            test_loss[epoch] = running_loss         

            acc[epoch] = 100. * float(correct) / float(total)
            if best_acc[run] < acc[epoch]:
                best_acc[run] = acc[epoch]

        test_loss_list.append(test_loss) 
        training_loss_list.append(training_loss)

        acc_list.append(acc)
        if (epoch + 1) % 2 == 0:
            print('At epoch: %s, training_loss: %f, test_loss: %f, best accuracy: %.3f, time elasped: %.3f' % (epoch + 1, training_loss[epoch], test_loss[epoch], best_acc[run], time.time()-start_time ))
            start_time = time.time()
             
    net_list.append(model.state_dict())
    
# overall state
state = {
    'net_list': net_list,
    'best_acc': best_acc,
    'num_epochs': num_epochs,
    'acc_list': acc_list,
    'training_loss_list': training_loss_list,
    'test_loss_list': test_loss_list,
}
dateStr = date.today().strftime("%Y%m%d")

if not os.path.isdir('log_data'):
   os.mkdir('log_data')
torch.save(state, './log_data/' + dateStr + model_name + '_objects_' +  str(num_classes) + '_k_' + str(k)  + '.t7')
print('Avg acc: %f, std: %f: ' % (torch.mean(state['best_acc']), torch.std(state['best_acc'])))

