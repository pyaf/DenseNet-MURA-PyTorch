
# coding: utf-8

# In[ ]:

import os
import time
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets.folder import pil_loader

from densenet import densenet169
from utils import plot_training

get_ipython().magic('matplotlib inline')


# ### Prepare Data pipeline

# In[ ]:

data_cat = ['train', 'valid'] # data categories
study_data = {}
for phase in data_cat:
    BASE_DIR = 'MURA-v1.0/%s/XR_WRIST/' %(phase)
    patients = list(os.walk(BASE_DIR))[0][1]
    study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
    study_label = {'positive': 0, 'negative': 1}
    i = 0
    for patient in tqdm(patients[:100]):
        for study_type in os.listdir(BASE_DIR + patient):
            label = study_label[study_type.split('_')[1]]
            path = BASE_DIR + patient + '/' + study_type + '/'
            study_data[phase].loc[i] = [path, len(os.listdir(path)), label]
            i+=1


# In[ ]:

class StudyImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i+1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}
        return sample


# In[ ]:

data_transforms = {
    'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(), # coverts to tensor, scales to [0, 1]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: StudyImageDataset(study_data[x], transform=data_transforms[x]) for x in data_cat}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4) for x in data_cat}
dataset_sizes = {x: len(study_data[x]) for x in data_cat}
use_gpu = torch.cuda.is_available()
use_gpu = False


# ### Building the model

# In[ ]:

def n_p(x):
    '''convert numpy float to Variable tensor float'''
    if use_gpu:
        return torch.cuda.FloatTensor([x])
    else:
        return torch.FloatTensor([x])


# In[ ]:

def get_count(df, cat):
    '''
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for normal and "negative" for abnormal
    '''
    return df[df['Path'].str.contains(cat)]['Count'].sum()


# In[ ]:

# tas = total abnormal/negative studies, tns = total normal/positive studies

tas = {x: study_data[x]['Label'].value_counts()[1] for x in data_cat}
tns = {x: study_data[x]['Label'].value_counts()[0] for x in data_cat}

# tai = total abnormal images, tni = total normal images

tai = {x: get_count(study_data[x], 'negative') for x in data_cat}
tni = {x: get_count(study_data[x], 'positive') for x in data_cat}


Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}


print('tas:', tas)
print('tns:', tns, '\n')
print('tai:', tai)
print('tni:', tni, '\n')
print('Wt0 train:', Wt0['train'])
print('Wt0 valid:', Wt0['valid'])
print('Wt1 train:', Wt1['train'])
print('Wt1 valid:', Wt1['valid'])


# In[ ]:

def update_TP_TN(outputs, labels_data, TP, TN):
    '''
    Takes output and label_data and calculates True Positive (TP) and True Negative(TN)
    '''
    sum_array = (outputs + labels_data).cpu().numpy()
    TP += np.count_nonzero(sum_array == 2) # predicted = label = 1 and 1+1 = 2
    TN += np.count_nonzero(sum_array == 0) # predicted = label = 0 and 0+0 = 0
    return TP, TN


# In[ ]:

class Loss(nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(Loss, self).__init__()
        self.Wt1 = Variable(Wt1, requires_grad=False)
        self.Wt0 = Variable(Wt0, requires_grad=False)
        
    def forward(self, inputs, targets, phase):
        loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
        return loss


# In[ ]:

def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            TP, TN = 0, 0
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label']
                # wrap them in Variable
                labels = data['label'].type(torch.FloatTensor)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                if use_gpu:
                    preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                else:
                    preds = (outputs.data > 0.5).type(torch.FloatTensor)
                    
                running_corrects += torch.sum(preds == labels.data)
                TP, TN = update_TP_TN(preds, labels.data, TP, TN)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            sensitivity = TP / tai[phase]
            specificity = TN / tni[phase]
            
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('{} Sensitivity : {:.4f} Specificity: {:.4f}'.format(
                phase, sensitivity, specificity))

            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[ ]:

model = densenet169(pretrained=True)
if use_gpu:
    model = model.cuda()


# In[ ]:

criterion = nn.modules.loss.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)


# In[ ]:

model = train_model(model, criterion, optimizer, num_epochs=4)


# In[ ]:

# torch.save(model.state_dict(), 'models/v2.0.pth')


# In[ ]:

# model.load_state_dict(torch.load('models/v1.pth'))


# In[ ]:

# Model architecture used in this code
# model


# In[ ]:

get_ipython().system('jupyter nbconvert --to script v2.ipynb')


# In[ ]:




# In[ ]:



