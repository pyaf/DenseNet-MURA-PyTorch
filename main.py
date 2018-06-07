import time
import copy
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from tqdm import tqdm
from densenet import densenet169
from utils import n_p
from train import train_model, get_metrics
from pipeline import get_study_level_data, get_dataloaders


# #### load study level dict data

study_data = get_study_level_data(study_type='XR_WRIST')
data_cat = ['train', 'valid'] # data categories


# #### Create dataloaders pipeline

dataloaders = get_dataloaders(study_data, batch_size=1, study_level=True)
dataset_sizes = {x: len(study_data[x]) for x in data_cat}
print('dataset sizes, study count:', dataset_sizes)


# #### Build model

# count number of positive and negative studies in the dataloader, to be used to compute Wt1 and Wt0
images_count = {label: {x: 0 for x in data_cat} for label in [0, 1]}
for phase in data_cat:
    for data in tqdm(dataloaders[phase]):
        label = data['label'].numpy()[0]
        images_count[label][phase] += len(data['images'][0])
print(images_count)

Wt1 = {x: n_p(images_count[0][x] / (images_count[0][x] + images_count[1][x])) for x in data_cat}
Wt0 = {x: n_p(images_count[1][x] / (images_count[0][x] + images_count[1][x])) for x in data_cat}

print('Wt0:', Wt0)
print('Wt1:', Wt1)

class Loss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(Loss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0
        
    def forward(self, inputs, targets, phase):
        loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
        return loss

model = densenet169(pretrained=True)
model = model.cuda()



criterion = Loss(Wt1, Wt0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)


# #### Train model


model = train_model(model, criterion, optimizer, dataloaders, scheduler, num_epochs=20, v2=True)
# get_metrics(model, criterion, dataloaders, v2=True)

# save the model state
torch.save(model.state_dict(), 'model.pth')
