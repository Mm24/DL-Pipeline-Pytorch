import torch as t
from Exercise_4.code_skeleton.data import ChallengeDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from Exercise_4.code_skeleton.trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import Exercise_4.code_skeleton.model as arch
import pandas as pd
from sklearn.model_selection import train_test_split
from Exercise_4.code_skeleton.model import ResNet

# Parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 2}
params2 = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 2}
max_epochs = 100
# has to be changed on local machine
#dataset = '/home/cip/medtech2017/za25qota/dl-challenge/Exercise_4/code_skeleton/data.csv'
dataset = '/home/cip/medtech2016/ij19okej/dl_4ex/Exercise_4/code_skeleton/data.csv'
mode_train = "train"
mode_val = "val"
# train test split
df_dataset = pd.read_csv(dataset, sep=';')
train_set, test_set = train_test_split(df_dataset, test_size=0.2, random_state=0)
dataset_train = ChallengeDataset(train_set, mode_train)
dataset_val = ChallengeDataset(test_set, mode_val)


training_generator = DataLoader(dataset_train, **params)
validation_generator = DataLoader(dataset_val, **params2)

classes = ('crack', 'inactive')

model = ResNet()

#criterion = torch.nn.MultiLabelMarginLoss()
#criterion = torch.nn.LogSoftmax
# criterion = torch.nn.MSELoss()
criterion = torch.nn.BCELoss()
#criterion = torch.nn.BCEWithLogitsLoss()

#criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.001)

# here trainer
train = Trainer(model=model, crit=criterion, optim=optimizer, train_dl=training_generator, val_test_dl=validation_generator,
                early_stopping_patience=80)

res = train.fit(60)
print("Train loss:")
print(res[0])

print("Val loss:")
print(res[1])
# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.title('SGD lr 0.001 batchsize 16 BCE loss')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig('plots/losses.png')