from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import nn, optim
import glob

DATA_DIR = '' 
DATA_DIR_IMAGE = ''    
DATA_SAVE_MODEL = DATA_DIR_IMAGE    #'/scratch/nas_spiders/andreasante/Models/'
dtype = torch.float
#device = torch.device("cpu")
device = torch.device("cuda:0")  # Uncomment this to run on GPU

files = glob.glob(DATA_DIR + 'pulse_DM*.npy')+glob.glob(DATA_DIR + 'pulse_SNR*.npy')
label_file = glob.glob(DATA_DIR + 'pulse_labels_DM*.npy')+glob.glob(DATA_DIR + 'pulse_labels_SNR*.npy')
X_train =np.zeros((1,128,128)) 
labels_array = []

X_val = np.zeros((1,128,128))
val_labels_array = []

for file in files:
    images_in_file = np.load(file)
    images_in_file = images_in_file
    X_train = np.vstack((X_train,images_in_file))

for file in label_file:
    labels = np.load(file)
    labels_array.append(labels)

labels_array = np.array(labels_array)
labels_array.shape=(labels_array.size,1)


X_train = X_train[1::]

validation_files = glob.glob(DATA_DIR+'validation_pulse_DM*.npy')+glob.glob(DATA_DIR+'validation_pulse_SNR*.npy')

validation_labels_files = glob.glob(DATA_DIR+'validation_pulse_labels_DM*.npy')+glob.glob(DATA_DIR+'validation_pulse_labels_SNR*.npy')


for file in validation_files:
    images_in_file = np.load(file)
    images_in_file = images_in_file
    X_val = np.vstack((X_val,images_in_file))

for file in validation_labels_files:
    validation_labels = np.load(file)
    val_labels_array.append(validation_labels)

val_labels_array = np.array(val_labels_array)
val_labels_array.shape =(val_labels_array.size,1)

X_val = X_val[1::]

train_images = X_train.reshape(X_train.shape[0],1,128,128)
train_images_tensor = torch.tensor(train_images, device=device, dtype=dtype)
train_labels = labels_array
train_labels_tensor = torch.tensor(train_labels, device=device, dtype=dtype)
print(train_images_tensor.size(), train_labels_tensor.size())
train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size=2, shuffle=True)

validation_images = X_val.reshape(X_val.shape[0],1,128,128)
validation_images_tensor = torch.tensor(validation_images, device=device, dtype=dtype)
validation_labels = val_labels_array
validation_labels_tensor = torch.tensor(validation_labels, device=device, dtype=dtype)
validation_tensor = TensorDataset(validation_images_tensor, validation_labels_tensor)
validation_loader = DataLoader(validation_tensor, batch_size=2, shuffle=True)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
        
    def forward(self, x):

        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=false)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self,x):
        identity = x.clone()
        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dropout = nn.Dropout2d(0.2)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=2)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 =self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Sequential(
            nn.Linear(512*ResBlock.expansion, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,num_classes)
            )

    def forward(self,x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

       # x = self.dropout(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=1):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    

model = ResNet50(1, channels=1)
model.cuda()


loss_function = nn.MSELoss()

num_epoch = 100

learning_rate = 3.46e-5

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = []
validation_loss_list = list()

train_loss_cycle = list()
validation_loss_cycle = list()
counter = 1
previous_std = 1

for epoch in range(num_epoch):
    running_loss = 0
    
    validation_loss = 0
    counter = counter+1
        
            
    for images, labels in train_loader:
    
        optimizer.zero_grad()
        
        output=model(images)

        loss = loss_function(output,labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

        train_loss_cycle.append(loss.item())

    for images, labels in validation_loader:
        
        output=model(images)
    
        loss = loss_function(output,labels)
        
        validation_loss += loss.item()

        validation_loss_cycle.append(loss.item())
    
    train_mean_loss = running_loss/len(train_loader)
    val_mean_loss = validation_loss/len(validation_loader)

    validation_loss_list.append(val_mean_loss)
    train_loss_list.append(train_mean_loss)
    print('Epoch: ', epoch, 'Train Mean Loss: ', train_mean_loss)
    
    if epoch > 9:
        std = np.std(train_loss_list[epoch-4:epoch])
        mean = np.mean(train_loss_list[epoch-4:epoch])
        
        if train_loss_list[epoch]+500>mean-std+500:
            learning_rate = learning_rate*0.75
            optimizer = optim.Adam(model.parameters(), lr = learning_rate)
            print('Current learning rate: ', learning_rate)
            

train_loss_array = np.array(train_loss_list)
validation_loss_array = np.array(validation_loss_list)

train_loss_cycle_array = np.array(train_loss_cycle)
validation_loss_cycle_array = np.array(validation_loss_cycle)

torch.save(model.state_dict(), DATA_SAVE_MODEL+'RN50_ALL_SNR_50epochss.pth')


plt.plot(np.arange(9,num_epoch),train_loss_array[9:num_epoch], label='Training Data')
plt.plot(np.arange(9,num_epoch),validation_loss_array[9:num_epoch], label='Validation Data')
plt.title('Mean loss per epoch for train and validation data')
plt.ylabel('Average M.E.L')
plt.xlabel('Number of Epochs')
plt.legend()
plt.savefig(DATA_DIR_IMAGE+'RN50_ALL_SNR_50epochs.png',dpi=600)
plt.show()  

