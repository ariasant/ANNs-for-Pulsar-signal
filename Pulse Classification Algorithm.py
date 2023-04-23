from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import nn, optim
import glob


DATA_DIR = '' #directory where images are stored

files = glob.glob(DATA_DIR + 'simulated_pulse*.npy')
label_file = glob.glob(DATA_DIR + 'labels*.npy')
X_train =np.zeros((1,128,128)) 
labels = np.zeros(1) 

for file in files:
    X_train = np.vstack((X_train,np.load(file)))

for label in label_file:

    labels = np.vstack((labels,np.load(label)))

X_train = X_train[1::]
labels = labels[1::]

X_val = np.load(DATA_DIR + 'validation.npy')
val_labels = np.load(DATA_DIR + 'validation_label.npy')

train_images = X_train.reshape(X_train.shape[0],1,128,128)
train_images_tensor = torch.tensor(train_images).float()
train_labels = np.reshape(labels, -1)
train_labels_tensor = torch.tensor(train_labels).long()
train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_tensor, batch_size = 10, shuffle=True)

validation_images = X_val.reshape(X_val.shape[0],1,128,128)
validation_images_tensor = torch.tensor(validation_images).float()
validation_labels = np.reshape(val_labels, -1)
validation_labels_tensor = torch.tensor(validation_labels).long()
validation_tensor = TensorDataset(validation_images_tensor, validation_labels_tensor)
validation_loader = DataLoader(validation_tensor, batch_size = 10, shuffle=True)


model = nn.Sequential(
        
        nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),
	nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(kernel_size=4, stride=4),
        nn.Flatten(),
        nn.Linear(4*4*512,150),
        nn.Dropout2d(0.5),
        nn.Linear(150,50),
        nn.Dropout2d(0.5),
        nn.Linear(50,2),
        nn.LogSoftmax(dim=1))


loss_function = nn.CrossEntropyLoss()

num_epoch = 20

learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list = list()
validation_loss_list = list()

train_loss_cycle = list()
validation_loss_cycle = list()

for epoch in range(num_epoch):
     
    running_loss = 0
    
    validation_loss = 0

    counter = 0
        
    for images, labels in train_loader:
        
        optimizer.zero_grad()
        
        output=model(images)

        loss = loss_function(output,labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

        train_loss_cycle.append(loss.item())

        print(counter)

        counter += 1

    for images, labels in validation_loader:
        
        output=model(images)
    
        loss = loss_function(output,labels)
        
        validation_loss += loss.item()

        validation_loss_cycle.append(loss.item())
    
    train_mean_loss = running_loss/len(train_loader)
    val_mean_loss = validation_loss/len(validation_loader)

    train_loss_list.append(train_mean_loss)
    validation_loss_list.append(val_mean_loss)

    print('Epoch: ', epoch, 'Train Mean loss: ', train_mean_loss, 'Validation Mean loss: ', val_mean_loss)
    counter = 0

train_loss_array = np.array(train_loss_list)
validation_loss_array = np.array(validation_loss_list)

train_loss_cycle_array = np.array(train_loss_cycle)
validation_loss_cycle_array = np.array(validation_loss_cycle)

plt.plot(np.arange(0,num_epoch),train_loss_array,label='Train set')
plt.plot(np.arange(0,num_epoch),validation_loss_array,label='Validation set')
plt.title('Mean loss per epoch for train and validation data')
plt.xlabel('Number of epochs')
plt.ylabel('Mean Loss')
plt.legend()
plt.savefig(DATA_DIR+'Mean_loss_train_vs_validation_SNR_5_50.jpeg',dpi=400)
plt.show()

print('Training done')  

torch.save(model, DATA_DIR+'CNN_5_50_SNR.pth')
