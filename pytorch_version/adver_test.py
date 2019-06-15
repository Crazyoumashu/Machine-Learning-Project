# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:48:45 2019

@author: 黄浩威
"""

from __future__ import print_function

import itertools
import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from adversarialbox.utils import to_var

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
from PIL import Image
from sklearn.metrics import confusion_matrix
path = "FER2013_VGG19"
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def perturb(X_nat, y, epsilon):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        '''
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons
        '''
        X = np.copy(X_nat)

        X_var = to_var(torch.from_numpy(X), requires_grad=True)
        y_var = to_var(torch.LongTensor(y))
        '''
        scores = self.model(X_var)
        loss = self.loss_fn(scores, y_var)
        '''
        Modell = VGG('VGG19')
        Modell.cuda()
        scores = Modell(X_var)
        #loss = Modell.loss_fn(scores,y_var)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores,y_var)
        loss.backward()
        grad_sign = X_var.grad.data.cpu().sign().numpy()

        X += epsilon * grad_sign
        X = np.clip(X, 0, 1)

        return X


net = VGG('VGG19')
net.load_state_dict(torch.load('./FER2013_VGG19/PublicTest_model.t7',map_location='cpu')['net'])
net.cuda()
cut_size = 44
transform_train = transforms.Compose([
    #transforms.RandomCrop(44),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #transforms.ToTensor(),
])
'''
trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=128, shuffle=False, num_workers=0)
'''
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=128, shuffle=False, num_workers=0)
correct = 0
total = 0
all_target = []
for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):

    print(np.shape(inputs))
    print(np.shape(targets))
    bs, ncrops, c, h, w = np.shape(inputs)
    inputs = np.reshape(inputs,newshape = [10*bs,3,44,44])
    label = []
    for i in range(bs):
        for j in range(10):
            label.append(targets[i].item())
    label = torch.LongTensor(label)
    Returnnn = perturb(inputs,label,0.05)
    Returnnn = torch.Tensor(Returnnn)
    inputs = np.reshape(Returnnn, newshape = [bs,10,3,44,44])
    inputs = inputs.view(-1, c, h, w)
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    outputs = net(inputs)

    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
    _, predicted = torch.max(outputs_avg.data, 1)

    total += targets.size(0)
    correct += predicted.eq(targets.data).cpu().sum()
    if batch_idx == 0:
        all_predicted = predicted
        all_targets = targets
    else:
        all_predicted = torch.cat((all_predicted, predicted),0)
        all_targets = torch.cat((all_targets, targets),0)
acc = 100. * correct / total
print("accuracy: %0.3f" % acc)

# Compute confusion matrix
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                      title=' Confusion Matrix (Accuracy: %0.3f%%)' %acc)
plt.savefig(os.path.join(path, '1pr_cm.png'))
plt.close()
'''
dataiter = iter(PrivateTestloader)
imgs, labels = next(dataiter)
unloader = transforms.ToPILImage()
print(np.shape(imgs))
print(np.shape(labels))
imgs = np.reshape(imgs,newshape = [1280,3,44,44])
image = unloader(imgs[0])
print(labels[0].item())
label = []
for i in range(128):
    for j in range(10):
        label.append(labels[i].item())
label = torch.Tensor(label)
print(np.shape(label))
print(label)
plt.imshow(image)
plt.pause(1)
'''
#images = unloader(imgs)


'''
Returnnn = perturb(imgs,labels,0.05)
Returnnn = torch.Tensor(Returnnn)
print(np.shape(Returnnn))

image = unloader(Returnnn[0])
plt.imshow(image)
plt.pause(1)
'''
print("hello")
