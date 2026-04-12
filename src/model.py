import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import (
    CNN_DROPOUT,      
    CNN_LEARNING_RATE,
    device            
)


#Starting the simple CNN
class SimpleCNN(nn.Module):
  #Autofilled by copiolt
  def __init__(self):
    super(SimpleCNN, self).__init__()

    #First layer: Input wafer 1, output 32, with kernel 3x3
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding =1)
    #adding batch normalization
    self.bn1 = nn.BatchNorm2d(32)

    #Second layer: take 32, output 64, same kernel
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
    self.bn2 = nn.BatchNorm2d(64)

    #3rd layer
    self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
    self.bn3 = nn.BatchNorm2d(128)

    #4th layer
    self.conv4 = nn.Conv2d(in_channels = 128, out_channels =128, kernel_size = 3, padding = 1)
    self.bn4 = nn.BatchNorm2d(128)

    #5th layer
    self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding =1)
    self.bn5 = nn.BatchNorm2d(256)

    #reduce the size by half
    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    #Try other Pooling methods like avgpool or globalavgpool
    #self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)
    #self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))


    #added dropout for overfitting
    self.dropout = nn.Dropout(CNN_DROPOUT)
    #try different dropout rates or other regularization methods like weight decay or early stopping

    #connect layers; 13x13 because 52 -> 26 -> 13
    self.fc1 = nn.Linear(in_features = 256 * 1 * 1, out_features = 128)
    self.fc2Classify = nn.Linear(in_features = 128, out_features = 9)
    #self.fc3Regression = nn.Linear(in_features = 128, out_features = 8) #Regression

  def forward(self, x):
    #52 - 26
    x = self.pool(F.relu(self.bn1((self.conv1(x)))))
    #26 - 13
    x = self.pool(F.relu(self.bn2((self.conv2(x)))))
    #
    x = self.pool(F.relu(self.bn3((self.conv3(x)))))
    #Recommended dropout to deal with overfitting
    x = self.dropout(x)
    #
    x = self.pool(F.relu(self.bn4((self.conv4(x)))))
    #
    x = self.pool(F.relu(self.bn5((self.conv5(x)))))
    x = self.dropout(x)

    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    #understand different use of activation functions

    #apply dropout
    x = self.dropout(x)
    ClassificationResult = self.fc2Classify(x)
    #RegressionResult = self.fc3Regression(x)

    return ClassificationResult
  
def buildModel():
  #MIxed set
  model = SimpleCNN().to(device)
  #Loss function/ Added new loss for regression
  criterion_class = nn.BCEWithLogitsLoss()
  #criterion_regress = nn.MSELoss()
  #Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr = CNN_LEARNING_RATE) #Changing around learning rate
  return model, criterion_class, optimizer