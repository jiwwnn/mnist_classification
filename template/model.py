
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv = nn.Sequential(
            # conv1 (5*5)*1*6 + 6 = 156
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # conv2 (5*5*6*16) + 16 = 2416
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
            # fc1 (16*5*5*120) + 120 = 48120
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            
            # fc2 (120*84) + 84 = 10164
            nn.Linear(120, 84),
            nn.ReLU(),
            
            # fc3 (84*10) + 10 = 850
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )
        
    # Total Number of Parameters = 156 + 2416 + 48120 + 10164 + 850 = 61706

    def forward(self, img):
        x = self.conv(img)
        x = x.view(-1, 16*5*5)
        output = self.fc(x)
        return output


class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP,self).__init__()
        
        self.fc = nn.Sequential(
            # fc1 (32*32*58) + 58 = 59450
            nn.Linear(32*32, 58),
            nn.ReLU(),
            
            # fc2 (58*33) + 33 = 1947    
            nn.Linear(58, 33),
            nn.ReLU(),
            
            # fc3 (33*10) + 10 = 340
            nn.Linear(33,10),
            nn.Softmax(dim=1)
        )
    # Total Number of Parameters = 59450 + 1947 + 340 = 61737

    def forward(self, img):
        img = img.reshape(-1, 32*32)
        output = self.fc(img)
        return output
    

class LeNet5_Regularized(nn.Module):
    def __init__(self):
        super(LeNet5_Regularized, self).__init__()
        
        self.conv = nn.Sequential(
            # conv1 (5*5)*1*6 + 6 = 156
            nn.Conv2d(1, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2),
            
            # conv2 (5*5*6*16) + 16 = 2416
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
            # fc1 (16*5*5*120) + 120 = 48120
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # fc2 (120*84) + 84 = 10164
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # fc3 (84*10) + 10 = 850
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )
    # Total Number of Parameters = 156 + 2416 + 48120 + 10164 + 850 = 61706
    
    def forward(self, img):
        x = self.conv(img)
        x = x.view(-1, 16*5*5)
        output = self.fc(x)
        return output



