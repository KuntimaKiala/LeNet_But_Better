import torch 
from torch import nn 

class LeNet(nn.Module) :
    
    def __init__(self, out_l1=120, out_l2=84, output_size=10) :
        super(LeNet,self).__init__()
        self.out_fcl1 = out_l1
        self.out_fcl2 = out_l2
        self.output_size = output_size
        
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(6,16, kernel_size=(5,5), padding=(0,0), stride=1), 
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        
        
        
        self.dense = nn.Flatten()
        
        self.fcl_1 = nn.Sequential(nn.Linear(16*5*5, self.out_fcl1),
                                   nn.Dropout(p=0.5),
                                   nn.ReLU())
        
        self.fcl_2 = nn.Sequential(nn.Linear(self.out_fcl1,self.out_fcl2),
                                   nn.Dropout(p=0.3),
                                   nn.ReLU(),
                                   ) 
        
        self.fcl_3 = nn.Sequential(nn.Linear(self.out_fcl2,self.out_fcl2//2),
                                   nn.Dropout(p=0.1),
                                   nn.ReLU()) 
        
        self.fcl_4 = nn.Sequential(nn.Linear(self.out_fcl2//2,self.output_size)) 
        

        
        self.head = nn.Softmax(dim=1)
        
        self.net = nn.Sequential(self.conv_layer_1,
                                 self.conv_layer_2,
                                 self.dense,
                                 self.fcl_1,
                                 self.fcl_2, 
                                 self.fcl_3,
                                 self.fcl_4,
                                 self.head)
        
    def forward(self, x) :
        x = self.net(x)
        return x
        