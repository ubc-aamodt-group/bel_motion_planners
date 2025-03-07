import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

'''
This file contains the neural network architectures used in the project.
The baseline architecture has three sizes: small, medium, and large.
'''
class MLP_lrg(nn.Module):
    def __init__(self, input_size, output_size,dp=0.5):
        super(MLP_lrg, self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(896, 768),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(768, 512),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(512, 384),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(384, 256),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(256, 256),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(256, 128),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(128, 64),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(64, 32),nn.PReLU(),
        nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out

class MLP_med(nn.Module):
    def __init__(self, input_size, output_size,dp=0.5):
        super(MLP_med, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(896, 512), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(512, 384), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(64, 32), nn.PReLU(),
                    nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out

class MLP_sml(nn.Module):
    def __init__(self, input_size, output_size,dp=0.5):
        super(MLP_sml, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(1280, 512), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(256, 32), nn.PReLU(),
                    nn.Linear(32, output_size))

    def forward(self, x):
        out = self.fc(x)
        return out

""" 
The BEL architecture has three sizes: small, medium, and large.
"""
class MLP_bel_lrg(nn.Module):
    def __init__(self, input_size, output_size,num_bits=40,dp=0.5):
        super(MLP_bel_lrg, self).__init__()
        self.num_joints=output_size
        self.fc = nn.Sequential(
        nn.Linear(input_size, 1280),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(1280, 1024),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(1024, 896),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(896, 768),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(768, 512),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(512, 384),nn.PReLU(),nn.Dropout(p=dp),
        nn.Linear(384, 256),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(256, 256),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(256, 128),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(128, 64),nn.PReLU(), nn.Dropout(p=dp),
        nn.Linear(64, 32),nn.PReLU())
        self.fclast=nn.ModuleList().cuda()
        for _ in range(0,output_size):
            self.fclast.append(nn.Sequential( nn.Linear(32, 20), nn.Linear(20, num_bits)))

    def forward(self, x):
        if(x.dim()==1):
            x=x.unsqueeze(0)
            
        out = self.fc(x)
        final = (self.fclast[0])(out)
        for fc in self.fclast[1:]:
            t= fc(out)
            final=torch.cat((final,t),dim=1)
        return final
    
class MLP_bel_med(nn.Module):
    def __init__(self, input_size, output_size,num_bits=40,dp=0.5):
        super(MLP_bel_med, self).__init__()
        self.num_joints=output_size
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(1280, 896), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(896, 512), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(512, 384), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(384, 256), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(64, 32), nn.PReLU())
        self.fclast=nn.ModuleList().cuda()
        for _ in range(0,output_size):
            self.fclast.append(nn.Sequential( nn.Linear(32, 20), nn.Linear(20, num_bits)))

    def forward(self, x):
        if(x.dim()==1):
            x=x.unsqueeze(0)
            
        out = self.fc(x)
        final = (self.fclast[0])(out)
        for fc in self.fclast[1:]:
            t= fc(out)
            final=torch.cat((final,t),dim=1)
        return final

class MLP_bel_sml(nn.Module):
    def __init__(self, input_size, output_size,num_bits=40,dp=0.5):
        super(MLP_bel_sml, self).__init__()
        self.num_joints=output_size
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(1280, 512), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(p=dp),
                    nn.Linear(256, 32), nn.PReLU())
        self.fclast=nn.ModuleList().cuda()
        for _ in range(0,output_size):
            self.fclast.append(nn.Sequential( nn.Linear(32, 20), nn.Linear(20, num_bits)))

    def forward(self, x):
        if(x.dim()==1):
            x=x.unsqueeze(0)
        out = self.fc(x)
        final = (self.fclast[0])(out)
        for fc in self.fclast[1:]:
            t= fc(out)
            final=torch.cat((final,t),dim=1)
        return final

"""
This is the encoder architecture used in the Kinova experiments.
"""        

class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 786), nn.PReLU(),
                                     nn.Linear(786, 512), nn.PReLU(),
                                     nn.Linear(512, 256), nn.PReLU(),
                                     nn.Linear(256, output_size))

    def forward(self, x):
        x = self.encoder(x)
        return x

