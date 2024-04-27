import torch.nn as nn
import torch
from dataset import NoiseDataset

class SimpleNet(nn.Module):
    def __init__(self, in_feature=101, hidden_size=256):
        super(SimpleNet, self).__init__()
        self.real_part_lstm = nn.LSTM(in_feature, hidden_size, batch_first=True)
        self.imag_part_lstm = nn.LSTM(in_feature, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.real_part_fc = nn.Linear(hidden_size, in_feature)
        self.imag_part_fc = nn.Linear(hidden_size, in_feature)
    
    def forward(self, x):
        x = x.permute((1,0,2,3))
        r, i = x[0], x[1]
        r,_ = self.real_part_lstm(r)
        i,_ = self.imag_part_lstm(i)
        r_out = self.real_part_fc(r)
        i_out = self.imag_part_fc(i)
        return self.relu(r_out), self.relu(i_out)