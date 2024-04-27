import torch
import torch.nn as nn
import torch.optim as op
from networks import Net
from simple_network import SimpleNet
from dataset import NoiseDataset, TestDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import io
import scipy.io.wavfile
from pydub import AudioSegment
from torch.optim.lr_scheduler import StepLR


# net = Net()
choice = 'GCRN'
if choice == 'simple':
    net = SimpleNet()
else:
    net = Net()
ds = NoiseDataset()
loader = DataLoader(ds, batch_size=16)
loss_fn = nn.MSELoss()
optimizer = op.Adam(net.parameters(), lr=1)
scheduler = StepLR(optimizer, step_size=30)
max_epoch = 100

epoch_list = []
loss_list = []
for e in range(max_epoch):
    total_loss = 0
    cnt = 0
    for _, data_dict in enumerate(loader):
        optimizer.zero_grad()
        # print(data_dict['seg'].dtype)
        stft_out = torch.stft(data_dict['seg'].float(), n_fft=320, win_length=320, hop_length=160, return_complex=False)
        stft_out = stft_out.permute((0,3,1,2))
        # print(stft_out.shape)
        # b,2,161,101
        with torch.enable_grad():
            out = net(stft_out)
        out_stacked = torch.stack((out[0], out[1]), dim=1).float()
        # print(out_stacked.shape)
        # b,2,101,161
        truth = torch.stft(data_dict['rev'].float(), n_fft=320, win_length=320, hop_length=160, return_complex=False)
        if choice == 'simple':
            truth = truth.permute((0,3,1,2))
        else:
            truth = truth.permute((0,3,2,1))
        loss = loss_fn(out_stacked, truth)
        total_loss += loss.item()
        cnt += 1
        loss.backward()
        optimizer.step()
    print('|Epoch: %d | Loss: %.5f |' % (e, total_loss / cnt))
    epoch_list.append(e+1)
    loss_list.append(total_loss / cnt)

print('Finish training')
with open('loss_%s.txt' % choice, 'w') as f:
    for i in range(len(epoch_list)):
        f.write(str(epoch_list[i]))
        f.write(' ')
        f.write(str(loss_list[i]))
        f.write('\n')

plt.plot(epoch_list, loss_list, '.-', color='r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_figure_%s.png' % choice)

# save model
torch.save(net.state_dict(), 'model_%d_%s.pt' % (max_epoch, choice))

# Testing
testset = TestDataset()
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
with torch.no_grad():
    res_list = []
    for _, data in enumerate(test_loader):
        stft_out = torch.stft(data.float(), n_fft=320, win_length=320, hop_length=160, return_complex=False)
        stft_out = stft_out.permute((0,3,1,2))
        net_out = net(stft_out)
        if choice == 'GCRN':
            x_real = net_out[0].permute((0,2,1))
            x_imag = net_out[1].permute((0,2,1))
        else:
            x_real = net_out[0]
            x_imag = net_out[1]
        x = torch.complex(x_real, x_imag)
        out = torch.istft(x, n_fft=320, win_length=320, hop_length=160, return_complex=False)
        res_list.append(out.numpy())
    recon_signal = np.concatenate(res_list).reshape((-1,)).astype(np.int16)
    recon_sound = AudioSegment(
        data=recon_signal.tobytes(),
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    recon_sound.export('recon_signal_%s_%d.wav' % (choice, max_epoch), format='wav')
