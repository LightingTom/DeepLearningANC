import torch
from networks import Net
from simple_network import SimpleNet
from dataset import NoiseDataset, TestDataset
from torch.utils.data import DataLoader
import numpy as np
from pydub import AudioSegment

choice = 'GCRN'
if choice == 'simple':
    net = SimpleNet()
else:
    net = Net()
# please modify the file name here corresponding to your trained model
net.load_state_dict(torch.load('model_100.pt'))
testset = TestDataset()
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
with torch.no_grad():
    res_list = []
    for _, data in enumerate(test_loader):
        # print('data:', data.shape)
        # b, 16000
        stft_out = torch.stft(data.float(), n_fft=320, win_length=320, hop_length=160, return_complex=False)
        # print(stft_out.shape)
        # b,161,101,2
        stft_out = stft_out.permute((0,3,1,2))
        net_out = net(stft_out)
        if choice == 'GCRN':
            x_real = net_out[0].permute((0,2,1))
            x_imag = net_out[1].permute((0,2,1))
        else:
            x_real = net_out[0]
            x_imag = net_out[1]
        x = torch.complex(x_real, x_imag)
        # print('x:', x.shape)
        # 1, 161, 101
        out = torch.istft(x, n_fft=320, win_length=320, hop_length=160, return_complex=False)
        res_list.append(out.numpy())
    recon_signal = np.concatenate(res_list).reshape((-1,)).astype(np.int16)
    print(recon_signal.shape)
    recon_sound = AudioSegment(
        data=recon_signal.tobytes(),
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    recon_sound.export('recon_signal_%s.wav' % choice, format='wav')