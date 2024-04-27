from torch.utils.data import Dataset, DataLoader
from pydub import AudioSegment
import numpy as np


class NoiseDataset(Dataset):
    def __init__(self):
        self.seg_list = []
        self.rev_list = []
        n = 6
        for i in range(1,n+1):
            path = 'noise/noise_sample_%d.wav' % i
            audio = AudioSegment.from_wav(path)
            for i in range(0, len(audio) - 500, 500):
                audio_seg = audio[i:i+1000]
                # print(audio_seg.get_array_of_samples()[0:10])
                self.seg_list.append(np.array(audio_seg.get_array_of_samples()))
                self.rev_list.append(np.array(audio_seg.invert_phase().get_array_of_samples()))
                
    def __len__(self):
        return len(self.seg_list)
    
    def __getitem__(self, idx):
        return {'seg':self.seg_list[idx], 'rev':self.rev_list[idx]}


class TestDataset(Dataset):
    def __init__(self):
        self.seg_list = []
        n = 1
        path = 'noise/test_noise_%d.wav' % n
        audio = AudioSegment.from_wav(path)
        # print(len(audio))
        for i in range(0, len(audio) - 1000 + 1, 1000):
            # print(i)
            audio_seg = audio[i:i+1000]
            # print(audio_seg.get_array_of_samples()[0:10])
            self.seg_list.append(np.array(audio_seg.get_array_of_samples()))

    def __len__(self):
        return len(self.seg_list)
    
    def __getitem__(self, idx):
        return self.seg_list[idx]


# ds = TestDataset()
# print('total:', len(ds))
# dl = DataLoader(ds, batch_size=4, shuffle=False)
# for i, data in enumerate(dl):
#     print(data.shape)
#     break
# ds =  NoiseDataset()
# print('total:', len(ds))
# dl = DataLoader(ds, batch_size=4)
# for i, data_dict in enumerate(dl):
#     print(data_dict['seg'].shape)
#     print(data_dict['rev'].shape)
#     break