from pydub import AudioSegment
from scipy.signal import stft, istft
import numpy as np
import io
import scipy.io.wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.pyplot as plt

# generate graph
# s_list = []
# g_list = []
# e_list = []
# with open('loss_simple.txt', 'r') as f:
#     arr = f.readlines()
#     for i in arr:
#         e_list.append(int(i.split(' ')[0]))
#         s_list.append(float(i.split(' ')[1]))
# with open('loss_GCRN.txt', 'r') as f:
#     arr = f.readlines()
#     for i in arr:
#         g_list.append(float(i.split(' ')[1]))

# plt.plot(e_list, s_list, '.-', color='b')
# plt.plot(e_list, g_list, '.-', color='r')
# plt.legend(('LSTM', 'GCRN'), loc='center right')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('loss_combined.png')
# audio = AudioSegment.from_wav('noise/noise_sample_5.wav')
# arr = np.array(audio.get_array_of_samples())
# wav_io = io.BytesIO()
# scipy.io.wavfile.write(wav_io, 16000, arr)
# wav_io.seek(0)
# recon_sound = AudioSegment.from_wav(wav_io)
# recon_sound.export('noise5_recon.wav', format='wav')

# audio1 = AudioSegment.from_wav('code/recon_signal.wav')
# audio2 = AudioSegment.from_wav('code/recon_stft_small.wav')
# print(audio1.sample_width)

# total = 0
# cnt = 0
# arr1 = np.array(audio1.get_array_of_samples())
# arr2 = np.array(audio2.get_array_of_samples())
# print(audio1.get_array_of_samples()[0:100])

# print('audio:', len(audio))
# arr = audio.get_array_of_samples()
# for i in arr[0:100]:
#     print(i)
# inverse_sound = AudioSegment.from_wav('code/recon_signal.wav')
# inverse_sound -= 5

# n = 6
# for i in range(1, n+1):
#     path = 'noise/noise_sample_%d.wav' % i
#     audio = AudioSegment.from_wav(path)
#     invert = audio.invert_phase()
#     invert.export('noise_sample_%d_invert.wav' % i)

# cnt = 0
# total = 0
# zero = 0
# # for i in audio.get_array_of_samples():
# #     if i == 0:
# #         zero += 1
# #     total += i
# #     cnt += 1
# print(total / cnt)
# print(zero)

# arr1 = np.array(audio.get_array_of_samples())
# arr2 = np.array(inverse_sound.get_array_of_samples())
# for i in range(10):
#     print(arr1, ' ', arr2)
# USE sounddevice
# arr_ori = np.array(audio.get_array_of_samples())
# sd.play(arr_ori, audio.frame_rate)
# sd.wait()
# arr_inv = np.array(inverse_sound.get_array_of_samples())
# print('ori:', arr_ori.shape)
# print('inv:', arr_inv.shape)
# combine = np.column_stack((arr_ori, arr_inv))
# print('combine:', combine.shape)
# fs = audio.frame_rate
# sd.play(combine, fs)
# sd.wait()

# inverse_sound.export('inverse.wav', format='wav')

# Combine res and test
audio = AudioSegment.from_wav('noise/test_noise_1.wav')
inverse_sound = AudioSegment.from_wav('recon_signal_simple_100.wav')
combine = AudioSegment.from_mono_audiosegments(audio, inverse_sound)
combine.export('combined_lstm.wav', format='wav')


# overlay：叠加
# combine2 = audio.overlay(inverse_sound)
# arr = combine2.get_array_of_samples()
# total = 0
# for i in arr:
#     total += i
# print('total:', total)
# combine2.export('reduction2.wav', format='wav')

# audio = AudioSegment.from_file("noise/test_noise1.m4a")[0:1000]
# print('ori frame rate:', audio.frame_rate)
# print(np.array(audio.get_array_of_samples()).shape)
# new_audio = audio.set_frame_rate(16000)
# print('new audio:', np.array(new_audio.get_array_of_samples()).shape)



# new_audio = new_audio[500:14500]
# print('len:', np.array(new_audio.get_array_of_samples()).shape)
# seg = new_audio[0:20]
# print('frame rate:', audio.frame_rate)
# print('channels:', audio.channels)
# new_audio.export("noise/test_noise_1.wav", format='wav')
# raw_arr = new_audio.get_array_of_samples()
# arr = np.array(new_audio.get_array_of_samples())
# arr_seg = np.array(seg.get_array_of_samples())
# print('seg:', arr_seg.shape)
# # print(arr == 0)
# print('raw:', raw_arr[0])
# print('arr:', arr[0])
# freqs, times, zxx = stft(arr_seg, fs=16000, nperseg=320, noverlap=160)
# plt.pcolormesh(times, freqs, np.abs(zxx), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# x_real = np.real(zxx)
# x_imag = np.imag(zxx)
# print('real:', x_real.shape)
# print('imag:', x_imag.shape)
# print('freqs:', freqs.shape)
# print('times:', times.shape)

# _, reconstructed = istft(x_real + 1j*x_imag, fs=16000, nperseg=320, noverlap=160)
# print(reconstructed.shape)
# wav_io = io.BytesIO()
# scipy.io.wavfile.write(wav_io, 16000, reconstructed)
# wav_io.seek(0)
# recon_sound = AudioSegment.from_wav(wav_io)
# recon_sound.export('recon.wav', format='wav')
# print('recon:', reconstructed.dtype)
# same = 0
# cnt = 0
# total = 0
# for i in range(len(reconstructed)):
#     if reconstructed[i] - arr[i] == 0:
#         same += 1
#     else:
#         cnt += 1
#         total += abs(reconstructed[i] - arr[i])
# print('same:', same)
# print('avg:', total / cnt)
# reconstructed_audio = AudioSegment(
#     reconstructed,
#     frame_rate=16000,
#     sample_width=new_audio.sample_width,
#     channels=1
# )
# reconstructed_audio.export('recons.wav', format='wav')
