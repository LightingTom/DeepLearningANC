import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import torch
from networks import Net
import time


net = Net()
net.eval()
net.load_state_dict(torch.load('model_100.pt'))
print('Finish loading model')

# audio_data is np array
def calculate_anti_noise(audio_data):
    # print('ori:', audio_data[0:100])
    # audio_seg = AudioSegment(
    #     data=audio_data.tobytes(),
    #     sample_width=2,
    #     frame_rate=44100,
    #     channels=1
    # )
    # # print(np.array(audio_seg.get_array_of_samples())[0:100])
    # audio_seg = audio_seg.set_frame_rate(16000)
    # inp = np.array(audio_seg.get_array_of_samples())
    # inp = torch.from_numpy(inp).float().unsqueeze(0)
    inp = torch.from_numpy(audio_data).float()
    # print('inp:', inp.shape)
    # 1,16000
    stft_out = torch.stft(inp, n_fft=320, win_length=320, hop_length=160, return_complex=False)
    stft_out = stft_out.permute((0,3,1,2))
    start = time.time()
    with torch.no_grad():
        net_out = net(stft_out)
    end = time.time()
    print('Model inference time:', end - start)
    # approximately 0.6-0.8s
    x_real = net_out[0].permute((0,2,1))
    x_imag = net_out[1].permute((0,2,1))
    x = torch.complex(x_real, x_imag)
    out = torch.istft(x, n_fft=320, win_length=320, hop_length=160, return_complex=False)
    out = out.numpy().astype(np.int16).reshape((-1,))
    out_seg = AudioSegment(
         data=out.tobytes(),
         sample_width=2,
         frame_rate=16000,
         channels=1
    )
    # out_seg = out_seg.set_frame_rate(44100)
    out_arr = np.array(out_seg.get_array_of_samples())
    # out_arr = np.append(out_arr, np.array([0,0]))
    return out_arr.reshape((-1,1))
    



def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
        
    
    # Convert the input data to a format suitable for your model
    # print('indata:', indata.shape)
    # indata: nparray; shape: 44100, 1
    audio_data = np.array(indata[:, 0], dtype=np.int16).reshape((1, -1))  # Assuming mono input
    # print(audio_data.shape)
    anti_noise = calculate_anti_noise(audio_data)
    
    # Process the data through your model to get the anti-noise signal
    # anti_noise = process_with_model(audio_data)
    
    # Calculate the noise-cancelled output
    # output_audio = audio_data - anti_noise
    
    # Place the processed audio into outdata for playback
    outdata[:] = anti_noise

print("default:", sd.default.device)
print(sd.query_devices())
try:
    with sd.Stream(callback=audio_callback,
                   samplerate=16000,
                   channels=1,
                   blocksize=16000,
                   dtype='int16'):
        print("Streaming started. Press Ctrl+C to stop.")
        while True:
            pass
except KeyboardInterrupt:
        print("Streaming stopped.")
except Exception as e:
        print("An error occurred:", e)