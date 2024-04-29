# Active Noise Cancellation of Drone Propeller Noise using machine learning

## Code instruction

To run the code, make sure to have the correct dependencies:

```bash
pip install -r requirements.txt
```

### Training

To train the GCRN model, change line16 in ``model.py`` to ``choice = 'GCRN'``, then run `python model.py`

To train the LSTM model, change line16 in `model.py` to `choice = 'simple'`, then run ``python model.py``

After training, you will get a `.png` file and a `.txt`  of the loss, a `.wav` file, which is the anti-noise signal, and a `.pt` file of the trained model.

### Testing

If you want to test a specific `.pt` file, you can use `python test_model.py`. Remeber to change the path to the `.pt` file and the model type (please specify in `choice`).

### Real-time set up

You can run a real-time experiment using `python test_sd.py`. You can specify your model in line 11.



## Audio Results

The noise sample used for testing is `test_noise_1.wav`. 

The anti-noise signals generated are `recon_signal_GCRN_100.wav` and `recon_signal_simple_100.wav` corresponding to the 2 models.

The resulting sound is in `Result_sound` is the combination of the noise and anti-noise signal. The combination here means use different speaker to output the noise.



## Acknowledgement

The `network.py` is modified from https://github.com/JupiterEthan/GCRN-complex/tree/master.