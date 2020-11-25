from instrument_recognition.models.torchopenl3 import OpenL3Mel128
from instrument_recognition.models.timefreq import Melspectrogram

openl3_torch = OpenL3Mel128(maxpool_kernel=(16, 24), use_kapre=True)
newspec = Melspectrogram()

#-----------------------------
#-----------------------------
#-----------------------------

import openl3
import torch 
openl3_keras = openl3.models.load_audio_embedding_model(input_repr='mel128', content_type='music', embedding_size=512)

# openl3_keras.summary()
#-----------------------------
#-----------------------------
#-----------------------------

# inp = openl3_keras.get_layer('batch_normalization_9').input
# oup = openl3_keras.get_layer('max_pooling2d_8').output

# from keras import Model, Input

# inp = Input(shape=(128, 199, 1), dtype='float32')
# layers = [l for i, l in enumerate(openl3_keras.layers) if i > 1 and i < (len(openl3_keras.layers) - 1)]
# x = inp
# for l in layers:
#     x = l(x)

# openl3_nospec = Model(inputs=[inp], outputs=[x])

import numpy as np
openl3_torch.freeze()

import torchaudio
import instrument_recognition.utils.audio as audio_utils
# random audio
audio, sr = torchaudio.load('/home/hugo/CHONK/data/philharmonia/all-samples/banjo/banjo_Cs5_very-long_forte_normal.mp3')
audio = audio_utils.resample(audio.unsqueeze(0), sr, 48000).numpy()[:, :, 0:48000]
audio = torch.randn(1, 1, 48000).numpy()

torch_result = openl3_torch(openl3_torch.melspec(audio)).permute(0, 2, 3, 1).reshape(-1).detach().numpy()
torch_newspecresult = openl3_torch(newspec(torch.from_numpy(audio).float())).permute(0, 2, 3, 1).reshape(-1).detach().numpy()
# keras_nospec_result = openl3_nospec.predict(openl3_torch.melspec(audio).detach().numpy().transpose(0, 2, 3, 1)).flatten()
keras_result = openl3_keras.predict(audio).flatten()
keras_result, _ = openl3.get_audio_embedding(audio[0][0], 48000, center=False, input_repr='mel128', content_type='music', embedding_size=512)

#----
keras_melspec = openl3_torch.melspec(audio)
new_melspec = newspec(torch.from_numpy(audio).float())
#---
print(35*'-')
print(35*'-')
print('TESTING TORCH VS KERAS')
print(35*'-')
print(35*'-')
print('SPECTROGRAMS')
print(35*'-')
print('TORCH')
print(f'\tmean: {( new_melspec ).mean()}')
print(f'\tstd: {( new_melspec ).std()}')
print('KERAS')
print(f'\tmean: {( keras_melspec ).mean()}')
print(f'\tstd: {( keras_melspec ).std()}')
print('ME - KERAS')
print(f'\tmean: {( keras_melspec    -   new_melspec).mean()}')
print(f'\tstd: {( keras_melspec    -   new_melspec).std()}')
print('PERCENT ERROR')

print(35*'-')
print('OPENL3 EMBEDDINGS')
print(35*'-')
print('TORCH')
print(f'\tmean: {( torch_newspecresult ).mean()}')
print(f'\tstd: {( torch_newspecresult ).std()}')
print('KERAS')
print(f'\tmean: {( keras_result ).mean()}')
print(f'\tstd: {( keras_result ).std()}')
print('KERAS - TORCH')
print(f'\tmean: {( keras_result   -    torch_newspecresult).mean()}')
print(f'\tstd: {(  keras_result   -    torch_newspecresult ).std()}')
print('PERCENT ERROR')
print(f'\error : {((keras_result   -    torch_newspecresult)/keras_result).mean()*100}%')
print('KL DIVERGENGE')
torch_newspecresult = torch.from_numpy(torch_newspecresult)
keras_result = torch.from_numpy(keras_result)
print(f'\kl div: {torch.nn.functional.kl_div(torch_newspecresult, keras_result)}')
#----

# lets see what a melspec looks like
# import torch
# r = openl3_torch.melspec(audio)
# print(torch.max(r))
# print(torch.min(r))


#-----------------------------
#-----------------------------
#-----------------------------

# # NOW WE NEED TO FIND A MELSPEC THAT IS THE SAME
# import torch
# original_melspec = openl3_torch.melspec(torch.tensor(audio))

# import torchaudio

# torch_spec = torchaudio.transforms.MelSpectrogram(
#     sample_rate=48000, 
#     n_fft=2048, 
#     hop_length=242, 
#     power=1,
#     n_mels=128, 
#     normalized=True
# )(torch.Tensor(audio))
# torch_spec = torchaudio.transforms.AmplitudeToDB('magnitude', top_db=80)(torch_spec)

# import librosa

# librosa_spec = librosa.feature.melspectrogram(audio[0][0], sr=48000, n_fft=2048, hop_length=242, power=2, htk=True)
# librosa_spec = librosa.core.amplitude_to_db(librosa_spec)

# print(librosa_spec)
# print(librosa_spec.mean())
# print(librosa_spec.std())

# print(original_melspec)
# print(original_melspec.mean())
# print(original_melspec.std())


# print(torch_spec)
# print(torch_spec.mean())
# print(torch_spec.std())

# print(( original_melspec - torch_spec.detach().numpy()).mean())

# # print(openl3_keras.summary())

# "Created Spectrogram and Melspectrogram layers as nn.Modules. 
# The Spectrogram layer consists of two 1D convolutional layers, each with N / 2 + 1 output channels. The layers are fixed to a DFT filterbank basis, with the option of having the filterbank being trainable.
# The Melspectrogram layer takes a Spectrogram from above and applies a linear transformation that maps from a DFT basis to a Mel basis. 
# The Melspectrogram layer is approximately equal to the Kapre Melspectrogram layer, with an error average of 0.18 and a standard deviation of 0.674. "

#----------------------------
# RESULTS (B4 Fine tuning)
#     -----------------------------------
#     -----------------------------------
#     TESTING TORCH VS KERAS
#     -----------------------------------
#     -----------------------------------
#     SPECTROGRAMS
#     -----------------------------------
#     TORCH
#             mean: -15.480896949768066
#             std: 2.1568427085876465
#     KERAS
#             mean: -15.457451820373535
#             std: 2.1554760932922363
#     ME - KERAS
#             mean: 0.023450544103980064
#             std: 0.36934223771095276
#     -----------------------------------
#     OPENL3 EMBEDDINGS
#     -----------------------------------
#     TORCH
#             mean: 1.371492862701416
#             std: 0.9290154576301575
#     KERAS
#             mean: 1.5489599704742432
#             std: 1.0386818647384644
#
#     KERAS - TORCH
#             mean: 0.1774664670228958
#             std: 0.6748143434524536