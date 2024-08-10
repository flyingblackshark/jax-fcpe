import audax.core
import audax.core.functional
import audax.core.stft
import jax.numpy as jnp
from models import CFNaiveMelPE
import librosa
import audax
from librosa.filters import mel as librosa_mel_fn

wav, sr = librosa.load("test.wav", sr=16000)
model = CFNaiveMelPE(128,360,512,6,8,1975.5,32.7,True,0.1,0.1)
from convert import load_params
params = load_params()
wav = jnp.asarray(wav)
WIN_SIZE = 1024
HOP_SIZE = 160
N_FFT = 1024
NUM_MELS = 128
window = jnp.hanning(WIN_SIZE)
pad_size = (WIN_SIZE-HOP_SIZE)//2
wav = jnp.pad(wav, (pad_size, pad_size),mode="reflect")
f0_min = 80.
f0_max = 880.

spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
mel_basis = librosa_mel_fn(sr=sr, n_fft=N_FFT, n_mels=NUM_MELS, fmin=0, fmax=8000)
mel_basis = jnp.asarray(mel_basis,dtype=jnp.float32)
spec = spec.transpose(0,2,1)
mel = jnp.matmul(mel_basis, spec)
mel = jnp.log(jnp.clip(mel, min=1e-5) * 1)
mel = mel.transpose(0,2,1)
f0 = model.apply({"params":params},mel,threshold=0.006,method=model.infer)
uv = (f0 < f0_min).astype(jnp.float32)
f0 = f0 * (1 - uv)
print(f0)

