import audax.core
import audax.core.functional
import jax.numpy as jnp
import torch
import audax
WIN_SIZE = 1024
HOP_SIZE = 160
N_FFT = 1024
NUM_MELS = 128
a = torch.randn(1,44100)
hann_window_tensor = torch.hann_window(WIN_SIZE)
b = torch.stft(a,N_FFT,HOP_SIZE,WIN_SIZE,hann_window_tensor,center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
b = torch.sqrt(b.real**2+b.imag**2+(1e-9))
a2 = jnp.asarray(a.numpy())

hann_window_array = jnp.hanning(WIN_SIZE)
b2 = audax.core.functional.stft(a2,N_FFT,HOP_SIZE,WIN_SIZE,hann_window_array,center=False,onesided=True)
b2 = jnp.sqrt(b2.real**2+b2.imag**2+(1e-9))
b2 = b2.transpose(0,2,1)

print((b.numpy() - b2).sum())