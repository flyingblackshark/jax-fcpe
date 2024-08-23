
# FCPE jax version 
## This version is working perfectly fine. 😀 
### Original https://github.com/CNChTu/FCPE

# Example
```Python
import jax_fcpe
import jax.numpy as jnp

a = jnp.ones((16000))
f0 = jax_fcpe.get_f0(a,16000)
print(f0)
```
## Advanced Usage
```Python
WIN_SIZE = 1024
HOP_SIZE = 160
N_FFT = 1024
NUM_MELS = 128
f0_min = 80.
f0_max = 880.
mel_basis = librosa_mel_fn(sr=16000, n_fft=N_FFT, n_mels=NUM_MELS, fmin=0, fmax=8000)
mel_basis = jnp.asarray(mel_basis,dtype=jnp.float32)

def get_f0(wav,model,params):
    wav = jnp.asarray(wav)
    window = jnp.hanning(WIN_SIZE)
    pad_size = (WIN_SIZE-HOP_SIZE)//2
    wav = jnp.pad(wav, ((0,0),(pad_size, pad_size)),mode="reflect")
    spec = audax.core.stft.stft(wav,N_FFT,HOP_SIZE,WIN_SIZE,window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))
    spec = spec.transpose(0,2,1)
    mel = jnp.matmul(mel_basis, spec)
    mel = jnp.log(jnp.clip(mel, min=1e-5) * 1)
    mel = mel.transpose(0,2,1)

    def model_predict(mel):
        f0 = model.apply(params,mel,threshold=0.006,method=model.infer)
        uv = (f0 < f0_min).astype(jnp.float32)
        f0 = f0 * (1 - uv)
        return f0
    return model_predict(mel).squeeze(-1)
```
