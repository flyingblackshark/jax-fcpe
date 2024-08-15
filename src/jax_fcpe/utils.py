from os import environ
from pathlib import Path
import os
from .models import CFNaiveMelPE
from .convert import convert_torch_weights
import flax
def convert_torch_weights_to_msgpack(torch_weights_path: Path, write_path: Path):
    if write_path.exists():
        return

    if not write_path.exists():
        write_path.parent.mkdir(parents=True, exist_ok=True)


    weights = convert_torch_weights(torch_weights_path)
    weights = flax.serialization.msgpack_serialize(weights)
    with open(write_path, "wb") as msgpack_file:
        msgpack_file.write(weights)

def download_model():
    if 'JAX_FCPE_CACHE' in environ and environ['JAX_FCPE_CACHE'].strip() and os.path.isabs(environ['JAX_FCPE_CACHE']):
        cache_home = environ['JAX_FCPE_CACHE']
        cache_home = Path(cache_home)
    else:
        cache_home = Path.home() / ".cache" / "jax_fcpe"

    # metadata_path = (
    #     cache_home
    #     / f"weights_{model_type}_{model_bitrate}_{tag}.json"
    # )
    jax_write_path = (
        cache_home
        / "fcpe_c_v001.npy"
    )

    if jax_write_path.exists():
        return jax_write_path

    download_link = "https://raw.githubusercontent.com/CNChTu/FCPE/main/torchfcpe/assets/fcpe_c_v001.pt"

    torch_model_path = (
        cache_home
        / "fcpe_c_v001.pt"
    )

    if not torch_model_path.exists():
        torch_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the model
        import requests

        response = requests.get(download_link)

        if response.status_code != 200:
            raise ValueError(
                f"Could not download model. Received response code {response.status_code}"
            )
        torch_model_path.write_bytes(response.content)

    convert_torch_weights_to_msgpack(torch_model_path, jax_write_path)

    # remove torch model because it's not needed anymore.
    if torch_model_path.exists():
        os.remove(torch_model_path)

    return jax_write_path

def load_model(load_path: str = None,):
    if not load_path:
        load_path = download_model()
    with open(load_path, "rb") as msgpack_file:
        msgpack_content = msgpack_file.read()
    params =  flax.serialization.msgpack_restore(msgpack_content)
    params = {"params":params}
    model = CFNaiveMelPE(128,360,512,6,8,1975.5,32.7,True,0.1,0.1)
    return model,params
    