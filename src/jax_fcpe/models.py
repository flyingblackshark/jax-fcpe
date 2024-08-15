import jax
import jax.numpy as jnp
import flax.linen as nn
from .model_conformer_naive import ConformerNaiveEncoder


class CFNaiveMelPE(nn.Module):
    """
    Conformer-based Mel-spectrogram Prediction Encoderc in Fast Context-based Pitch Estimation

    Args:
        input_channels (int): Number of input channels, should be same as the number of bins of mel-spectrogram.
        out_dims (int): Number of output dimensions, also class numbers.
        hidden_dims (int): Number of hidden dimensions.
        n_layers (int): Number of conformer layers.
        f0_max (float): Maximum frequency of f0.
        f0_min (float): Minimum frequency of f0.
        use_fa_norm (bool): Whether to use fast attention norm, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
        use_harmonic_emb (bool): Whether to use harmonic embedding, default False
        use_pre_norm (bool): Whether to use pre norm, default False
    """
    input_channels: int
    out_dims: int
    hidden_dims: int = 512
    n_layers: int = 6
    n_heads: int = 8
    f0_max: float = 1975.5
    f0_min: float = 32.70
    use_fa_norm: bool = False
    conv_only: bool = False
    conv_dropout: float = 0.
    atten_dropout: float = 0.
    use_harmonic_emb: bool = False
    @nn.compact
    def __call__(self, x: jnp.ndarray, _h_emb=None,deterministic=None) -> jnp.ndarray:
        """
        Args:
            x (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
            _h_emb (int): Harmonic embedding index, like 0, 1, 2， only use in train. Default: None.
        return:
            torch.Tensor: Predicted f0 latent, shape (B, T, out_dims).
        """
        #x = jnp.swapaxes(x,-1,-2)
        x = nn.Conv(self.hidden_dims, 3, 1 ,name="input_stack_conv1")(x)
        x = nn.GroupNorm(num_groups=4,name="input_stack_norms")(x)
        x = nn.leaky_relu(x)
        x = nn.Conv(self.hidden_dims, 3, 1 ,name="input_stack_conv2")(x)
        #x = jnp.swapaxes(x,-1,-2)
        # if self.harmonic_emb is not None:
        #     if _h_emb is None:
        #         x = x + self.harmonic_emb(torch.LongTensor([0]).to(x.device))
        #     else:
        #         x = x + self.harmonic_emb(torch.LongTensor([int(_h_emb)]).to(x.device))
        x = ConformerNaiveEncoder(
            num_layers=self.n_layers,
            num_heads=self.n_heads,
            dim_model=self.hidden_dims,
            use_norm=self.use_fa_norm,
            conv_only=self.conv_only,
            conv_dropout=self.conv_dropout,
            atten_dropout=self.atten_dropout,
            name="net"
        )(x,deterministic=deterministic)
        x = nn.LayerNorm(name="norm")(x)
        x = FlaxDenseWithWeightNorm(self.hidden_dims,self.out_dims,name="output_proj")(x)
        x = nn.sigmoid(x)
        return x  # latent (B, T, out_dims)

    #@torch.no_grad()
    def latent2cents_decoder(self,
                             y: jnp.ndarray,
                             threshold: float = 0.05,
                             mask: bool = True
                             ) -> jnp.ndarray:
        """
        Convert latent to cents.
        Args:
            y (torch.Tensor): Latent, shape (B, T, out_dims).
            threshold (float): Threshold to mask. Default: 0.05.
            mask (bool): Whether to mask. Default: True.
        return:
            torch.Tensor: Cents, shape (B, T, 1).
        """
        cent_table = jnp.linspace(self.f0_to_cent(jnp.asarray([self.f0_min]))[0],
                                           self.f0_to_cent(jnp.asarray([self.f0_max]))[0],
                                           self.out_dims)
        B, N, _ = y.size()
        ci = cent_table[None, None, :].expand(B, N, -1)
        rtn = jnp.sum(ci * y, axis=-1, keepdim=True) / jnp.sum(y, axis=-1, keepdim=True)  # cents: [B,N,1]
        if mask:
            confident = jnp.max(y, dim=-1, keepdim=True)[0]
            confident_mask = jnp.ones_like(confident)
            confident_mask[confident <= threshold] = float("-INF")
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    #@torch.no_grad()
    def latent2cents_local_decoder(self,
                                   y: jnp.ndarray,
                                   threshold: float = 0.05,
                                   mask: bool = True
                                   ) -> jnp.ndarray:
        """
        Convert latent to cents. Use local argmax.
        Args:
            y (torch.Tensor): Latent, shape (B, T, out_dims).
            threshold (float): Threshold to mask. Default: 0.05.
            mask (bool): Whether to mask. Default: True.
        return:
            torch.Tensor: Cents, shape (B, T, 1).
        """
        cent_table = jnp.linspace(self.f0_to_cent(jnp.asarray([self.f0_min]))[0],
                                           self.f0_to_cent(jnp.asarray([self.f0_max]))[0],
                                           self.out_dims)
        
        B, N, _ = y.shape
        ci = jnp.broadcast_to(cent_table[None, None, :],(B, N,cent_table.shape[0]))
        confident  = jnp.max(y, axis=-1, keepdims=True)
        max_index = jnp.argmax(y ,axis=-1, keepdims=True)
        local_argmax_index = jnp.arange(0, 9) + (max_index - 4)
        local_argmax_index = jnp.where(local_argmax_index < 0,0,local_argmax_index)
        local_argmax_index = jnp.where(local_argmax_index >= self.out_dims , self.out_dims - 1,local_argmax_index)
        ci_l = jnp.take_along_axis(ci, axis=-1, indices=local_argmax_index)
        y_l = jnp.take_along_axis(y, axis=-1, indices=local_argmax_index)
        rtn = jnp.sum(ci_l * y_l, axis=-1, keepdims=True) / jnp.sum(y_l, axis=-1, keepdims=True)  # cents: [B,N,1]
        if mask:
            confident_mask = jnp.ones_like(confident)
            confident_mask = jnp.where(confident <= threshold,float("-INF"),confident_mask)
            rtn = rtn * confident_mask
        return rtn  # (B, T, 1)

    #@torch.no_grad()
    def gaussian_blurred_cent2latent(self, cents):  # cents: [B,N,1]
        """
        Convert cents to latent.
        Args:
            cents (torch.Tensor): Cents, shape (B, T, 1).
        return:
            torch.Tensor: Latent, shape (B, T, out_dims).
        """
        cent_table = jnp.linspace(self.f0_to_cent(jnp.asarray([self.f0_min]))[0],
                                           self.f0_to_cent(jnp.asarray([self.f0_max]))[0],
                                           self.out_dims)
        gaussian_blurred_cent_mask = (1200. * jnp.log2(jnp.asarray([self.f0_max / 10.])))[0]
        mask = (cents > 0.1) & (cents < gaussian_blurred_cent_mask)
        # mask = (cents>0.1) & (cents<(1200.*np.log2(self.f0_max/10.)))
        B, N, _ = cents.shape
        ci = jnp.broadcast_to(cent_table[None, None, :], (B, N, -1))
        return jnp.exp(-jnp.square(ci - cents) / 1250) * mask.astype(jnp.float32)

    def infer(self,
              mel: jnp.ndarray,
              decoder: str = "local_argmax",  # "argmax" or "local_argmax"
              threshold: float = 0.05,
              ) -> jnp.ndarray:
        """
        Args:
            mel (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
            decoder (str): Decoder type. Default: "local_argmax".
            threshold (float): Threshold to mask. Default: 0.05.
        """
        latent = self.__call__(mel,deterministic=True)
        if decoder == "argmax":
            cents = self.latent2cents_decoder(latent, threshold=threshold)
        elif decoder == "local_argmax":
            cents = self.latent2cents_local_decoder(latent, threshold=threshold)
        else:
            raise ValueError(f"  [x] Unknown decoder type {decoder}.")
        f0 = self.cent_to_f0(cents)
        return f0  # (B, T, 1)

    # def train_and_loss(self, mel, gt_f0, loss_scale=10):
    #     """
    #     Args:
    #         mel (torch.Tensor): Input mel-spectrogram, shape (B, T, input_channels) or (B, T, mel_bins).
    #         gt_f0 (torch.Tensor): Ground truth f0, shape (B, T, 1).
    #         loss_scale (float): Loss scale. Default: 10.
    #     return: loss
    #     """
    #     if mel.shape[-2] != gt_f0.shape[-2]:
    #         _len = min(mel.shape[-2], gt_f0.shape[-2])
    #         mel = mel[:, :_len, :]
    #         gt_f0 = gt_f0[:, :_len, :]
    #     gt_cent_f0 = self.f0_to_cent(gt_f0)  # mel f0, [B,N,1]
    #     x_gt = self.gaussian_blurred_cent2latent(gt_cent_f0)  # [B,N,out_dim]
    #     if self.harmonic_emb is not None:
    #         x = self.forward(mel, _h_emb=0)
    #         x_half = self.forward(mel, _h_emb=1)
    #         x_gt_half = self.gaussian_blurred_cent2latent(gt_cent_f0 / 2)
    #         x_gt_double = self.gaussian_blurred_cent2latent(gt_cent_f0 * 2)
    #         x_double = self.forward(mel, _h_emb=2)
    #         loss = F.binary_cross_entropy(x, x_gt)
    #         loss_half = F.binary_cross_entropy(x_half, x_gt_half)
    #         loss_double = F.binary_cross_entropy(x_double, x_gt_double)
    #         loss = loss + (loss_half + loss_double) / 2
    #         loss = loss * loss_scale
    #     else:
    #         x = self.forward(mel)  # [B,N,out_dim]
    #         loss = F.binary_cross_entropy(x, x_gt) * loss_scale
    #     return loss

    #@torch.no_grad()
    def cent_to_f0(self, cent: jnp.ndarray) -> jnp.ndarray:
        """
        Convert cent to f0. Args: cent (torch.Tensor): Cent, shape = (B, T, 1). return: torch.Tensor: f0, shape = (B, T, 1).
        """
        f0 = 10. * 2 ** (cent / 1200.)
        return f0  # (B, T, 1)

    #@torch.no_grad()
    def f0_to_cent(self, f0: jnp.ndarray) -> jnp.ndarray:
        """
        Convert f0 to cent. Args: f0 (torch.Tensor): f0, shape = (B, T, 1). return: torch.Tensor: Cent, shape = (B, T, 1).
        """
        cent = 1200. * jnp.log2(f0 / 10.)
        return cent  # (B, T, 1)
class FlaxDenseWithWeightNorm(nn.Module):
    features_in : int
    features_out : int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            features=self.features_out,
            kernel_init=jax.nn.initializers.he_normal(),
            dtype=self.dtype,
        )
        weight_shape = (
            self.features_out,
            self.features_in
        )
        self.weight_v = self.param(
            "weight_v", jax.nn.initializers.he_normal(), weight_shape
        )
        self.weight_g = self.param(
            "weight_g",
            lambda _: jnp.linalg.norm(self.weight_v, axis=1)[ :,None],
        )
        self.bias = self.param("bias", jax.nn.initializers.zeros, (self.dense.features,))

    def _get_normed_weights(self):
        weight_v_norm = jnp.linalg.norm(self.weight_v, axis=1)[:,None]
        normed_weight_v = jnp.divide(self.weight_v, weight_v_norm)
        normed_kernel = jnp.multiply(normed_weight_v, self.weight_g)
        return normed_kernel

    def __call__(self, hidden_states):
        kernel = self._get_normed_weights()
        # hidden_states = jnp.pad(
        #     hidden_states, ((0, 0), (self.prev_padding, self.prev_padding), (0, 0))
        # )
        hidden_states = self.dense.apply(
            {"params": {"kernel": kernel.T, "bias": self.bias}}, hidden_states
        )
        return hidden_states
    
if __name__ =="__main__":
    model = CFNaiveMelPE(128,360,512,6,6,1975.5,32.7,True,0.1,0.1)
    #params = model.init(jax.random.PRNGKey(0),jnp.ones((1,50,128)),deterministic=True)
    #flatten_param = traverse_util.flatten_dict(params,sep='.')
    from convert import load_params
    params = load_params()
    output2 = model.apply({"params":params},jnp.ones((1,50,128)),method=model.infer)
