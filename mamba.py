"""Simple, minimal implementation of Mamba in one file of Numpy adapted from (1) and inspired from (2).

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

References:
    (1) mamba-minimal (John Ma)
        https://github.com/johnma2006/mamba-minimal/
    (2) llama3.np (Sang Park)
        https://github.com/likejazz/llama3.np
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

import numpy as np
from einops import einsum, rearrange
from transformers import AutoTokenizer

_MAX_NEW_TOKENS = 18


def load_model(pretrained_model_name: str) -> Mamba:
    """Load pretrained weights from HuggingFace into model.

    Args:
        pretrained_model_name: One of
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'

    Returns:
        model: Mamba model with weights loaded

    """
    import torch
    from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(
            model_name,
            CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        return json.load(open(resolved_archive_file))

    def load_state_dict_hf(model_name):
        resolved_archive_file = cached_file(
            model_name,
            WEIGHTS_NAME,
            _raise_exceptions_for_missing_entries=False,
        )
        return torch.load(
            resolved_archive_file,
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )

    config_data = load_config_hf(pretrained_model_name)
    args = ModelArgs(
        d_model=config_data["d_model"],
        n_layer=config_data["n_layer"],
        vocab_size=config_data["vocab_size"],
    )
    state_dict = load_state_dict_hf(pretrained_model_name)

    weights = {}
    for key in state_dict:
        new_key = key.replace("backbone.", "")
        weights[new_key] = state_dict[key].numpy()

    model = Mamba(weights, args)

    return model


@dataclass
class ModelArgs:
    """
    Data class for storing model-specific arguments.

    Args:
        d_model (int): Model dimension.
        n_layer (int): Number of layers.
        vocab_size (int): Vocabulary size.
        d_state (int, optional): State dimension (default: 16).
        expand (int, optional): Expansion factor (default: 2).
        dt_rank (Union[int, str], optional): Rank for Δ (default: "auto").
        d_conv (int, optional): Convolution dimension (default: 4).
        pad_vocab_size_multiple (int, optional): Padding vocabulary size multiple (default: 8).
        conv_bias (bool, optional): Whether to use bias in convolution layers (default: True).
        bias (bool, optional): Whether to use bias in linear layers (default: False).

    Attributes:
        d_inner (int): Inner dimension calculated as expand * d_model.

    Notes:
        - If dt_rank is set to "auto", it computes it as the ceiling of d_model / 16.
        - Ensures that vocab_size is a multiple of pad_vocab_size_multiple.
    """

    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba:
    def __init__(self, weights: Mapping[str, np.ndarray], args: ModelArgs):
        """
        Full Mamba model.

        Args:
            weights (Mapping[str, np.ndarray]): Pre-trained weights.
            args (ModelArgs): Model-specific arguments.
        """
        self.args = args
        self.embedding = Embedding(weight=weights.get("embedding.weight"))
        self.layers = [
            ResidualBlock(i, weights, args) for i in range(args.n_layer)
        ]
        self.norm_f = RMSNorm(weight=weights.get("norm_f.weight"))

        # Tie output projection to embedding weights.
        # See "Weight Tying" paper
        self.lm_head = Linear(weight=self.embedding.weight, bias=None)

    def __call__(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Mamba model.

        Args:
            input_ids (np.ndarray): shape (b, l), dtype long.

        Returns:
            np.ndarray: shape (b, l, vocab_size). The output logits tensor.

        Official Implementation:
            class MambaLMHeadModel, see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L118
        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    def generate(self, input_ids: np.ndarray, max_new_tokens: int):
        _, L = input_ids.shape
        for _ in range(L, max_new_tokens):
            logits = self(input_ids)[:, -1]
            next_id = np.argmax(logits, axis=-1, keepdims=True)
            input_ids = np.concatenate([input_ids, next_id], axis=-1)
            yield next_id


class ResidualBlock:
    def __init__(
        self, layer_id: int, weights: Mapping[str, np.ndarray], args: ModelArgs
    ):
        """
        Residual block for Mamba-based models.

        Args:
            layer_id (int): Identifier for the layer.
            weights (Mapping[str, np.ndarray]): Pre-trained weights.
            args (ModelArgs): Model-specific arguments.
        """
        self.args = args
        self.mixer = MambaBlock(
            in_proj=Linear(
                weight=weights.get(f"layers.{layer_id}.mixer.in_proj.weight"),
                bias=None,
            ),
            conv1d=MambaConv1d(
                weight=weights.get(f"layers.{layer_id}.mixer.conv1d.weight"),
                bias=weights.get(f"layers.{layer_id}.mixer.conv1d.bias"),
            ),
            x_proj=Linear(
                weight=weights.get(f"layers.{layer_id}.mixer.x_proj.weight"),
                bias=None,
            ),
            dt_proj=Linear(
                weight=weights.get(f"layers.{layer_id}.mixer.dt_proj.weight"),
                bias=weights.get(f"layers.{layer_id}.mixer.dt_proj.bias"),
            ),
            A_log=weights.get(f"layers.{layer_id}.mixer.A_log"),
            D=weights.get(f"layers.{layer_id}.mixer.D"),
            out_proj=Linear(
                weight=weights.get(f"layers.{layer_id}.mixer.out_proj.weight"),
                bias=None,
            ),
            args=args,
        )

        self.norm = RMSNorm(
            weight=weights.get(f"layers.{layer_id}.norm.weight")
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the residual block.

        Args:
            x (np.ndarray): shape (b, l, d).

        Returns:
            np.ndarray: shape (b, l, d).

        Official Implementation:
            Block.forward(), see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L142

            Note: The official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ...

        """
        output = self.mixer(self.norm(x)) + x
        return output


class MambaBlock:
    def __init__(
        self,
        in_proj: Linear,
        conv1d: MambaConv1d,
        x_proj: Linear,
        dt_proj: Linear,
        A_log: np.ndarray,
        D: np.ndarray,
        out_proj: Linear,
        args: ModelArgs,
    ):
        """
        A single Mamba block, as described in Figure 3 in Section 3.4 of the Mamba paper [1].

        Args:
            in_proj (Linear): shape (d, 2*d_in). Linear layer for input projection.
            conv1d (MambaConv1d): shape (d_in, 1, d_conv). Mamba-specific 1D convolutional layer.
            x_proj (Linear): shape (d_in, dt_rank+2*d_state). Linear layer for projecting input-specific Δ, B, and C.
            dt_proj (Linear): shape (dt_rank, d_in). Linear layer for projecting Δ dt_rank to d_in.
            A_log (np.ndarray): shape (d_in, d). Matrix A_log.
            D (np.ndarray): shape (d_in,). Vector D.
            out_proj (Linear): shape (d_in, d). Linear layer for output projection.
            args (ModelArgs): Model-specific arguments.
        """
        self.args = args
        self.in_proj: Linear = in_proj
        self.conv1d: MambaConv1d = conv1d
        self.x_proj: Linear = x_proj
        self.dt_proj: Linear = dt_proj
        self.A_log: np.ndarray = A_log
        self.D: np.ndarray = D
        self.out_proj: Linear = out_proj

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Mamba block.

        Args:
            x (np.ndarray): Input tensor of shape (b, l, d).

        Returns:
            np.ndarray: Output tensor of shape (b, l, d).
        """
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = np.split(
            x_and_res,
            indices_or_sections=(self.args.d_inner, 2 * self.args.d_inner),
            axis=-1,
        )[:-1]

        x = self.conv1d(x)
        x = silu(x)

        y = self.ssm(x)
        y = y * silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x: np.ndarray) -> np.ndarray:
        """
        Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1] [1].
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x (np.ndarray): shape (b, l, d_in).

        Returns:
            np.ndarray: shape (b, l, d_in).

        Official Implementation:
            mamba_inner_ref(), see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        References:
            [1] Mamba paper: https://arxiv.org/abs/2106.16067
            [2] The Annotated S4: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆, A, B, C, D (state space parameters)
        # A and D are input-independent (see Mamba paper [1], Section 3.5.2 for A's interpretation)
        # ∆, B, C are input-dependent (a key difference between Mamba and linear time-invariant S4)

        A = -np.exp(self.A_log.astype(float))  # shape (d_in, n)
        D = self.D.astype(float)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = np.split(
            x_dbl,
            indices_or_sections=(
                self.args.dt_rank,
                self.args.dt_rank + n,
                self.args.dt_rank + 2 * n,
            ),
            axis=-1,
        )[
            :-1
        ]  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # Similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(
        self,
        u: np.ndarray,
        delta: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """
        Performs the selective scan algorithm as described in the Mamba paper [1].
        This function computes the output based on input data and state space parameters.
        See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u (np.ndarray): shape (b, l, d_in). Input tensor.
            delta (np.ndarray): shape (b, l, d_in). Step size tensor.
            A (np.ndarray): shape (d_in, n). Matrix A.
            B (np.ndarray): shape (b, l, n). Tensor B.
            C (np.ndarray): shape (b, l, n). Tensor C.
            D (np.ndarray): shape (d_in,). Vector D.

        Returns:
            np.ndarray: Output tensor of shape (b, l, d_in).

        Official Implementation:
            selective_scan_ref(), see https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: Some parts have been refactored from `selective_scan_ref`, so the functionality may not match exactly.

        References:
            [1] Mamba paper: https://arxiv.org/abs/2106.16067
            [2] The Annotated S4: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        deltaA = np.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        deltaB_u = einsum(
            delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n"
        )

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that is additionally hardware-aware (like FlashAttention).
        x = np.zeros((b, d_in, n))
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)

        y = np.stack(ys, axis=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class MambaConv1d:
    """
    A 1 dimensional convolution layer with pre-defined weights and optional bias that is applied on each channel separately.

    Args:
        weight (np.ndarray): The weight tensor for the convolution layer.
        bias (np.ndarray or None): The bias tensor (optional). If None, no bias is applied.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies 1D convolution to the input tensor `x`.

        Args:
            x (np.ndarray): Input tensor with shape (batch_size, sequence_length, in_channels).

        Returns:
            np.ndarray: Output tensor after 1D convolution.
        """
        x = rearrange(x, "b l d_in -> b d_in l")

        batch_size, in_channels, length = x.shape
        out_channels, _, kernel_size = self.weight.shape

        assert in_channels == out_channels

        output_length = length + kernel_size - 1
        output_tensor = np.empty((batch_size, in_channels, output_length))

        for b in range(batch_size):
            for i in range(in_channels):
                output_tensor[b, i, :] = np.convolve(
                    x[b, i, :], self.weight[i, 0, ::-1], mode="full"
                )
                if self.bias is not None:
                    output_tensor[b, i, :] += self.bias[i]

        output_tensor = output_tensor[:, :, :length]
        output_tensor = rearrange(output_tensor, "b d_in l -> b l d_in")

        return output_tensor


class Linear:
    """
    Represents a linear transformation layer.

    Args:
        weight (np.ndarray): The weight matrix for the linear transformation.
        bias (np.ndarray or None): The bias vector (optional). If None, no bias is applied.
    """

    def __init__(self, weight: np.ndarray, bias: np.ndarray):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the linear transformation to the input tensor `x`.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output tensor after linear transformation.
        """
        output_tensor = x @ self.weight.T
        if self.bias is not None:
            output_tensor += self.bias

        return output_tensor


class Embedding:
    """
    Represents an embedding layer with pre-defined weights.

    Args:
        weight (np.ndarray): The weight matrix for the embedding layer.
    """

    def __init__(self, weight: np.ndarray):
        self.weight = weight

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the embedding vectors for the given indices `x`.

        Args:
            x (np.ndarray): Indices of the desired embeddings, shape: (b, l).

        Returns:
            np.ndarray: The embedding vectors corresponding to the input indices.
        """
        return self.weight[x]


class RMSNorm:
    def __init__(self, weight: np.ndarray, eps: float = 1e-5):
        """
        Initializes an instance of the RMSNorm class.

        Args:
            weight (np.ndarray): Weight vector for normalization.
            eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-5.
        """
        self.weight = weight
        self.eps = eps

    def __call__(self, x: np.ndarray):
        """
        Applies RMS normalization to the input tensor.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Normalized tensor.
        """
        # Compute the root mean square along the last dimension
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * self.weight / rms


def silu(x: np.ndarray) -> np.ndarray:
    """
    Applies the Sigmoid Linear Unit (SiLU) activation function to the input tensor.

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Output tensor after applying SiLU.
    """
    return x / (1 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Applies the Softplus activation function to the input tensor.

    Args:
        x (np.ndarray): Input tensor.

    Returns:
        np.ndarray: Output tensor after applying Softplus.
    """
    return np.log(1 + np.exp(x))


def generate_text(model: Mamba, tokenizer: AutoTokenizer, prompt: str) -> None:
    """
    Generates text using a pre-trained language model.

    Args:
        model (Mamba): The pre-trained language model.
        tokenizer (AutoTokenizer): The tokenizer for encoding input prompts.
        prompt (str): Input prompt for text generation.
    """

    # Print the input prompt
    print(f"\n{prompt}", end="")

    # Encode the input prompt and initialize token count
    input_ids = np.array([tokenizer.encode(prompt)])
    _, L = input_ids.shape

    # Generate text
    start = time.time()
    for id in model.generate(input_ids, max_new_tokens=_MAX_NEW_TOKENS):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [tokenizer.eos_token_id, tokenizer.bos_token_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()

    # Calculate elapsed time and tokens per second
    elapsed = time.time() - start
    print(
        f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s"
    )


if __name__ == "__main__":
    # Read input prompt
    prompt = "I have a dream that" if len(sys.argv) == 1 else sys.argv[1]

    # Load the pre-trained language model and tokenizer
    model = load_model("state-spaces/mamba-130m")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    generate_text(model=model, tokenizer=tokenizer, prompt=prompt)
