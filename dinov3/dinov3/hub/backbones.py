# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from typing import List, Optional, Union
from urllib.parse import urlparse
from pathlib import Path

from collections import OrderedDict
import torch
from safetensors.torch import load_file
from .utils import DINOV3_BASE_URL

def convert_hf_to_meta(model, hf_state_dict):
    """Convert a Hugging Face DINOv3 state dict to Meta's format."""
    model_state = model.state_dict()
    new_state_dict = OrderedDict()

    def get_required(key):
        if key not in hf_state_dict:
            raise KeyError(f"Hugging Face checkpoint is missing parameter: {key}")
        return hf_state_dict[key]

    def copy_parameter(src_key, dst_key):
        if dst_key not in model_state:
            return

        if src_key not in hf_state_dict:
            raise KeyError(
                f"Missing source parameter {src_key} while converting to {dst_key}"
            )

        new_state_dict[dst_key] = hf_state_dict[src_key]

    def match_tensor_shape(tensor, target_tensor, name):
        """Reconcile singleton token dimensions between implementations."""
        if tensor.shape == target_tensor.shape:
            return tensor

        # [1, 1, C] -> [1, C]
        if (
            tensor.ndim == 3
            and target_tensor.ndim == 2
            and tensor.shape[0] == 1
            and tensor.shape[1] == 1
        ):
            tensor = tensor.squeeze(1)

        # [1, C] -> [1, 1, C]
        elif (
            tensor.ndim == 2
            and target_tensor.ndim == 3
            and target_tensor.shape[0] == 1
            and target_tensor.shape[1] == 1
        ):
            tensor = tensor.unsqueeze(1)

        if tensor.shape != target_tensor.shape:
            raise RuntimeError(
                f"Cannot convert the shape of {name}:\n"
                f"HF shape: {tuple(tensor.shape)}\n"
                f"Meta shape: {tuple(target_tensor.shape)}"
            )

        return tensor

    # ================================================================

    # ================================================================

    if "cls_token" in model_state:
        cls_token = get_required("embeddings.cls_token")
        new_state_dict["cls_token"] = match_tensor_shape(
            cls_token,
            model_state["cls_token"],
            "cls_token",
        )

    if "mask_token" in model_state:
        mask_token = get_required("embeddings.mask_token")
        new_state_dict["mask_token"] = match_tensor_shape(
            mask_token,
            model_state["mask_token"],
            "mask_token",
        )

    if "storage_tokens" in model_state:
        register_tokens = get_required("embeddings.register_tokens")
        new_state_dict["storage_tokens"] = match_tensor_shape(
            register_tokens,
            model_state["storage_tokens"],
            "storage_tokens",
        )

    copy_parameter(
        "embeddings.patch_embeddings.weight",
        "patch_embed.proj.weight",
    )

    copy_parameter(
        "embeddings.patch_embeddings.bias",
        "patch_embed.proj.bias",
    )

    # ================================================================
    # 2. Transformer Blocks
    # ================================================================

    num_blocks = len(model.blocks)

    for i in range(num_blocks):
        src = f"layer.{i}"
        dst = f"blocks.{i}"

        # ------------------------------------------------------------
        # QKV Weight
        # HF: q_proj / k_proj / v_proj
        # Meta: qkv
        # ------------------------------------------------------------

        q_weight = get_required(
            f"{src}.attention.q_proj.weight"
        )
        k_weight = get_required(
            f"{src}.attention.k_proj.weight"
        )
        v_weight = get_required(
            f"{src}.attention.v_proj.weight"
        )

        qkv_weight_key = f"{dst}.attn.qkv.weight"

        if qkv_weight_key in model_state:
            new_state_dict[qkv_weight_key] = torch.cat(
                [q_weight, k_weight, v_weight],
                dim=0,
            )

        # ------------------------------------------------------------
        # QKV Bias
        #


        # ------------------------------------------------------------

        q_bias = hf_state_dict.get(
            f"{src}.attention.q_proj.bias"
        )
        k_bias = hf_state_dict.get(
            f"{src}.attention.k_proj.bias"
        )
        v_bias = hf_state_dict.get(
            f"{src}.attention.v_proj.bias"
        )

        qkv_bias_key = f"{dst}.attn.qkv.bias"

        if qkv_bias_key in model_state:
            available_biases = [
                bias
                for bias in (q_bias, k_bias, v_bias)
                if bias is not None
            ]

            if len(available_biases) == 0:
                new_state_dict[qkv_bias_key] = (
                    model_state[qkv_bias_key].clone()
                )
            else:
                reference_bias = available_biases[0]

                if q_bias is None:
                    q_bias = torch.zeros_like(reference_bias)

                if k_bias is None:
                    k_bias = torch.zeros_like(reference_bias)

                if v_bias is None:
                    v_bias = torch.zeros_like(reference_bias)

                new_state_dict[qkv_bias_key] = torch.cat(
                    [q_bias, k_bias, v_bias],
                    dim=0,
                )


        qkv_bias_mask_key = f"{dst}.attn.qkv.bias_mask"

        if qkv_bias_mask_key in model_state:
            new_state_dict[qkv_bias_mask_key] = (
                model_state[qkv_bias_mask_key].clone()
            )

        # ------------------------------------------------------------
        # Attention Output Projection
        # ------------------------------------------------------------

        copy_parameter(
            f"{src}.attention.o_proj.weight",
            f"{dst}.attn.proj.weight",
        )

        copy_parameter(
            f"{src}.attention.o_proj.bias",
            f"{dst}.attn.proj.bias",
        )

        # ------------------------------------------------------------
        # LayerNorm
        # ------------------------------------------------------------

        copy_parameter(
            f"{src}.norm1.weight",
            f"{dst}.norm1.weight",
        )

        copy_parameter(
            f"{src}.norm1.bias",
            f"{dst}.norm1.bias",
        )

        copy_parameter(
            f"{src}.norm2.weight",
            f"{dst}.norm2.weight",
        )

        copy_parameter(
            f"{src}.norm2.bias",
            f"{dst}.norm2.bias",
        )

        # ------------------------------------------------------------
        # LayerScale
        # ------------------------------------------------------------

        copy_parameter(
            f"{src}.layer_scale1.lambda1",
            f"{dst}.ls1.gamma",
        )

        copy_parameter(
            f"{src}.layer_scale2.lambda1",
            f"{dst}.ls2.gamma",
        )

        # ------------------------------------------------------------
        # MLP
        # ------------------------------------------------------------

        copy_parameter(
            f"{src}.mlp.up_proj.weight",
            f"{dst}.mlp.fc1.weight",
        )

        copy_parameter(
            f"{src}.mlp.up_proj.bias",
            f"{dst}.mlp.fc1.bias",
        )

        copy_parameter(
            f"{src}.mlp.down_proj.weight",
            f"{dst}.mlp.fc2.weight",
        )

        copy_parameter(
            f"{src}.mlp.down_proj.bias",
            f"{dst}.mlp.fc2.bias",
        )

    # ================================================================

    #

    # rope_embed.periods
    # qkv.bias_mask
    #

    # ================================================================

    for key, value in model_state.items():
        if key not in new_state_dict:
            new_state_dict[key] = value.clone()

    # ================================================================

    # ================================================================

    model_keys = set(model_state.keys())
    converted_keys = set(new_state_dict.keys())

    missing_keys = sorted(model_keys - converted_keys)
    unexpected_keys = sorted(converted_keys - model_keys)

    if missing_keys:
        raise RuntimeError(
            "Converted state dict is missing parameters:\n"
            + "\n".join(missing_keys)
        )

    if unexpected_keys:
        raise RuntimeError(
            "Converted state dict has unexpected parameters:\n"
            + "\n".join(unexpected_keys)
        )

    # ================================================================

    # ================================================================

    shape_mismatches = []

    for key in model_state:
        model_shape = tuple(model_state[key].shape)
        converted_shape = tuple(new_state_dict[key].shape)

        if model_shape != converted_shape:
            shape_mismatches.append(
                f"{key}: "
                f"model={model_shape}, "
                f"checkpoint={converted_shape}"
            )
    if shape_mismatches:
        raise RuntimeError(
            "The following parameters have mismatched shapes:\n"
            + "\n".join(shape_mismatches)
        )
    return new_state_dict


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_vit_model_arch(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
):
    if "plus" in compact_arch_name:
        model_arch = compact_arch_name.replace("plus", f"{patch_size}plus")
    else:
        model_arch = f"{compact_arch_name}{patch_size}"
    return model_arch


def _make_dinov3_vit_model_url(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    model_arch = _make_dinov3_vit_model_arch(patch_size=patch_size, compact_arch_name=compact_arch_name)
    version_suffix = f"_{version}" if version else ""
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"{model_name}_{model_arch}"
    model_filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: float | None = None,
    pos_embed_rope_max_period: float | None = None,
    pos_embed_rope_normalize_coords: str = "separate",
    pos_embed_rope_shift_coords: float | None = None,
    pos_embed_rope_jitter_coords: float | None = None,
    pos_embed_rope_rescale_coords: float | None = None,
    pos_embed_rope_dtype: str = "fp32",
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: float | None = None,
    norm_layer: str = "layernorm",
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    check_hash: bool = False,
    **kwargs,
):
    from ..models.vision_transformer import DinoVisionTransformer

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed_rope_base=pos_embed_rope_base,
        pos_embed_rope_min_period=pos_embed_rope_min_period,
        pos_embed_rope_max_period=pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=pos_embed_rope_dtype,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale_init,
        norm_layer=norm_layer,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        proj_bias=proj_bias,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
    )
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_vit_model_url(
                patch_size=patch_size,
                compact_arch_name=compact_arch_name,
                version=version,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)

        path = weights
        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            hf_state_dict = load_file(path, device="cpu")
            state_dict = convert_hf_to_meta(model, hf_state_dict)
        else:  # .pth, .pt, .bin
            state_dict = torch.load(path, map_location="cpu")

        model.load_state_dict(state_dict, strict=True)

    else:
        model.init_weights()
    return model


def _make_dinov3_convnext_model_url(
    *,
    compact_arch_name: str = "convnext_base",
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""

    model_dir = f"{model_name}_{compact_arch_name}"
    model_filename = f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_convnext(
    in_chans: int = 3,
    depths: List[int] = [3, 3, 27, 3],
    dims: List[int] = [128, 256, 512, 1024],
    compact_arch_name: str = "convnext_base",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    **kwargs,
):
    from ..models.convnext import ConvNeXt

    model_kwargs = dict(
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    model_kwargs.update(**kwargs)
    model = ConvNeXt(**model_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_convnext_model_url(
                compact_arch_name=compact_arch_name,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "4057cbaa"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitsplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitb",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) != 1:
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        hash = matches[0]
        if hash == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "46503df0"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitlplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "7c1da9a5"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vithplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a955f4ea"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a6675841"
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_convnext_tiny(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "21b726bb"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["tiny"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_tiny",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_small(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "296db49d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["small"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_small",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_base(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "801f2ba9"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["base"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_base",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_large(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "61fa432d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["large"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_large",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model
