#!/usr/bin/env python3
"""Shared helpers for ASR backends."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple


_WHISPERX_ALIGN_MODEL = None
_WHISPERX_ALIGN_META = None
_WHISPERX_ALIGN_LANGUAGE = None
_WHISPERX_PATCHED = False


def select_asr_device(env_var: str, default: str = "cpu") -> str:
    """Pick a device for ASR models, allowing env override."""
    override = os.environ.get(env_var)
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return default


def resolve_torch_dtype(value: Optional[str]):
    if not value or value == "auto":
        return "auto"
    try:
        import torch
        return getattr(torch, value)
    except Exception:
        return "auto"


def ensure_lzma() -> bool:
    try:
        import lzma  # noqa: F401
        return True
    except Exception:
        try:
            from backports import lzma as backports_lzma
            sys.modules["lzma"] = backports_lzma
            return True
        except Exception:
            return False


def ensure_qwen3_weights() -> Optional[Path]:
    """Ensure Qwen3-0.6B weights are available for Fun-ASR-Nano."""
    target_dir = (
        Path.home()
        / ".cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/Qwen3-0.6B"
    )
    weights_path = target_dir / "model.safetensors"
    if weights_path.exists():
        return weights_path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors", local_dir=str(target_dir))
    )


def patch_funasr_load_in_8bit() -> bool:
    try:
        import transformers
    except Exception:
        return False
    if getattr(transformers, "_funasr_load_in_8bit_patched", False):
        return True
    original = transformers.AutoModelForCausalLM.from_pretrained

    def _patched(*args, **kwargs):
        kwargs.pop("load_in_8bit", None)
        return original(*args, **kwargs)

    transformers.AutoModelForCausalLM.from_pretrained = _patched
    transformers._funasr_load_in_8bit_patched = True
    return True


def patch_transformers_video_processor() -> bool:
    """Guard transformers video processor lookup against None values."""
    try:
        from transformers.models.auto import video_processing_auto as vpa
    except Exception:
        return False
    if getattr(vpa, "_audio_cloze_none_guard", False):
        return True
    original = vpa.video_processor_class_from_name

    def _patched(class_name: Optional[str]):
        if class_name is None:
            return None
        return original(class_name)

    vpa.video_processor_class_from_name = _patched
    vpa._audio_cloze_none_guard = True
    return True


def _init_whisperx() -> None:
    global _WHISPERX_PATCHED
    ensure_lzma()
    import os as _os
    import numpy as _np
    import torch as _torch
    import typing as _typing
    import collections as _collections
    from omegaconf import listconfig, dictconfig, base, nodes as oc_nodes
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    _os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0")
    if not hasattr(_np, "NaN"):
        _np.NaN = _np.nan
    if _WHISPERX_PATCHED:
        return
    safe_nodes = [v for v in vars(oc_nodes).values() if isinstance(v, type)]
    try:
        _torch.serialization.add_safe_globals([
            listconfig.ListConfig,
            dictconfig.DictConfig,
            base.ContainerMetadata,
            _typing.Any,
            list,
            dict,
            tuple,
            set,
            _collections.defaultdict,
            int,
            float,
            str,
            bool,
            _torch.torch_version.TorchVersion,
        ] + safe_nodes)
    except Exception:
        pass
    orig_load = _torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    _torch.load = _patched_load

    orig_from_pretrained = Wav2Vec2ForCTC.from_pretrained

    def _safe_from_pretrained(*args, **kwargs):
        kwargs.setdefault("use_safetensors", True)
        return orig_from_pretrained(*args, **kwargs)

    Wav2Vec2ForCTC.from_pretrained = _safe_from_pretrained

    if not hasattr(Wav2Vec2Processor, "sampling_rate"):
        def _sampling_rate(self):
            fe = getattr(self, "feature_extractor", None)
            return getattr(fe, "sampling_rate", None)
        Wav2Vec2Processor.sampling_rate = property(_sampling_rate)
    _WHISPERX_PATCHED = True


def whisperx_align_tokens(audio_path: Path, transcript: str, language_code: str) -> list[dict]:
    """Return alignment tokens from WhisperX (chars when available)."""
    try:
        _init_whisperx()
        import whisperx
    except Exception as e:
        raise RuntimeError(f"missing deps: {e}")
    global _WHISPERX_ALIGN_MODEL, _WHISPERX_ALIGN_META, _WHISPERX_ALIGN_LANGUAGE
    device = "cpu"
    if (
        _WHISPERX_ALIGN_MODEL is None
        or _WHISPERX_ALIGN_META is None
        or _WHISPERX_ALIGN_LANGUAGE != language_code
    ):
        _WHISPERX_ALIGN_MODEL, _WHISPERX_ALIGN_META = whisperx.load_align_model(
            language_code=language_code,
            device=device,
        )
        _WHISPERX_ALIGN_LANGUAGE = language_code
    audio = whisperx.load_audio(str(audio_path))
    duration = audio.shape[0] / whisperx.audio.SAMPLE_RATE
    segments = [{"start": 0.0, "end": float(duration), "text": transcript}]
    result = whisperx.align(
        segments,
        _WHISPERX_ALIGN_MODEL,
        _WHISPERX_ALIGN_META,
        audio,
        device,
        return_char_alignments=True,
    )
    tokens: list[dict] = []
    for seg in result.get("segments", []):
        chars = seg.get("chars") or []
        if chars:
            for ch in chars:
                tokens.append({
                    "text": ch.get("char") or ch.get("text") or "",
                    "start": ch.get("start"),
                    "end": ch.get("end"),
                    "score": ch.get("score"),
                })
        else:
            for word in seg.get("words", []) or []:
                tokens.append({
                    "text": word.get("word", ""),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "score": word.get("score") or word.get("probability"),
                })
    return tokens


def load_glm_asr_transformers(
    model_id: str,
    device_env: str = "GLM_ASR_DEVICE",
    dtype_env: str = "GLM_ASR_DTYPE",
) -> Tuple[object, object]:
    ensure_lzma()
    from transformers import AutoModelForSeq2SeqLM, AutoProcessor

    dtype = resolve_torch_dtype(os.environ.get(dtype_env, "auto"))
    device_override = os.environ.get(device_env)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if device_override:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=None
        )
        model.to(device_override)
    else:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto"
            )
        except Exception:
            device = select_asr_device(device_env)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=None
            )
            model.to(device)
    return model, processor
