#!/usr/bin/env python3
"""Shared helpers for ASR backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple


_WHISPERX_ALIGN_MODEL = None
_WHISPERX_ALIGN_META = None
_WHISPERX_ALIGN_LANGUAGE = None
_WHISPERX_PATCHED = False
_QWEN3_OMNI_MLX_MODEL = None
_QWEN3_OMNI_MLX_PROCESSOR = None
_QWEN3_OMNI_MLX_MODEL_ID = None
_QWEN3_OMNI_MLX_UNSUPPORTED = False
_FUNASR_MODEL = None
_GLM_ASR_MODEL = None
_GLM_ASR_PROCESSOR = None

QWEN3_OMNI_PROMPT_ZH = "请逐字转写音频内容，英文单词保持英文拼写，不要音译或翻译。只输出转写文本。"
QWEN3_OMNI_PROMPT_JA = "日本語の音声を書き起こしてください。英単語は英語のまま、翻訳しないでください。出力は書き起こしのみ。"


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


def require_lzma() -> None:
    try:
        import lzma  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Python was built without lzma support. Reinstall Python with liblzma "
            "available (macOS/pyenv: `brew install xz` then rebuild Python)."
        ) from e


require_lzma()


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


def mlx_whisper_transcribe(
    audio_path: Path,
    language: str,
    word_timestamps: bool = False,
) -> str:
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError(f"mlx-whisper requires: {e}")
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language=language,
        word_timestamps=word_timestamps,
    )
    return result.get("text", "").strip()


def qwen3_omni_prompt(language: str) -> str:
    override = os.environ.get("QWEN3_OMNI_PROMPT")
    if override:
        return override
    if language == "zh":
        return QWEN3_OMNI_PROMPT_ZH
    if language == "ja":
        return QWEN3_OMNI_PROMPT_JA
    return "Please transcribe the audio into text."


def qwen3_omni_max_new_tokens(default: int = 1024) -> int:
    return int(os.environ.get("QWEN3_OMNI_MAX_NEW_TOKENS", str(default)))


def _load_qwen3_omni_mlx(model_id: str):
    global _QWEN3_OMNI_MLX_MODEL, _QWEN3_OMNI_MLX_PROCESSOR, _QWEN3_OMNI_MLX_MODEL_ID
    if (
        _QWEN3_OMNI_MLX_MODEL is None
        or _QWEN3_OMNI_MLX_PROCESSOR is None
        or _QWEN3_OMNI_MLX_MODEL_ID != model_id
    ):
        patch_transformers_video_processor()
        from mlx_vlm import load as mlx_load
        model, processor = mlx_load(model_id)
        if hasattr(model, "config") and hasattr(processor, "tokenizer"):
            if not hasattr(model.config, "eos_token_id"):
                model.config.eos_token_id = processor.tokenizer.eos_token_id
        _QWEN3_OMNI_MLX_MODEL = model
        _QWEN3_OMNI_MLX_PROCESSOR = processor
        _QWEN3_OMNI_MLX_MODEL_ID = model_id
    return _QWEN3_OMNI_MLX_MODEL, _QWEN3_OMNI_MLX_PROCESSOR


def qwen3_omni_mlx_transcribe(
    audio_path: Path,
    *,
    model_id: str,
    language: str,
    prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
) -> str:
    global _QWEN3_OMNI_MLX_UNSUPPORTED
    if os.environ.get("QWEN3_OMNI_DISABLE_MLX"):
        raise RuntimeError("Qwen3-Omni MLX backend disabled via QWEN3_OMNI_DISABLE_MLX")
    if _QWEN3_OMNI_MLX_UNSUPPORTED:
        raise RuntimeError("Qwen3-Omni MLX backend unavailable (previous failure)")

    model, processor = _load_qwen3_omni_mlx(model_id)
    import librosa
    import numpy as np
    import mlx.core as mx

    prompt_text = prompt or qwen3_omni_prompt(language)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "placeholder"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    formatted = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    audio, _ = librosa.load(str(audio_path), sr=sr)
    processed = processor(
        text=formatted,
        audio=[audio],
        padding=True,
        return_attention_mask=True,
        return_tensors=None,
    )

    input_ids = mx.array(processed["input_ids"])
    feature_attention_mask = np.array(processed["feature_attention_mask"], dtype=np.int32)
    audio_feature_lengths = np.sum(feature_attention_mask, axis=-1, dtype=np.int32)
    input_features = np.array(processed["input_features"])

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    thinker_eos = getattr(model.config, "eos_token_id", None)
    if thinker_eos is None:
        thinker_eos = getattr(tokenizer, "eos_token_id", None)
    try:
        thinker_result, _ = model.generate(
            input_ids,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens or qwen3_omni_max_new_tokens(),
            thinker_temperature=0.0,
            thinker_top_p=1.0,
            thinker_eos_token_id=thinker_eos,
            input_features=mx.array(input_features),
            feature_attention_mask=mx.array(feature_attention_mask),
            audio_feature_lengths=mx.array(audio_feature_lengths, dtype=mx.int32),
        )
    except Exception as e:
        _QWEN3_OMNI_MLX_UNSUPPORTED = True
        raise RuntimeError(
            "Qwen3-Omni MLX failed. Ensure mlx-vlm >= 0.3.10 (GitHub) and "
            f"audio decoding support are available. Error: {e}"
        ) from e
    sequences = thinker_result.sequences
    prompt_len = input_ids.shape[1]
    decoded = tokenizer.batch_decode(
        sequences[:, prompt_len:].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def funasr_transcribe(
    audio_path: Path,
    *,
    model_id: str,
    language: str,
    device_env: str = "FUNASR_DEVICE",
) -> str:
    try:
        from funasr import AutoModel
        from funasr.models.fun_asr_nano import model as funasr_nano_model  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"funasr not installed: {e}")
    ensure_qwen3_weights()
    patch_funasr_load_in_8bit()
    global _FUNASR_MODEL
    if _FUNASR_MODEL is None:
        device = select_asr_device(device_env)
        _FUNASR_MODEL = AutoModel(
            model=model_id,
            device=device,
            disable_update=True
        )
    res = _FUNASR_MODEL.generate(
        input=[str(audio_path)],
        cache={},
        batch_size=1,
        language=language,
        itn=True
    )
    return res[0]["text"] if res else ""


def glm_asr_transcribe_transformers(
    audio_path: Path,
    *,
    model_id: str,
    max_new_tokens_env: str = "GLM_ASR_MAX_NEW_TOKENS",
) -> str:
    global _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR
    if _GLM_ASR_MODEL is None or _GLM_ASR_PROCESSOR is None:
        _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR = load_glm_asr_transformers(model_id)
    model, processor = _GLM_ASR_MODEL, _GLM_ASR_PROCESSOR
    inputs = processor.apply_transcription_request(str(audio_path))
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        inputs = inputs.to(model.device, dtype=dtype)
    else:
        inputs = inputs.to(model.device)
    try:
        import torch
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(os.environ.get(max_new_tokens_env, "512"))
            )
    except Exception:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(os.environ.get(max_new_tokens_env, "512"))
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    return decoded[0].strip() if decoded else ""


def load_glm_asr_transformers(
    model_id: str,
    device_env: str = "GLM_ASR_DEVICE",
    dtype_env: str = "GLM_ASR_DTYPE",
) -> Tuple[object, object]:
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
