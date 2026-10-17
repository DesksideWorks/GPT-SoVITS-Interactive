"""Interactive TTS: PyTorch SoVITS with PyTorch or ORT semantic inference.

Cache-only: this engine REQUIRES a precomputed prompt cache (``.cache.pt``)
for every synthesize call. The reference-side models (CNHubert SSL,
ERes2NetV2 SV) are intentionally NOT loaded here and their weights are not
part of the runtime distribution. The separate resource-preparation workflow
creates caches after models or reference audio change; normal synthesis uses
the prebuilt caches shipped with the package.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import sys
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore")

cwd = os.path.dirname(os.path.abspath(__file__))
default_nltk_data = os.path.join(cwd, "resources", "nltk_data")
if not os.environ.get("NLTK_DATA"):
    os.environ["NLTK_DATA"] = default_nltk_data
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

from semantic_backend import load_gpt
from semantic_prompt import load_for_checkpoint, prepare_inputs
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.LangSegmenter import LangSegmenter
from GPT_SoVITS.text.cleaner import clean_text
from prompt_cache import PromptCache, load_prompt_cache
from runtime_backend import DEVICE_CHOICES, resolve_backend, resolve_device as resolve_runtime_device
from timeline import (
    english_word2ph,
    finalize_timestamps,
    japanese_word2ph,
    process_attention,
    scale_timestamps_to_duration,
)

DEFAULT_NOISE_SCALE = 0.35
DEFAULT_BENCHMARK_ROUNDS = 2
DEFAULT_BENCHMARK_WARMUP_TEXT = "\u3053\u3093\u306b\u3061\u306f\u3002"
DEFAULT_BENCHMARK_SEGMENTS = (
    "\u3053\u3093\u306b\u3061\u306f\u3001\u4eca\u65e5\u3082\u3088\u308d\u3057\u304f\u304a\u9858\u3044\u3057\u307e\u3059\u3002",
    "\u5c11\u3057\u7720\u3044\u3051\u308c\u3069\u3001\u697d\u3057\u304f\u304a\u8a71\u3057\u3057\u307e\u3057\u3087\u3046\u3002",
    "\u3055\u3063\u304d\u306e\u7d9a\u304d\u306b\u3064\u3044\u3066\u3001\u3082\u3046\u5c11\u3057\u3060\u3051\u805e\u304b\u305b\u3066\u304f\u3060\u3055\u3044\u3002",
    "\u6e96\u5099\u304c\u3067\u304d\u305f\u3089\u3001\u6b21\u306e\u4e88\u5b9a\u3092\u4e00\u7dd2\u306b\u78ba\u8a8d\u3057\u307e\u3059\u3002",
)
DEFAULT_BENCHMARK_TEXT = "".join(DEFAULT_BENCHMARK_SEGMENTS)
DEFAULT_BENCHMARK_OPTIONS = {
    "text_lang": "auto",
    "top_k": 15,
    "top_p": 1.0,
    "temperature": 1.0,
    "noise_scale": DEFAULT_NOISE_SCALE,
    "speed": 1.0,
    "pause_length": 0.0,
    "return_timeline": True,
}


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off", "no")


def frontend_language(language: Optional[str]) -> str:
    language = (language or "").replace("all_", "").lower()
    return language if language in ("zh", "ja", "en") else ""


# Characters each frontend actually speaks; a semantic prefix is not spent on bare punctuation.
SPOKEN_TEXT = {"zh": r"[\u4e00-\u9fff]", "ja": r"[\u3040-\u30ff\u4e00-\u9fff]", "en": r"[A-Za-z]"}


# Honorifics after a name: ちゃん くん さん さま たん.
HONORIFICS = ("ちゃん", "くん", "さん", "さま", "たん")
HONORIFICS_AFTER_HAN = ("ちゃん", "さん")


def _scripts(text: str) -> List[str]:
    """Per character: H hiragana, K katakana (and a name's honorific), C Han, O anything else."""
    kinds = []
    for char in text:
        code = ord(char)
        if 0x3041 <= code <= 0x3096 or char in "ゝゞ":
            kinds.append("H")
        elif 0x30A1 <= code <= 0x30FA or char in "ーヽヾ゙゚":
            kinds.append("K")
        elif 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or char == "々":
            kinds.append("C")
        else:
            kinds.append("O")
    # The ちゃん of ツクヨミちゃん or 阿米ちゃん is part of a name, not a sign of Japanese grammar around it.
    # After Han only ちゃん and さん: 行くんだ is okurigana.
    for i in range(len(text) - 1):
        if kinds[i] in ("K", "C") and kinds[i + 1] == "H":
            for honorific in (HONORIFICS if kinds[i] == "K" else HONORIFICS_AFTER_HAN):
                if text.startswith(honorific, i + 1):
                    kinds[i + 1:i + 1 + len(honorific)] = ["K"] * len(honorific)
                    break
    return kinds


def _han_runs(kinds: List[str]) -> List[Tuple[int, int, bool]]:
    """(start, end, written into hiragana) for each run of Han characters."""
    runs = []
    start = 0
    while start < len(kinds):
        if kinds[start] != "C":
            start += 1
            continue
        end = start
        while end < len(kinds) and kinds[end] == "C":
            end += 1
        runs.append((start, end, (start > 0 and kinds[start - 1] == "H") or (end < len(kinds) and kinds[end] == "H")))
        start = end
    return runs


def reads_as_chinese(text: str) -> bool:
    """Han and kana together: Chinese with Japanese words in it, or Japanese.

    In Japanese nearly every Han run carries okurigana or a particle; in Chinese the kana are names and set
    phrases. So it is Chinese when more Han characters stand apart from hiragana than there is hiragana.
    fastText cannot tell: it reads the Han next to a name, or a whole clause, as Japanese.
    """
    kinds = _scripts(text)
    if "C" not in kinds or ("H" not in kinds and "K" not in kinds):
        return False
    apart = sum(end - start for start, end, written in _han_runs(kinds) if not written)
    return apart > kinds.count("H")


def _japanese_words(text: str) -> List[Tuple[str, str]]:
    """(language, text) runs of Chinese text: kana words, with any Han written into hiragana, are Japanese."""
    kinds = _scripts(text)
    japanese = [kind in ("H", "K") for kind in kinds]
    for start, end, written in _han_runs(kinds):
        if written:
            japanese[start:end] = [True] * (end - start)
    runs: List[List[str]] = []
    leading = ""
    for char, kind, is_japanese in zip(text, kinds, japanese):
        # Punctuation goes with what it follows; digits are read in Chinese.
        if kind == "O" and not char.isalnum():
            if runs:
                runs[-1][1] += char
            else:
                leading += char
            continue
        language = "ja" if is_japanese else "zh"
        if runs and runs[-1][0] == language:
            runs[-1][1] += char
        else:
            runs.append([language, leading + char])
            leading = ""
    if leading:
        runs.append(["zh", leading])
    return [(language, item) for language, item in runs]


def _reads_in_chinese(text: str, language: Optional[str]) -> bool:
    """Whether Japanese words in this request are islands in Chinese (asked for in Chinese, or detected so)."""
    requested_language = (language or "auto").lower()
    if requested_language in ("auto", ""):
        return reads_as_chinese(text)
    return frontend_language(requested_language) == "zh"


def language_spans(text: str, language: Optional[str]) -> List[Tuple[str, str]]:
    """(language, text) runs as the frontend reads them; adjacent runs of one language merge.

    Kana have no Chinese reading, so in Chinese text Japanese words (a name like コハルネアミ) are their own
    Japanese runs and everything else is Chinese.
    """
    requested_language = (language or "auto").lower()
    if requested_language == "en":
        return [("en", text)]
    # Automatic language detection must not inherit the Japanese voice reference.
    segment_default = "" if requested_language in ("auto", "") else frontend_language(requested_language)
    if requested_language not in ("auto", "") and not segment_default:
        raise ValueError(f"Unsupported TTS language: {language}")
    chinese = _reads_in_chinese(text, requested_language)
    if chinese:
        segment_default = "zh"
    detected: List[Tuple[str, str]] = []
    for tmp in LangSegmenter.getTexts(text, segment_default):
        if detected and tmp["lang"] == detected[-1][0]:
            detected[-1] = (tmp["lang"], detected[-1][1] + tmp["text"])
        else:
            detected.append((tmp["lang"], tmp["text"]))
    if not chinese:
        return detected
    spans: List[Tuple[str, str]] = []
    for lang, item in detected:
        for piece_lang, piece in (_japanese_words(item) if lang == "zh" else [(lang, item)]):
            if spans and piece_lang == spans[-1][0]:
                spans[-1] = (piece_lang, spans[-1][1] + piece)
            else:
                spans.append((piece_lang, piece))
    return spans


def resolve_device(requested: str = "auto") -> Any:
    return resolve_runtime_device(requested)


def split_text(text: str) -> List[str]:
    """Sentences, short ones merged. Each piece is a slice of `text`: the whitespace between merged
    sentences stays, so callers can locate every piece in the original text."""
    sentences: List[Tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[\u3002\uff01\uff1f!?\u2026\n]", text):
        sentences.append((start, match.end()))
        start = match.end()
    sentences.append((start, len(text)))
    merged: List[Tuple[int, int]] = []
    current = None
    current_length = 0
    for begin, end in sentences:
        piece = text[begin:end]
        if not piece.strip():
            continue
        begin += len(piece) - len(piece.lstrip())
        end -= len(piece) - len(piece.rstrip())
        if current is not None and current_length + (end - begin) < 20:
            current = (current[0], end)
            current_length += end - begin
        else:
            if current is not None:
                merged.append(current)
            current = (begin, end)
            current_length = end - begin
    if current is not None:
        merged.append(current)
    return [text[begin:end] for begin, end in merged]


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Attribute {item} not found") from exc

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(f"Attribute {item} not found") from exc


class GPTSoVITSInference:
    """Runtime engine. Prompt cache is required; ssl/sv are never loaded."""

    def __init__(
        self,
        gpt_path: str,
        sovits_path: str,
        prompt_cache_path: str,
        device: str = "cuda",
        device_name: Optional[str] = None,
        gpt_device: Optional[str] = None,
        cpu_threads: Optional[int] = None,
        pipeline: Optional[bool] = None,
    ):
        init_started = time.perf_counter()
        self.startup_metrics: Dict[str, float] = {}
        self.is_hibernated = False
        self.lifecycle_error: Optional[str] = None
        cpu_threads_value = cpu_threads
        if cpu_threads_value is None:
            raw_cpu_threads = os.environ.get("GSV_TTS_CPU_THREADS", "").strip()
            cpu_threads_value = int(raw_cpu_threads) if raw_cpu_threads.isdigit() else None
        if cpu_threads_value is not None and cpu_threads_value > 0:
            torch.set_num_threads(int(cpu_threads_value))
            print(f"PyTorch CPU threads: {torch.get_num_threads()}")

        self.pipeline = pipeline
        self.backend = resolve_backend(device, device_name)
        gpt_device_override = (gpt_device or os.environ.get("GSV_TTS_GPT_DEVICE", "")).strip()
        self.gpt_backend = resolve_backend(gpt_device_override, device_name) if gpt_device_override else self.backend
        self.device = self.backend.torch_device
        # ORT accepts CPU text tensors and owns the GPU Transformer and KV state.
        self.gpt_device = "cpu" if self.gpt_backend.is_directml else self.gpt_backend.torch_device
        self.device_kind = self.backend.kind
        self.gpt_device_kind = self.gpt_backend.kind
        self.is_half = self.backend.is_half
        self.gpt_is_half = self.gpt_backend.is_half
        self.last_metrics: Dict[str, Any] = {}

        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        self.prompt_cache_path = prompt_cache_path
        # Small immutable CPU data; temporary device copies belong to each call.
        self._semantic_prompts = load_for_checkpoint(gpt_path, sovits_checkpoint=sovits_path)

        if not prompt_cache_path:
            raise ValueError(
                "GPTSoVITSInference requires prompt_cache_path. "
                "Prepare a reference cache with the separate resource-preparation workflow."
            )
        if not os.path.isfile(prompt_cache_path):
            raise FileNotFoundError(
                f"prompt cache file not found: {prompt_cache_path}. "
                "Prepare the reference cache with the separate resource-preparation workflow."
            )

        # Load the cache first so we know the metadata before we load any
        # model weights (gives nicer logging up front and lets us bail fast
        # on a bad cache).
        self._prompt_cache: PromptCache = load_prompt_cache(
            prompt_cache_path,
            expected_is_half=self.is_half,
            device=self.device,
        )
        self.startup_metrics["backend_and_prompt_sec"] = time.perf_counter() - init_started
        meta = self._prompt_cache.metadata
        print(
            f"[prompt_cache] loaded {prompt_cache_path}\n"
            f"[prompt_cache] ref_audio_filename={meta.ref_audio_filename}, "
            f"ref_text='{meta.ref_text[:40]}...', generated_at={meta.generated_at}"
        )

        print(
            f"Loading GPT/SoVITS on {self.backend.label} "
            f"[{self.device_kind}] (half precision: {self.is_half})..."
        )
        if self.gpt_backend.kind != self.backend.kind or self.gpt_backend.label != self.backend.label:
            print(
                f"GPT semantic model override: {self.gpt_backend.label} "
                f"[{self.gpt_device_kind}] (half precision: {self.gpt_is_half})"
            )

        stage_started = time.perf_counter()
        self.t2s_model = load_gpt(gpt_path, self.gpt_backend)
        self.config = self.t2s_model.config
        self.gpt_backend.synchronize()
        self.startup_metrics["gpt_build_and_transfer_sec"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        dict_s2 = load_sovits_new(sovits_path)
        self.startup_metrics["sovits_read_sec"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        self.hps = DictToAttrRecursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"

        if "config" in dict_s2 and "model" in dict_s2["config"] and "version" in dict_s2["config"]["model"]:
            model_version = dict_s2["config"]["model"]["version"]
        elif "sv_emb.weight" in dict_s2["weight"]:
            model_version = "v2Pro"
        else:
            _, model_version, _ = get_sovits_version_from_path_fast(sovits_path)

        self.hps.model.version = model_version
        print(f"Detected SoVITS model version: {model_version}")

        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        if self.is_half:
            self.vq_model = self.vq_model.half()
        self.vq_model = self.vq_model.to(self.device).eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        del dict_s2
        self.backend.synchronize()
        self.startup_metrics["sovits_build_and_transfer_sec"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        self.warmup()
        self.startup_metrics["kernel_warmup_sec"] = time.perf_counter() - stage_started
        self.startup_metrics["engine_total_sec"] = time.perf_counter() - init_started
        print(f"[tts_startup] {self.startup_metrics}")

    @property
    def sample_rate(self) -> int:
        return int(self.hps.data.sampling_rate)

    @property
    def model_version(self) -> str:
        return str(self.hps.model.version)

    @property
    def prompt_cache(self) -> PromptCache:
        return self._prompt_cache

    def _move_to_cpu_for_release(self, value: Any) -> None:
        if isinstance(value, torch.nn.Module):
            value.to("cpu")

    def _move_runtime(self, sleeping: bool) -> None:
        """Move all owned tensor state, including caches outside state_dict."""
        from torch.nn.utils.weight_norm import WeightNorm

        if sleeping:
            self.t2s_model.hibernate()
        else:
            self.t2s_model.wake()
        for model, target in ((self.vq_model, "cpu" if sleeping else self.device),):
            model.to(target)
            for module in model.modules():
                # Attention retained for the previous utterance's timeline.
                if isinstance(getattr(module, "attn", None), torch.Tensor):
                    module.attn = None
                # Legacy weight_norm stores a derived tensor outside buffers.
                # Recompute it on the destination so it cannot retain old VRAM.
                with torch.no_grad():
                    for hook in module._forward_pre_hooks.values():
                        if isinstance(hook, WeightNorm):
                            setattr(module, hook.name, hook.compute_weight(module))
                if isinstance(getattr(module, "weight_inv", None), torch.Tensor):
                    module.weight_inv = module.weight_inv.to(target)
        self.t2s_model.model.rebind_inference_blocks()
        cache = self._prompt_cache
        for name in ("prompt_semantic", "bert1", "refer_spec", "sv_emb"):
            target = "cpu" if sleeping or name in ("prompt_semantic", "bert1") else self.device
            setattr(cache, name, getattr(cache, name).to(target))
        self.gpt_backend.synchronize()
        self.backend.synchronize()

    def hibernate(self) -> None:
        if self.lifecycle_error:
            raise RuntimeError(self.lifecycle_error)
        if self.is_hibernated:
            return
        started = time.perf_counter()
        try:
            self._move_runtime(sleeping=True)
            gc.collect()
            self.backend.empty_cache()
            if self.gpt_backend is not self.backend:
                self.gpt_backend.empty_cache()
            self.is_hibernated = True
        except Exception as exc:
            self.lifecycle_error = f"TTS hibernation failed: {exc}"
            raise
        print(f"[tts_lifecycle] hibernated seconds={time.perf_counter() - started:.3f}")

    def wake(self) -> None:
        if self.lifecycle_error:
            raise RuntimeError(self.lifecycle_error)
        if not self.is_hibernated:
            return
        started = time.perf_counter()
        try:
            self._move_runtime(sleeping=False)
            self.is_hibernated = False
        except Exception as exc:
            self.lifecycle_error = f"TTS wake failed: {exc}"
            raise
        print(f"[tts_lifecycle] awake seconds={time.perf_counter() - started:.3f}")

    def close(self) -> None:
        semantic = getattr(self, "t2s_model", None)
        if semantic is not None:
            semantic.close()
        for name in ("t2s_model", "vq_model"):
            value = getattr(self, name, None)
            if value is not None:
                try:
                    self._move_to_cpu_for_release(value)
                except Exception as exc:
                    print(f"Release warning for {name}: {exc}")
                setattr(self, name, None)
        self._prompt_cache = None  # type: ignore[assignment]
        gc.collect()
        self.backend.empty_cache()
        if self.gpt_backend is not self.backend:
            self.gpt_backend.empty_cache()

    def warmup(self) -> None:
        print("Warming up GPT/SoVITS backends...")
        try:
            phones, bert, _ = self.get_phones_and_bert(
                "\u3053\u3093\u306b\u3061\u306f\u3002", "ja", self.hps.model.version
            )
            dummy_prompt = torch.zeros((1, 1), dtype=torch.long, device=self.gpt_device)
            dummy_bert = torch.zeros(
                (1, 1024, len(phones) + 1),
                dtype=torch.float16 if self.gpt_is_half else torch.float32,
                device=self.gpt_device,
            )
            dummy_phones = torch.LongTensor(phones + [0]).unsqueeze(0).to(self.gpt_device)
            dummy_phones_len = torch.tensor([dummy_phones.shape[-1]], device=self.gpt_device)
            if self.gpt_backend.is_directml:
                self.t2s_model.warmup()
                pred_semantic = torch.zeros((1, 16), dtype=torch.long)
            else:
                pred_semantic = None
            with torch.inference_mode():
                if pred_semantic is None:
                    pred_semantic, _ = self.t2s_model.model.infer_panel(
                        dummy_phones, dummy_phones_len, dummy_prompt, dummy_bert,
                        top_k=5, top_p=1, temperature=1, early_stop_num=50,
                    )
            dummy_spec = torch.zeros(
                (1, self.hps.data.filter_length // 2 + 1, 10),
                device=self.device,
                dtype=torch.float16 if self.is_half else torch.float32,
            )
            dummy_semantic = pred_semantic[:, dummy_prompt.shape[1] :].unsqueeze(0).detach().clone().to(self.device)
            with torch.no_grad():
                _ = self.vq_model.decode(
                    dummy_semantic,
                    torch.LongTensor(phones).unsqueeze(0).to(self.device),
                    [dummy_spec],
                    noise_scale=DEFAULT_NOISE_SCALE,
                    sv_emb=[self._prompt_cache.sv_emb] if getattr(self.vq_model, "is_v2pro", False) else None,
                )
            self.gpt_backend.synchronize()
            self.backend.synchronize()
            print("Warmup complete.")
        except Exception as exc:
            print(f"Warmup warning: {exc}")

    def empty_bert(self, phones: List[int]) -> torch.Tensor:
        return torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if self.gpt_is_half else torch.float32,
            device=self.gpt_device,
        )

    def get_phones_and_bert_with_metadata(
        self,
        text: str,
        language: str,
        version: str,
        default_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        # A line break left inside merged sentences is read like a space, not as the Japanese
        # frontend's "." pause; same length, so timeline indices still match the original text.
        text = re.sub(r" {2,}", " ", text.replace("\n", " "))
        textlist: List[str] = []
        langlist: List[str] = []
        for lang, item_text in language_spans(text, language):
            if lang not in ("zh", "ja", "en"):
                raise ValueError(f"Unsupported detected TTS language: {lang}")
            langlist.append(lang)
            textlist.append(item_text)

        phones_list: List[List[int]] = []
        phone_symbols_list: List[str] = []
        bert_list: List[torch.Tensor] = []
        norm_text_list: List[str] = []
        word2ph_list: List[int] = []

        for item_text, lang in zip(textlist, langlist):
            phone_symbols, word2ph, norm_text = clean_text(item_text, lang, version)
            phone_ids = cleaned_text_to_sequence(phone_symbols, version)
            if lang == "ja":
                final_word2ph = japanese_word2ph(norm_text, phone_symbols)
            elif lang == "en":
                final_word2ph = english_word2ph(norm_text, phone_symbols)
            else:
                final_word2ph = word2ph
            if lang == "zh":
                from chinese_frontend import get_bert_feature
                bert = get_bert_feature(norm_text, final_word2ph).to(self.gpt_device, dtype=self._gpt_dtype())
            else:
                bert = self.empty_bert(phone_ids)
            phones_list.append(phone_ids)
            phone_symbols_list.extend(list(phone_symbols))
            bert_list.append(bert)
            norm_text_list.append(norm_text)
            word2ph_list.extend(final_word2ph)

        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)
        return {
            "phones": phones,
            "phone_symbols": phone_symbols_list,
            "word2ph": word2ph_list,
            "bert": bert.to(torch.float16 if self.gpt_is_half else torch.float32),
            "norm_text": norm_text,
            "language_spans": [dict(language=lang, text=item) for item, lang in zip(textlist, langlist)],
        }

    def get_phones_and_bert(
        self,
        text: str,
        language: str,
        version: str,
        default_lang: Optional[str] = None,
    ) -> Tuple[List[int], torch.Tensor, str]:
        meta = self.get_phones_and_bert_with_metadata(text, language, version, default_lang)
        return meta["phones"], meta["bert"], meta["norm_text"]

    def _mrte_attention(self) -> Optional[torch.Tensor]:
        try:
            attn = self.vq_model.enc_p.mrte.cross_attention.attn
        except AttributeError:
            return None
        if attn is None:
            return None
        return attn.float()

    def _gpt_dtype(self) -> torch.dtype:
        return torch.float16 if self.gpt_is_half else torch.float32

    def _pipeline_enabled(self, requested: Optional[bool], segment_count: int) -> bool:
        if segment_count < 2:
            return False
        if self.gpt_backend.kind == self.backend.kind and self.gpt_backend.label == self.backend.label:
            return False
        if requested is not None:
            return requested
        return env_flag("GSV_TTS_PIPELINE", True)

    def _text_segments(self, text: str, text_lang: str) -> List[str]:
        """Fit bounded graph inputs without dropping text or splitting phonemes."""
        segments = split_text(text)
        budget = getattr(self.t2s_model, "max_segment_phones", None)
        if budget is None:
            return segments
        semantic_prompt = self._semantic_prompt_for_text(text, text_lang)
        if semantic_prompt is not None:
            budget = self.t2s_model.segment_phone_budget(semantic_prompt.phone_count, semantic_prompt.token_count)
        def fit(segment):
            meta = self.get_phones_and_bert_with_metadata(
                segment, text_lang, self.model_version, default_lang=self._prompt_cache.metadata.ref_lang or "ja")
            if len(meta['phones']) <= budget:
                return [segment]
            if len(segment) < 2:
                raise ValueError("Text cannot fit the GPT phoneme budget")
            mid = len(segment)//2
            boundaries = [i+1 for i,c in enumerate(segment[:-1]) if c in " ,，、;；:："]
            cut = min(boundaries, key=lambda i: abs(i-mid)) if boundaries else mid
            return fit(segment[:cut]) + fit(segment[cut:])
        return [part for segment in segments for part in fit(segment)]

    def _semantic_prompt_for_text(self, text: str, text_lang: str):
        """Condition requests read wholly in one language that has a reviewed reference."""
        prompts = getattr(self, '_semantic_prompts', None)
        if not prompts or not re.search("|".join(SPOKEN_TEXT.values()), text):
            return None
        languages = {language for language, _ in language_spans(text, text_lang)}
        # A Japanese name inside Chinese leaves it a Chinese request; without the prefix Chinese can stall in
        # silence. Other mixes are not conditioned on any single language.
        if languages == {"zh", "ja"} and _reads_in_chinese(text, text_lang):
            languages = {"zh"}
        if len(languages) != 1:
            return None
        language = languages.pop()
        prompt = prompts.get(language)
        return prompt if prompt is not None and re.search(SPOKEN_TEXT[language], text) else None

    def _gpt_inputs(self, target_meta, semantic_prompt=None):
        return prepare_inputs(target_meta, self.gpt_device, self._gpt_dtype(), semantic_prompt)

    def _generate_segment_semantic(
        self,
        seg: str,
        text_lang: str,
        prompt_lang_for_segment: str,
        top_k: int,
        top_p: float,
        temperature: float,
        semantic_prompt=None,
    ) -> Dict[str, Any]:
        t_seg_text_start = time.perf_counter()
        target_meta = self.get_phones_and_bert_with_metadata(
            seg, text_lang, self.hps.model.version, default_lang=prompt_lang_for_segment
        )
        phones2 = target_meta["phones"]
        self.gpt_backend.synchronize()
        text_time = time.perf_counter() - t_seg_text_start

        all_phoneme_ids, all_phoneme_len, prompt, bert = self._gpt_inputs(target_meta, semantic_prompt)

        t_gpt_start = time.perf_counter()
        with torch.inference_mode():
            pred_semantic, idx = self.t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=50 * 30,
            )
            # Both backends return the prefix in complete mode. Acoustic input
            # and timeline indices must contain only newly generated speech.
            prefix_length = 0 if prompt is None else prompt.shape[-1]
            pred_semantic = pred_semantic[:, prefix_length:].unsqueeze(0)
            if pred_semantic.shape[-1] == 0:
                raise RuntimeError('GPT generated no target speech')
            idx = int(pred_semantic.shape[-1])
        self.gpt_backend.synchronize()
        pred_semantic = pred_semantic.detach().clone()

        return {
            "target_meta": target_meta,
            "phones2": phones2,
            "pred_semantic": pred_semantic,
            "idx": int(idx),
            "text_time": text_time,
            "gpt_time": time.perf_counter() - t_gpt_start,
        }

    def synthesize(
        self,
        text: str,
        text_lang: str = "auto",
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        speed: float = 1.0,
        pause_length: float = 0.3,
        return_timeline: bool = False,
        pipeline: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        print(f"Inferencing: {text} ({text_lang})")
        self.gpt_backend.synchronize()
        self.backend.synchronize()
        t_all_start = time.perf_counter()
        semantic_prompt = self._semantic_prompt_for_text(text, text_lang)
        # Every fragment of a prefixed request is read in the prefix's language, not detected again.
        frontend_lang = semantic_prompt.language if semantic_prompt is not None else text_lang
        segments = self._text_segments(text, frontend_lang)
        if not segments:
            self.last_metrics = {
                "text_lang": text_lang,
                "segment_count": 0,
                "effective_pipeline": False,
                "target_text_cleaning_sec": 0.0,
                "gpt_semantic_sec": 0.0,
                "gpt_tokens": 0,
                "gpt_tokens_per_sec": 0.0,
                "sovits_audio_decode_sec": 0.0,
                "first_segment_latency_sec": 0.0,
                "audio_duration_sec": 0.0,
                "inference_sec": 0.0,
                "rtf": None,
                "timestamp_count": 0,
            }
            return np.zeros(0, dtype=np.float32), []

        final_audios: List[np.ndarray] = []
        final_timestamps: List[Dict[str, Any]] = []
        audio_offset_samples = 0
        source_search_offset = 0
        sr = self.sample_rate

        cache = self._prompt_cache
        refer_spec = cache.refer_spec.to(self.device)
        sv_emb = cache.sv_emb.to(self.device)
        prompt_lang_for_segment = cache.metadata.ref_lang or "ja"

        total_text_time = 0.0
        total_gpt_time = 0.0
        total_sovits_time = 0.0
        t_first_segment = 0.0
        total_gpt_tokens = 0
        use_pipeline = self._pipeline_enabled(self.pipeline if pipeline is None else pipeline, len(segments))
        if use_pipeline:
            print("Segment pipeline: enabled (GPT and SoVITS use separate backends)")

        executor: Optional[ThreadPoolExecutor] = None
        next_future: Optional[Future] = None
        if use_pipeline:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpt")
            next_future = executor.submit(
                self._generate_segment_semantic,
                segments[0],
                frontend_lang,
                prompt_lang_for_segment,
                top_k,
                top_p,
                temperature,
                semantic_prompt,
            )

        try:
            for seg_idx, seg in enumerate(segments):
                print(f"Processing segment {seg_idx + 1}/{len(segments)}: {seg}")
                segment_char_offset = text.find(seg, source_search_offset)
                if segment_char_offset < 0:
                    segment_char_offset = source_search_offset
                source_search_offset = min(len(text), segment_char_offset + len(seg))

                if use_pipeline:
                    assert next_future is not None
                    segment_result = next_future.result()
                    if seg_idx + 1 < len(segments):
                        assert executor is not None
                        next_future = executor.submit(
                            self._generate_segment_semantic,
                            segments[seg_idx + 1],
                            frontend_lang,
                            prompt_lang_for_segment,
                            top_k,
                            top_p,
                            temperature,
                            semantic_prompt,
                        )
                    else:
                        next_future = None
                else:
                    segment_result = self._generate_segment_semantic(
                        seg,
                        frontend_lang,
                        prompt_lang_for_segment,
                        top_k,
                        top_p,
                        temperature,
                        semantic_prompt,
                    )

                target_meta = segment_result["target_meta"]
                phones2 = segment_result["phones2"]
                pred_semantic = segment_result["pred_semantic"]
                total_text_time += float(segment_result["text_time"])
                total_gpt_time += float(segment_result["gpt_time"])
                total_gpt_tokens += int(segment_result["idx"])
                pred_semantic = pred_semantic.to(self.device)

                t_sovits_start = time.perf_counter()
                with torch.no_grad():
                    audio = self.vq_model.decode(
                        pred_semantic,
                        torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                        [refer_spec],
                        noise_scale=noise_scale,
                        speed=speed,
                        sv_emb=[sv_emb],
                    )[0][0]
                    attention = self._mrte_attention() if return_timeline else None
                self.backend.synchronize()
                total_sovits_time += time.perf_counter() - t_sovits_start

                audio_np = audio.cpu().float().numpy()
                audio_np = audio_np - np.mean(audio_np)
                final_audios.append(audio_np)

                if return_timeline and attention is not None:
                    semantic_len = int(attention.shape[2]) if attention.ndim == 4 else int(pred_semantic.shape[-1] * 2)
                    timestamps_batch = process_attention(
                        attention,
                        [target_meta["word2ph"]],
                        [semantic_len],
                        speed,
                        [target_meta["phone_symbols"]],
                    )
                    if timestamps_batch:
                        segment_offset_seconds = audio_offset_samples / float(sr)
                        segment_timestamps = finalize_timestamps(
                            timestamps_batch[0],
                            target_meta["norm_text"],
                            segment_offset_seconds,
                            original_text=seg,
                            char_offset=segment_char_offset,
                        )
                        final_timestamps.extend(
                            scale_timestamps_to_duration(
                                segment_timestamps,
                                len(audio_np) / float(sr),
                                segment_offset_seconds,
                            )
                        )

                audio_offset_samples += len(audio_np)
                if seg_idx == 0:
                    t_first_segment = time.perf_counter() - t_all_start

                if seg_idx < len(segments) - 1 and pause_length > 0:
                    pause_audio = np.zeros(int(sr * pause_length), dtype=np.float32)
                    final_audios.append(pause_audio)
                    audio_offset_samples += len(pause_audio)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

        audio_final = np.concatenate(final_audios).astype(np.float32)
        max_amp = np.abs(audio_final).max()
        if max_amp > 1e-5:
            audio_final = audio_final / max_amp * 0.9

        total_audio_duration = len(audio_final) / float(sr)
        total_inference_time = time.perf_counter() - t_all_start
        rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0.0
        gpt_tps = total_gpt_tokens / total_gpt_time if total_gpt_time > 0 else 0.0
        self.last_metrics = {
            "text_lang": text_lang,
            "segment_count": len(segments),
            "effective_pipeline": use_pipeline,
            "target_text_cleaning_sec": total_text_time,
            "gpt_semantic_sec": total_gpt_time,
            "gpt_tokens": total_gpt_tokens,
            "gpt_tokens_per_sec": gpt_tps,
            "sovits_audio_decode_sec": total_sovits_time,
            "first_segment_latency_sec": t_first_segment,
            "audio_duration_sec": total_audio_duration,
            "inference_sec": total_inference_time,
            "rtf": rtf,
            "timestamp_count": len(final_timestamps),
            "semantic_prompt_id": semantic_prompt.identifier if semantic_prompt is not None else None,
            "semantic_prompt_tokens": semantic_prompt.token_count if semantic_prompt is not None else 0,
        }

        print("\n--- Inference Performance Summary ---")
        print(f"Target Text Cleaning:  {total_text_time:.3f}s")
        print(f"GPT Semantic Gen:      {total_gpt_time:.3f}s ({gpt_tps:.2f} tokens/s)")
        print(f"SoVITS Audio Decode:   {total_sovits_time:.3f}s")
        print(f"First Segment Latency: {t_first_segment:.3f}s")
        print(f"Total Audio Duration:  {total_audio_duration:.3f}s")
        print(f"Total Inference Time:  {total_inference_time:.3f}s")
        print(f"Real Time Factor (RTF): {rtf:.4f}")
        print("-------------------------------------\n")

        return audio_final, final_timestamps

    def synthesize_stream(
        self,
        text: str,
        text_lang: str = "auto",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        noise_scale: float = DEFAULT_NOISE_SCALE,
        speed: float = 1.0,
        pause_length: float = 0.3,
        chunk_length: int = 24,
    ) -> Iterator[np.ndarray]:
        """Compatibility PCM iterator using the same committed-frame decoder."""
        from streaming_latents import LatentStream
        stream = LatentStream(self, token_chunk=chunk_length).stream(
            text, language=text_lang, speed=speed, pause=pause_length,
            noise_scale=noise_scale, top_k=top_k, top_p=top_p,
            temperature=temperature, timeline=False)
        try:
            for event in stream:
                if event['type'] == 'chunk':
                    yield event['pcm']
        finally:
            stream.close()

    def benchmark(self, rounds: int = DEFAULT_BENCHMARK_ROUNDS) -> Dict[str, Any]:
        """Run the fixed short-dialog benchmark on the loaded engine."""

        rounds = max(1, int(rounds))
        options = DEFAULT_BENCHMARK_OPTIONS.copy()
        benchmark_segments = split_text(DEFAULT_BENCHMARK_TEXT)

        warmup_audio, warmup_timestamps = self.synthesize(
            text=DEFAULT_BENCHMARK_WARMUP_TEXT,
            text_lang=str(options["text_lang"]),
            top_k=int(options["top_k"]),
            top_p=float(options["top_p"]),
            temperature=float(options["temperature"]),
            noise_scale=float(options["noise_scale"]),
            speed=float(options["speed"]),
            pause_length=float(options["pause_length"]),
            return_timeline=bool(options["return_timeline"]),
        )
        warmup_metrics = dict(self.last_metrics)
        warmup_metrics.update(
            {
                "audio_samples": int(len(warmup_audio)),
                "timestamp_count": len(warmup_timestamps),
            }
        )

        self.gpt_backend.empty_cache()
        self.backend.empty_cache()
        self.gpt_backend.reset_peak_memory_stats()
        self.backend.reset_peak_memory_stats()

        runs: List[Dict[str, Any]] = []
        for index in range(rounds):
            self.gpt_backend.synchronize()
            self.backend.synchronize()
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            audio, timestamps = self.synthesize(
                text=DEFAULT_BENCHMARK_TEXT,
                text_lang=str(options["text_lang"]),
                top_k=int(options["top_k"]),
                top_p=float(options["top_p"]),
                temperature=float(options["temperature"]),
                noise_scale=float(options["noise_scale"]),
                speed=float(options["speed"]),
                pause_length=float(options["pause_length"]),
                return_timeline=bool(options["return_timeline"]),
            )
            self.gpt_backend.synchronize()
            self.backend.synchronize()
            wall_sec = time.perf_counter() - wall_start
            cpu_sec = time.process_time() - cpu_start
            metrics = dict(self.last_metrics)
            average_cpu_cores = cpu_sec / wall_sec if wall_sec > 0 else 0.0
            logical_cpus = os.cpu_count() or 1
            runs.append(
                {
                    "round": index + 1,
                    "audio_samples": int(len(audio)),
                    "timestamp_count": len(timestamps),
                    "outer_wall_sec": wall_sec,
                    "process_cpu_sec": cpu_sec,
                    "average_cpu_cores": average_cpu_cores,
                    "machine_cpu_percent_of_logical": (average_cpu_cores / logical_cpus) * 100.0,
                    **metrics,
                }
            )

        def mean(key: str) -> Optional[float]:
            values = [float(run[key]) for run in runs if run.get(key) is not None]
            return sum(values) / len(values) if values else None

        aggregate = {
            "rtf": mean("rtf"),
            "inference_sec": mean("inference_sec"),
            "outer_wall_sec": mean("outer_wall_sec"),
            "process_cpu_sec": mean("process_cpu_sec"),
            "audio_duration_sec": mean("audio_duration_sec"),
            "gpt_semantic_sec": mean("gpt_semantic_sec"),
            "gpt_tokens": mean("gpt_tokens"),
            "gpt_tokens_per_sec": mean("gpt_tokens_per_sec"),
            "sovits_audio_decode_sec": mean("sovits_audio_decode_sec"),
            "first_segment_latency_sec": mean("first_segment_latency_sec"),
            "average_cpu_cores": mean("average_cpu_cores"),
            "machine_cpu_percent_of_logical": mean("machine_cpu_percent_of_logical"),
        }

        return {
            "benchmark_id": "interactive_dialog_short_4seg_v1",
            "description": "Fixed interactive-dialog benchmark: four short conversational segments.",
            "rounds": rounds,
            "text": DEFAULT_BENCHMARK_TEXT,
            "segments": benchmark_segments,
            "options": options,
            "sample_rate": self.sample_rate,
            "resolved_backends": {
                "gpt": self.gpt_backend.as_dict(),
                "sovits": self.backend.as_dict(),
            },
            "torch_threads": {
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
            },
            "warmup": warmup_metrics,
            "aggregate": aggregate,
            "runs": runs,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-SoVITS Interactive PyTorch Inference (cache-only)")
    parser.add_argument("--gpt_path", required=True, help="Path to GPT model (.ckpt)")
    parser.add_argument("--sovits_path", required=True, help="Path to SoVITS model (.pth)")
    parser.add_argument("--prompt_cache_path", required=True, help="Path to prebuilt .cache.pt")
    parser.add_argument("--device", default="cpu", choices=DEVICE_CHOICES)
    parser.add_argument("--device_name", default=None)
    parser.add_argument("--gpt_device", default=None, choices=DEVICE_CHOICES)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--lang", default="auto")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--noise_scale", type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument("--pause_length", type=float, default=0.3)
    parser.add_argument("--timeline", action="store_true")
    args = parser.parse_args()

    inference = GPTSoVITSInference(
        gpt_path=args.gpt_path,
        sovits_path=args.sovits_path,
        prompt_cache_path=args.prompt_cache_path,
        device=args.device,
        device_name=args.device_name,
        gpt_device=args.gpt_device,
        cpu_threads=args.cpu_threads,
        pipeline=args.pipeline,
    )
    audio, timestamps = inference.synthesize(
        args.text,
        text_lang=args.lang,
        noise_scale=args.noise_scale,
        pause_length=args.pause_length,
        return_timeline=args.timeline,
        pipeline=args.pipeline,
    )

    import soundfile as sf

    sf.write(args.output, audio, inference.sample_rate)
    print(f"Saved to {args.output}")
    if args.timeline:
        print(f"Timestamps: {len(timestamps)}")
