import argparse
import time

PROCESS_IMPORT_STARTED = time.perf_counter()

import asyncio
import anyio
import base64
import gc
import hmac
import io
import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, "GPT_SoVITS"))

from runtime_backend import DEVICE_CHOICES

API_IMPORT_SECONDS = time.perf_counter() - PROCESS_IMPORT_STARTED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gpt-sovits-torch-api")


parser = argparse.ArgumentParser(description="GPT-SoVITS Interactive PyTorch API Server", add_help=False)
parser.add_argument("--voices_config", default=os.environ.get("GSV_TTS_VOICES_CONFIG", "resources/voices.json"))
parser.add_argument("--device", default=os.environ.get("GSV_TTS_DEVICE", "cpu"), choices=DEVICE_CHOICES)
parser.add_argument("--device_name", default=os.environ.get("GSV_TTS_DEVICE_NAME") or None)
parser.add_argument("--gpt_device", default=os.environ.get("GSV_TTS_GPT_DEVICE") or None, choices=DEVICE_CHOICES)
parser.add_argument("--cpu_threads", type=int, default=int(os.environ["GSV_TTS_CPU_THREADS"]) if os.environ.get("GSV_TTS_CPU_THREADS", "").isdigit() else None)
parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--token", default=os.environ.get("GSV_TTS_TOKEN", ""))
parser.add_argument("--avatar", default="")
parser.add_argument("--reference", default="")
parser.add_argument("--resources", default=os.environ.get("GSV_TTS_RESOURCES", ""))
args, _ = parser.parse_known_args()

runtime_resources_dir = cwd
voices_config_base_dir = cwd


class TtsStartupStatus(IntEnum):
    OK = 0
    INVALID_ARGUMENTS = 10
    TOKEN_REQUIRED = 11
    AVATAR_REQUIRED = 12
    AVATAR_NOT_FOUND = 13
    REFERENCE_NOT_FOUND = 14
    CONFIG_LOAD_FAILED = 20
    RESOURCE_MISSING = 22
    MODEL_INIT_FAILED = 30
    PORT_UNAVAILABLE = 31
    STARTUP_FAILED = 90


STARTUP_MESSAGES = {
    TtsStartupStatus.OK: "TTS service is ready.",
    TtsStartupStatus.INVALID_ARGUMENTS: "Invalid startup arguments.",
    TtsStartupStatus.TOKEN_REQUIRED: "Startup token is required.",
    TtsStartupStatus.AVATAR_REQUIRED: "Startup avatar is required.",
    TtsStartupStatus.AVATAR_NOT_FOUND: "Startup avatar is not allowed by the runtime config.",
    TtsStartupStatus.REFERENCE_NOT_FOUND: "Startup reference is not allowed for this avatar.",
    TtsStartupStatus.CONFIG_LOAD_FAILED: "Runtime config could not be loaded.",
    TtsStartupStatus.RESOURCE_MISSING: "Required resource is missing or the installation is incomplete.",
    TtsStartupStatus.MODEL_INIT_FAILED: "TTS model failed to initialize.",
    TtsStartupStatus.PORT_UNAVAILABLE: "The local API port is unavailable.",
    TtsStartupStatus.STARTUP_FAILED: "TTS service failed to start.",
}


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    if not path:
        return path
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(base_dir or cwd, path))


def resolve_existing_path(path: str, *base_dirs: Optional[str]) -> str:
    if not path or os.path.isabs(path):
        return resolve_path(path)

    candidates = []
    for base_dir in base_dirs:
        if base_dir:
            candidates.append(resolve_path(path, base_dir))
    candidates.append(resolve_path(path, cwd))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def argument_present(*names: str) -> bool:
    argv = sys.argv[1:]
    return any(arg == name or arg.startswith(name + "=") for arg in argv for name in names)


def default_resources_dir() -> str:
    env_resources = os.environ.get("GSV_TTS_RESOURCES", "")
    if env_resources:
        return resolve_path(env_resources)
    bundled_resources = resolve_path("resources", cwd)
    if os.path.isdir(bundled_resources):
        return bundled_resources
    return cwd


def choose_voices_config_path(path: str, resources_dir: str, explicit: bool) -> str:
    if explicit:
        return resolve_path(path)
    candidates = []
    if resources_dir:
        candidates.append(resolve_path("voices.json", resources_dir))
        candidates.append(resolve_path(os.path.join("config", "voices.json"), resources_dir))
    candidates.append(resolve_path(path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def choose_runtime_path(path: str, resources_dir: str, explicit: bool) -> str:
    if explicit:
        return resolve_path(path)
    return resolve_existing_path(path, resources_dir, cwd)


def configure_resource_environment(resources_dir: str) -> None:
    resources_dir = resolve_path(resources_dir) if resources_dir else cwd
    pretrained_root = resolve_path("pretrained_models", resources_dir)
    nltk_data = resolve_path("nltk_data", resources_dir)

    os.environ["GSV_TTS_RESOURCES"] = resources_dir
    os.environ["GSV_TTS_PRETRAINED_MODELS"] = pretrained_root
    os.environ["NLTK_DATA"] = nltk_data


def first_value(data: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = data.get(name)
        if value not in (None, ""):
            return value
    return default


def startup_payload(
    status: TtsStartupStatus,
    port: Optional[int] = None,
    avatar_id: Optional[str] = None,
    reference_id: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event": "startup",
        "ok": status == TtsStartupStatus.OK,
        "code": status.name,
        "status": int(status),
        "message": STARTUP_MESSAGES.get(status, STARTUP_MESSAGES[TtsStartupStatus.STARTUP_FAILED]),
    }
    if detail:
        payload["detail"] = detail
    if port is not None:
        payload["port"] = port
    if avatar_id:
        payload["avatarId"] = avatar_id
    if reference_id:
        payload["referenceId"] = reference_id
    if status == TtsStartupStatus.OK and port is not None:
        payload["endpoint"] = f"http://127.0.0.1:{port}"
    return payload


def write_startup_status(payload: Dict[str, Any]) -> None:
    print("GSV_TTS_STARTUP " + json.dumps(payload, ensure_ascii=False), flush=True)


def fail_startup(
    status: TtsStartupStatus,
    detail: Optional[str] = None,
    port: Optional[int] = None,
    avatar_id: Optional[str] = None,
    reference_id: Optional[str] = None,
) -> None:
    write_startup_status(startup_payload(status, port, avatar_id, reference_id, detail))
    raise SystemExit(int(status))


def is_port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def audio_array_to_wav_bytes(audio_data: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio_data.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def authorize_request(request: Request, query_token: Optional[str] = None) -> None:
    expected = args.token
    if not expected:
        raise HTTPException(status_code=401, detail="unauthorized")

    auth = request.headers.get("authorization", "")
    bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
    header_token = request.headers.get("x-tts-token", "")
    supplied = query_token or bearer or header_token
    if supplied and hmac.compare_digest(str(supplied), str(expected)):
        return
    raise HTTPException(status_code=401, detail="unauthorized")


class ModelManager:
    def __init__(
        self,
        device: str,
        device_name: Optional[str] = None,
        gpt_device: Optional[str] = None,
        cpu_threads: Optional[int] = None,
        pipeline: Optional[bool] = None,
    ):
        self.engines: Dict[str, Any] = {}
        self.active_key: Optional[str] = None
        self.device = device
        self.device_name = device_name
        self.gpt_device = gpt_device
        self.cpu_threads = cpu_threads
        self.pipeline = pipeline

    def clear(self) -> None:
        old_engines = list(self.engines.values())
        self.engines.clear()
        self.active_key = None
        for engine in old_engines:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        old_engines.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_engine(self, gpt_path: str, sovits_path: str, prompt_cache_path: str) -> Any:
        import_started = time.perf_counter()
        from run_inference import GPTSoVITSInference

        full_gpt_path = resolve_existing_path(gpt_path, runtime_resources_dir, voices_config_base_dir, cwd)
        full_sovits_path = resolve_existing_path(sovits_path, runtime_resources_dir, voices_config_base_dir, cwd)
        full_cache_path = resolve_existing_path(prompt_cache_path, runtime_resources_dir, voices_config_base_dir, cwd)
        key = (
            f"{full_gpt_path}|{full_sovits_path}|{full_cache_path}|"
            f"device={self.device}|device_name={self.device_name or ''}|gpt_device={self.gpt_device or ''}|"
            f"cpu_threads={self.cpu_threads or ''}|pipeline={self.pipeline}"
        )
        if key not in self.engines:
            logger.info("[tts_startup] inference_import_sec=%.3f api_import_sec=%.3f",
                        time.perf_counter() - import_started, API_IMPORT_SECONDS)
            if self.active_key and self.active_key != key:
                self.clear()
            logger.info(
                "Loading TTS model engine: GPT=%s SoVITS=%s cache=%s device=%s device_name=%s gpt_device=%s cpu_threads=%s pipeline=%s",
                full_gpt_path,
                full_sovits_path,
                full_cache_path,
                self.device,
                self.device_name or "",
                self.gpt_device or "",
                self.cpu_threads or "",
                self.pipeline,
            )
            self.engines[key] = GPTSoVITSInference(
                gpt_path=full_gpt_path,
                sovits_path=full_sovits_path,
                prompt_cache_path=full_cache_path,
                device=self.device,
                device_name=self.device_name,
                gpt_device=self.gpt_device,
                cpu_threads=self.cpu_threads,
                pipeline=self.pipeline,
            )
            self.active_key = key
        engine = self.engines[key]
        engine.wake()
        return engine

    @property
    def active_engine(self):
        return self.engines.get(self.active_key)

    def hibernate(self) -> None:
        if self.active_engine is not None:
            self.active_engine.hibernate()


class VoiceManager:
    def __init__(self, config_path: str, required: bool = False, autoload: bool = True):
        self.config_path = resolve_path(config_path)
        self.config_base_dir = os.path.dirname(self.config_path) or cwd
        self.voices: Dict[str, Any] = {}
        self.active_voice_id: str = ""
        self.active_reference_id: str = "default"
        if autoload:
            self.load_configs(required=required)

    def configure(self, config_path: str) -> None:
        self.config_path = resolve_path(config_path)
        self.config_base_dir = os.path.dirname(self.config_path) or cwd

    def _normalize_configs(self, raw: Any) -> Dict[str, Any]:
        source = raw
        if isinstance(raw, dict):
            source = raw.get("voices") or raw.get("Voices") or raw.get("avatars") or raw.get("Avatars") or raw
        if isinstance(source, list):
            result = {}
            for item in source:
                if not isinstance(item, dict):
                    continue
                voice_id = first_value(item, "id", "Id", "model_id", "modelId", "avatarId")
                if voice_id:
                    result[str(voice_id)] = item
            return result
        if isinstance(source, dict):
            return {str(key): value for key, value in source.items() if isinstance(value, dict)}
        raise ValueError("Voice config must be a JSON object or list.")

    def load_configs(self, required: bool = False) -> None:
        if not os.path.exists(self.config_path):
            if required:
                raise FileNotFoundError(self.config_path)
            logger.warning("Voice config file not found: %s", self.config_path)
            self.voices = {}
        else:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.voices = self._normalize_configs(json.load(f))
            except Exception:
                if required:
                    raise
                logger.warning("Voice config could not be loaded: %s", self.config_path, exc_info=True)
                self.voices = {}

        if self.voices and self.active_voice_id not in self.voices:
            self.active_voice_id = next(iter(self.voices.keys()))
            self.active_reference_id = self.default_reference_id(self.voices[self.active_voice_id])
        logger.info("Loaded %d voice configurations.", len(self.voices))

    def resolve_file(self, path: str) -> str:
        return resolve_existing_path(path, runtime_resources_dir, self.config_base_dir, cwd)

    def get_voice(self, voice_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.voices:
            self.load_configs()
        selected = voice_id or self.active_voice_id
        if not selected and self.voices:
            selected = next(iter(self.voices.keys()))
        if selected not in self.voices:
            self.load_configs()
        if selected not in self.voices:
            raise HTTPException(status_code=404, detail=f"Voice '{selected}' not found.")
        return self.voices[selected]

    def default_reference_id(self, voice_config: Dict[str, Any]) -> str:
        default_id = first_value(voice_config, "default_reference_id", "defaultReferenceId", "DefaultReferenceId")
        if default_id:
            return str(default_id)
        refs = first_value(voice_config, "references", "References")
        if isinstance(refs, dict) and refs:
            return "default" if "default" in refs else next(iter(refs.keys()))
        if isinstance(refs, list) and refs:
            first = refs[0]
            return str(first_value(first, "reference_id", "referenceId", "id", "Id", default="default"))
        return "default"

    def reference_exists(self, voice_config: Dict[str, Any], reference_id: str) -> bool:
        refs = first_value(voice_config, "references", "References")
        if isinstance(refs, dict) and refs:
            return reference_id in refs
        if isinstance(refs, list) and refs:
            return any(
                str(first_value(item, "reference_id", "referenceId", "id", "Id", default="")) == reference_id
                for item in refs
                if isinstance(item, dict)
            )
        return reference_id == "default"

    def switch(self, voice_id: str, reference_id: Optional[str] = None, strict: bool = True) -> None:
        voice = self.get_voice(voice_id)
        self.active_voice_id = voice_id
        selected_reference = reference_id or self.default_reference_id(voice)
        if strict and not self.reference_exists(voice, selected_reference):
            raise HTTPException(status_code=404, detail=f"Reference '{selected_reference}' not found for voice '{voice_id}'.")
        self.active_reference_id = selected_reference

    def resolve_reference(self, voice_config: Dict[str, Any], reference_id: Optional[str], strict: bool = True) -> Dict[str, Any]:
        refs = first_value(voice_config, "references", "References")
        selected_id = reference_id or self.active_reference_id or self.default_reference_id(voice_config)

        ref: Dict[str, Any] = {}
        if isinstance(refs, dict) and refs:
            if selected_id not in refs:
                if strict:
                    raise HTTPException(status_code=404, detail=f"Reference '{selected_id}' not found.")
                selected_id = "default" if "default" in refs else next(iter(refs.keys()))
            ref = refs[selected_id]
        elif isinstance(refs, list) and refs:
            ref = next(
                (
                    item
                    for item in refs
                    if str(first_value(item, "reference_id", "referenceId", "id", "Id", default="")) == selected_id
                ),
                None,
            )
            if ref is None:
                if strict:
                    raise HTTPException(status_code=404, detail=f"Reference '{selected_id}' not found.")
                ref = refs[0]
                selected_id = str(first_value(ref, "reference_id", "referenceId", "id", "Id", default="default"))

        cache_path = first_value(ref, "cache_path", "cachePath", "CachePath") or first_value(
            voice_config, "cache_path", "cachePath", "CachePath"
        )

        return {
            "reference_id": selected_id or "default",
            "name": first_value(ref, "name", "Name", default=selected_id or "default"),
            "cache_path": cache_path,
        }


model_manager = ModelManager(args.device, args.device_name, args.gpt_device, args.cpu_threads, args.pipeline)
voice_manager = VoiceManager(args.voices_config, autoload=False)
class SerializedTtsOperations:
    """Keep model mutation exclusive through the final streaming response byte.

    Pure ASGI middleware deliberately holds the lock beyond endpoint return.
    Status remains a read-only snapshot and never wakes or creates an engine.
    """
    def __init__(self, app):
        self.app = app
        self.lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") == "POST":
            async with self.lock:
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


app = FastAPI(title="GPT-SoVITS Interactive PyTorch API")
app.add_middleware(SerializedTtsOperations)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 401:
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"success": False, "message": str(exc), "type": type(exc).__name__})


class SpeechRequest(BaseModel):
    input: str
    voice: str = "default"
    model: str = "gpt-sovits-v2"
    response_format: str = "wav"
    speed: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    text_lang: str = "auto"
    pause_length: Optional[float] = None
    noise_scale: Optional[float] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    ref_lang: Optional[str] = None


class TtsRequest(BaseModel):
    text: str = ""
    modelId: Optional[str] = None
    avatarId: Optional[str] = None
    referenceId: Optional[str] = None
    speed: Optional[float] = None
    noiseScale: Optional[float] = None
    temperature: Optional[float] = None
    topK: Optional[int] = None
    topP: Optional[float] = None
    g2pPriorityMode: Optional[int] = None
    generateTimeline: Optional[bool] = None
    languages: Optional[List[str]] = None
    textLang: Optional[str] = None
    chunkLength: Optional[int] = None

    @property
    def effective_model_id(self) -> Optional[str]:
        return self.modelId or self.avatarId


class SwitchModelRequest(BaseModel):
    modelId: Optional[str] = None
    avatarId: Optional[str] = None
    referenceId: Optional[str] = None

    @property
    def effective_model_id(self) -> Optional[str]:
        return self.modelId or self.avatarId


class BenchmarkRequest(BaseModel):
    modelId: Optional[str] = None
    avatarId: Optional[str] = None
    referenceId: Optional[str] = None

    @property
    def effective_model_id(self) -> Optional[str]:
        return self.modelId or self.avatarId


def resolve_runtime(model_id: Optional[str], reference_id: Optional[str]) -> Dict[str, Any]:
    if model_id:
        voice_manager.switch(model_id, reference_id, strict=True)
    elif reference_id:
        voice_manager.switch(voice_manager.active_voice_id, reference_id, strict=True)

    voice_id = voice_manager.active_voice_id
    voice_config = voice_manager.get_voice(voice_id)
    ref = voice_manager.resolve_reference(voice_config, reference_id, strict=True)
    voice_manager.active_reference_id = ref["reference_id"]

    gpt_path = first_value(voice_config, "gpt_path", "gptPath", "GptPath", "GPTPath")
    sovits_path = first_value(voice_config, "sovits_path", "sovitsPath", "SoVITSPath", "SovitsPath")
    if not gpt_path:
        raise HTTPException(status_code=400, detail="gpt_path is missing in voice config.")
    if not sovits_path:
        raise HTTPException(status_code=400, detail="sovits_path is missing in voice config.")
    cache_path = ref.get("cache_path")
    if not cache_path:
        raise HTTPException(
            status_code=400,
            detail=(
                f"cache_path is missing for reference '{ref['reference_id']}' in voice "
                f"'{voice_id}'. This distribution contains prebuilt reference caches; "
                "prepare a new cache with the resource-preparation workflow before adding a new reference."
            ),
        )

    return {
        "voice_id": voice_id,
        "voice_config": voice_config,
        "gpt_path": voice_manager.resolve_file(gpt_path),
        "sovits_path": voice_manager.resolve_file(sovits_path),
        "cache_path": voice_manager.resolve_file(cache_path),
        "reference_id": ref["reference_id"],
    }


def validate_runtime_files(runtime: Dict[str, Any]) -> None:
    for key in ("gpt_path", "sovits_path", "cache_path"):
        if not os.path.isfile(runtime[key]):
            raise FileNotFoundError(runtime[key])


def resolve_synthesis_options(
    voice_config: Dict[str, Any],
    speed: Optional[float] = None,
    noise_scale: Optional[float] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    pause_length: Optional[float] = None,
    text_lang: Optional[str] = None,
) -> Dict[str, Any]:
    defaults = first_value(voice_config, "defaults", "Defaults", default={}) or {}
    return {
        "speed": speed if speed is not None else first_value(defaults, "speed", "Speed", default=1.0),
        "noise_scale": noise_scale
        if noise_scale is not None
        else first_value(defaults, "noise_scale", "noiseScale", "NoiseScale", default=0.35),
        "temperature": temperature
        if temperature is not None
        else first_value(defaults, "temperature", "Temperature", default=1.0),
        "top_k": top_k if top_k is not None else first_value(defaults, "top_k", "topK", "TopK", default=15),
        "top_p": top_p if top_p is not None else first_value(defaults, "top_p", "topP", "TopP", default=1.0),
        "pause_length": pause_length
        if pause_length is not None
        else first_value(defaults, "pause_length", "pauseLength", "PauseLength", default=0.3),
        "text_lang": text_lang or first_value(defaults, "text_lang", "textLang", "TextLang", default="auto"),
    }


def synthesize_runtime(runtime: Dict[str, Any], text: str, opts: Dict[str, Any], return_timeline: bool):
    engine = model_manager.get_engine(
        runtime["gpt_path"], runtime["sovits_path"], runtime["cache_path"]
    )
    return engine.synthesize(
        text=text,
        text_lang=opts["text_lang"],
        top_k=opts["top_k"],
        top_p=opts["top_p"],
        temperature=opts["temperature"],
        noise_scale=opts["noise_scale"],
        speed=opts["speed"],
        pause_length=opts["pause_length"],
        return_timeline=return_timeline,
    ), engine


@app.post("/v1/audio/speech")
async def text_to_speech(request: SpeechRequest, raw_request: Request):
    authorize_request(raw_request)
    runtime = resolve_runtime(request.voice if request.voice != "default" else None, None)
    opts = resolve_synthesis_options(
        runtime["voice_config"],
        speed=request.speed,
        noise_scale=request.noise_scale,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        pause_length=request.pause_length,
        text_lang=request.text_lang,
    )
    (audio, _), engine = synthesize_runtime(runtime, request.input, opts, return_timeline=False)
    return Response(content=audio_array_to_wav_bytes(audio, engine.sample_rate), media_type="audio/wav")


@app.get("/v1/models")
async def list_openai_models(request: Request):
    authorize_request(request)
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "gpt-sovits-torch",
                "description": config.get("description", ""),
            }
            for name, config in voice_manager.voices.items()
        ],
    }


@app.get("/v1/voices")
async def list_voices(request: Request):
    authorize_request(request)
    return voice_manager.voices


@app.post("/v1/voices/reload")
async def reload_voices(request: Request):
    authorize_request(request)
    voice_manager.load_configs()
    return {"status": "success", "count": len(voice_manager.voices)}


@app.get("/api/tts/status")
async def get_status(request: Request, token: Optional[str] = None):
    authorize_request(request, token)
    return runtime_status()


def runtime_status():
    engine = model_manager.active_engine
    semantic = getattr(engine, "t2s_model", None) if engine is not None else None
    error = engine.lifecycle_error if engine is not None else None
    hibernated = engine is not None and engine.is_hibernated
    state = "error" if error else ("unloaded" if engine is None else ("hibernated" if hibernated else "ready"))
    return {
        "status": "online",
        "runtime_state": state,
        "hibernated": hibernated,
        "ready": state == "ready",
        "lifecycle_error": error,
        "active_model_id": voice_manager.active_voice_id,
        "active_reference_id": voice_manager.active_reference_id,
        "sample_rate": int(engine.sample_rate) if engine is not None else 0,
        "model_count": len(voice_manager.voices),
        "startup_metrics": engine.startup_metrics if engine is not None else {},
        "gpt_backend": getattr(semantic, "kind", None),
        "gpt_adapter": getattr(semantic, "adapter", None),
        # Allocator counters exclude CUDA context and driver/library resources.
        "cuda_allocated_mb": torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_initialized() else 0,
        "cuda_reserved_mb": torch.cuda.memory_reserved() / 2**20 if torch.cuda.is_initialized() else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/tts/hibernate")
async def hibernate_tts(request: Request):
    authorize_request(request)
    model_manager.hibernate()
    return runtime_status()


@app.post("/api/tts/wake")
async def wake_tts(raw_request: Request, request_data: Optional[SwitchModelRequest] = None):
    authorize_request(raw_request)
    request_data = request_data or SwitchModelRequest()
    runtime = resolve_runtime(request_data.effective_model_id, request_data.referenceId)
    validate_runtime_files(runtime)
    model_manager.get_engine(runtime["gpt_path"], runtime["sovits_path"], runtime["cache_path"])
    return runtime_status()


@app.get("/api/tts/models")
async def get_models(request: Request, token: Optional[str] = None):
    authorize_request(request, token)
    result = []
    for voice_id, config in voice_manager.voices.items():
        refs = first_value(config, "references", "References")
        if isinstance(refs, dict):
            references = [
                {
                    "reference_id": ref_id,
                    "name": first_value(ref, "name", "Name", default=ref_id),
                }
                for ref_id, ref in refs.items()
            ]
        elif isinstance(refs, list):
            references = [
                {
                    "reference_id": str(first_value(ref, "reference_id", "referenceId", "id", "Id", default="default")),
                    "name": first_value(ref, "name", "Name", default=None)
                    or str(first_value(ref, "reference_id", "referenceId", "id", "Id", default="default")),
                }
                for ref in refs
            ]
        else:
            references = [{"reference_id": "default", "name": "default"}]

        result.append(
            {
                "model_id": voice_id,
                "name": first_value(config, "name", "Name", "description", "Description", default=voice_id),
                "description": first_value(config, "description", "Description", default=""),
                "is_active": voice_id == voice_manager.active_voice_id,
                "default_reference_id": voice_manager.default_reference_id(config),
                "references": references,
            }
        )
    return result


@app.post("/api/tts/switch-model")
async def switch_model(request_data: SwitchModelRequest, raw_request: Request):
    authorize_request(raw_request)
    model_id = request_data.effective_model_id
    if not model_id:
        raise HTTPException(status_code=400, detail="modelId is required")
    voice_manager.switch(model_id, request_data.referenceId, strict=True)
    runtime = resolve_runtime(model_id, request_data.referenceId)
    validate_runtime_files(runtime)
    engine = model_manager.get_engine(
        runtime["gpt_path"], runtime["sovits_path"], runtime["cache_path"]
    )
    return {
        "success": True,
        "active_model_id": voice_manager.active_voice_id,
        "active_reference_id": voice_manager.active_reference_id,
        "sample_rate": int(engine.sample_rate),
    }


@app.post("/api/tts/benchmark")
async def benchmark_tts(raw_request: Request, request_data: Optional[BenchmarkRequest] = None):
    authorize_request(raw_request)
    request_data = request_data or BenchmarkRequest()
    runtime = resolve_runtime(request_data.effective_model_id, request_data.referenceId)
    validate_runtime_files(runtime)
    engine = model_manager.get_engine(
        runtime["gpt_path"], runtime["sovits_path"], runtime["cache_path"]
    )
    result = engine.benchmark()
    return {
        "success": True,
        "active_model_id": voice_manager.active_voice_id,
        "active_reference_id": voice_manager.active_reference_id,
        "device": model_manager.device,
        "gpt_device": model_manager.gpt_device,
        "cpu_threads": model_manager.cpu_threads,
        "pipeline": model_manager.pipeline,
        "benchmark": result,
    }


@app.post("/api/tts/predict")
async def predict(request_data: TtsRequest, raw_request: Request):
    authorize_request(raw_request)
    if not request_data.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    runtime = resolve_runtime(request_data.effective_model_id, request_data.referenceId)
    opts = resolve_synthesis_options(
        runtime["voice_config"],
        speed=request_data.speed,
        noise_scale=request_data.noiseScale,
        temperature=request_data.temperature,
        top_k=request_data.topK,
        top_p=request_data.topP,
        text_lang=request_data.textLang,
    )
    (audio, _), engine = synthesize_runtime(runtime, request_data.text, opts, return_timeline=False)
    return Response(content=audio_array_to_wav_bytes(audio, engine.sample_rate), media_type="audio/wav")


@app.post("/api/tts/predict-timeline")
async def predict_timeline(request_data: TtsRequest, raw_request: Request):
    authorize_request(raw_request)
    if not request_data.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    runtime = resolve_runtime(request_data.effective_model_id, request_data.referenceId)
    opts = resolve_synthesis_options(
        runtime["voice_config"],
        speed=request_data.speed,
        noise_scale=request_data.noiseScale,
        temperature=request_data.temperature,
        top_k=request_data.topK,
        top_p=request_data.topP,
        text_lang=request_data.textLang,
    )
    (audio, timestamps), engine = synthesize_runtime(runtime, request_data.text, opts, return_timeline=True)
    sr = int(engine.sample_rate)
    wav_bytes = audio_array_to_wav_bytes(audio, sr)
    return {
        "audio": base64.b64encode(wav_bytes).decode("ascii"),
        "media_type": "wav",
        "sample_rate": sr,
        "duration": len(audio) / float(sr) if sr > 0 else 0.0,
        "active_model_id": voice_manager.active_voice_id,
        "active_reference_id": voice_manager.active_reference_id,
        "timestamps": timestamps,
    }


@app.post("/api/tts/predict-stream")
async def predict_stream(request_data: TtsRequest, raw_request: Request):
    """Stream committed PCM with sample-based approximate character timing."""
    authorize_request(raw_request)
    if not request_data.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    runtime = resolve_runtime(request_data.effective_model_id, request_data.referenceId)
    opts = resolve_synthesis_options(
        runtime["voice_config"],
        speed=request_data.speed,
        noise_scale=request_data.noiseScale,
        temperature=request_data.temperature,
        top_k=request_data.topK,
        top_p=request_data.topP,
        text_lang=request_data.textLang,
    )
    engine = model_manager.get_engine(
        runtime["gpt_path"], runtime["sovits_path"], runtime["cache_path"]
    )
    sample_rate = int(engine.sample_rate)
    chunk_length = max(10, min(100, int(request_data.chunkLength or 24)))

    def event_stream():
        stream_started_at = time.perf_counter()
        first_chunk_latency = 0.0
        chunk_count = 0
        sample_offset = 0
        yield json.dumps(
            {
                "type": "start",
                "sample_rate": sample_rate,
                "channels": 1,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        stream = None
        try:
            from streaming_latents import LatentStream
            stream = LatentStream(engine,token_chunk=chunk_length).stream(
                text=request_data.text,
                language=opts["text_lang"],
                top_k=opts["top_k"],
                top_p=opts["top_p"],
                temperature=opts["temperature"],
                noise_scale=opts["noise_scale"],
                speed=opts["speed"],
                pause=opts["pause_length"],
            )
            for event in stream:
                if event['type'] != 'chunk':
                    continue
                pcm = np.asarray(event['pcm'], dtype="<f4")
                if pcm.size == 0:
                    continue
                chunk_count += 1
                if first_chunk_latency <= 0.0:
                    first_chunk_latency = time.perf_counter() - stream_started_at
                start = sample_offset
                sample_offset += int(pcm.size)
                yield json.dumps(
                    {
                        "type": "chunk",
                        "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
                        "audio_start_sample": start,
                        "audio_end_sample": sample_offset,
                        "sample_rate": sample_rate,
                        "channels": 1,
                        "is_final": False,
                        "chunk_index": chunk_count,
                        "timestamps": event["timestamps"],
                        "gpt_finished": event.get("gpt_finished", True),
                        "server_first_chunk_latency_seconds": first_chunk_latency,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n"
            yield json.dumps(
                {
                    "type": "done",
                    "sample_rate": sample_rate,
                    "channels": 1,
                    "audio_end_sample": sample_offset,
                    "is_final": True,
                    "chunk_count": chunk_count,
                    "server_first_chunk_latency_seconds": first_chunk_latency,
                    "server_total_seconds": time.perf_counter() - stream_started_at,
                    "audio_duration_seconds": sample_offset / float(sample_rate),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
        except Exception as exc:
            logger.exception("Streaming inference failed")
            yield json.dumps(
                {"type": "error", "error": str(exc)},
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"

        finally:
            if stream is not None:
                stream.close()

    async def response_events():
        iterator = event_stream()
        try:
            while True:
                item = await anyio.to_thread.run_sync(lambda: next(iterator, None))
                if item is None:
                    break
                yield item
        finally:
            # Finish the current decode and restore model precision before the
            # operation lock admits a voice switch or another inference request.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(iterator.close)

    return StreamingResponse(response_events(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    final_parser = argparse.ArgumentParser(description="GPT-SoVITS Interactive PyTorch API Server")
    final_parser.add_argument("--host", default="127.0.0.1")
    final_parser.add_argument("--port", type=int, default=8001)
    final_parser.add_argument("--device", default=args.device, choices=DEVICE_CHOICES)
    final_parser.add_argument("--device_name", default=args.device_name)
    final_parser.add_argument("--gpt_device", default=args.gpt_device, choices=DEVICE_CHOICES)
    final_parser.add_argument("--cpu_threads", type=int, default=args.cpu_threads)
    final_parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=args.pipeline)
    final_parser.add_argument("--voices_config", default=args.voices_config)
    final_parser.add_argument("--token", default=args.token)
    final_parser.add_argument("--avatar", default=args.avatar)
    final_parser.add_argument("--reference", default=args.reference)
    final_parser.add_argument("--resources", default=args.resources)

    try:
        final_args = final_parser.parse_args()
    except SystemExit as ex:
        if ex.code == 0:
            raise
        write_startup_status(startup_payload(TtsStartupStatus.INVALID_ARGUMENTS, detail="Invalid command line arguments."))
        raise SystemExit(int(TtsStartupStatus.INVALID_ARGUMENTS))

    final_args.host = "127.0.0.1"
    final_args.resources = resolve_path(final_args.resources) if final_args.resources else default_resources_dir()

    if not final_args.token:
        fail_startup(
            TtsStartupStatus.TOKEN_REQUIRED,
            port=final_args.port,
            avatar_id=final_args.avatar or None,
            reference_id=final_args.reference or None,
        )
    if not final_args.avatar:
        fail_startup(TtsStartupStatus.AVATAR_REQUIRED, port=final_args.port, reference_id=final_args.reference or None)
    if not is_port_available(final_args.host, final_args.port):
        fail_startup(
            TtsStartupStatus.PORT_UNAVAILABLE,
            port=final_args.port,
            avatar_id=final_args.avatar,
            reference_id=final_args.reference or None,
        )

    explicit_voices_config = argument_present("--voices_config")
    selected_voices_config = choose_voices_config_path(
        final_args.voices_config,
        final_args.resources,
        explicit_voices_config,
    )
    configure_resource_environment(final_args.resources)
    logger.info("Runtime resources: %s", final_args.resources)
    logger.info("Runtime NLTK data: %s", os.environ.get("NLTK_DATA", ""))

    args.device = final_args.device
    args.device_name = final_args.device_name
    args.gpt_device = final_args.gpt_device
    args.cpu_threads = final_args.cpu_threads
    args.pipeline = final_args.pipeline
    args.token = final_args.token
    model_manager.device = final_args.device
    model_manager.device_name = final_args.device_name
    model_manager.gpt_device = final_args.gpt_device
    model_manager.cpu_threads = final_args.cpu_threads
    model_manager.pipeline = final_args.pipeline
    runtime_resources_dir = final_args.resources
    voice_manager.configure(selected_voices_config)
    voices_config_base_dir = voice_manager.config_base_dir

    try:
        voice_manager.load_configs(required=True)
    except Exception as ex:
        fail_startup(
            TtsStartupStatus.CONFIG_LOAD_FAILED,
            detail=str(ex),
            port=final_args.port,
            avatar_id=final_args.avatar,
            reference_id=final_args.reference or None,
        )

    try:
        voice_manager.switch(final_args.avatar, final_args.reference or None, strict=True)
    except HTTPException as ex:
        detail = str(ex.detail)
        status = TtsStartupStatus.REFERENCE_NOT_FOUND if "Reference" in detail else TtsStartupStatus.AVATAR_NOT_FOUND
        fail_startup(
            status,
            detail=detail,
            port=final_args.port,
            avatar_id=final_args.avatar,
            reference_id=final_args.reference or None,
        )

    try:
        active_runtime = resolve_runtime(None, None)
        validate_runtime_files(active_runtime)
        active_engine = model_manager.get_engine(
            active_runtime["gpt_path"], active_runtime["sovits_path"], active_runtime.get("cache_path")
        )
        logger.info("Active model sample rate: %s", active_engine.sample_rate)
    except FileNotFoundError as ex:
        fail_startup(
            TtsStartupStatus.RESOURCE_MISSING,
            detail=str(ex),
            port=final_args.port,
            avatar_id=voice_manager.active_voice_id,
            reference_id=voice_manager.active_reference_id,
        )
    except Exception as ex:
        fail_startup(
            TtsStartupStatus.MODEL_INIT_FAILED,
            detail=str(ex),
            port=final_args.port,
            avatar_id=voice_manager.active_voice_id,
            reference_id=voice_manager.active_reference_id,
        )

    ready_payload = startup_payload(
        TtsStartupStatus.OK,
        port=final_args.port,
        avatar_id=voice_manager.active_voice_id,
        reference_id=voice_manager.active_reference_id,
    )

    class StartupReportingServer(uvicorn.Server):
        def __init__(self, config, payload):
            super().__init__(config)
            self.payload = payload
            self.reported = False

        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if not self.should_exit and not self.reported:
                write_startup_status(self.payload)
                self.reported = True

    config = uvicorn.Config(app, host=final_args.host, port=final_args.port, log_level="info")
    server = StartupReportingServer(config, ready_payload)
    try:
        server.run()
    except SystemExit:
        raise
    except OSError as ex:
        fail_startup(
            TtsStartupStatus.PORT_UNAVAILABLE,
            detail=str(ex),
            port=final_args.port,
            avatar_id=voice_manager.active_voice_id,
            reference_id=voice_manager.active_reference_id,
        )
