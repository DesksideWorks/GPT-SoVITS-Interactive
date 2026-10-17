# Third-Party TTS Engine Notice

This notice lists the main upstream projects used by **GPT-SoVITS Interactive**.

本文件列出 **GPT-SoVITS Interactive** 所使用的主要上游项目。

## GPT-SoVITS

- Project: GPT-SoVITS
- Upstream: https://github.com/RVC-Boss/GPT-SoVITS
- Role in this runtime: model architecture, GPT-SoVITS synthesis method, training framework, and original PyTorch inference foundation.
- License: MIT; full text: `GPT-SoVITS_LICENSE.txt`.

GPT-SoVITS provides the foundation of the voice synthesis method used by the bundled runtime.

## GPT-SoVITS_minimal_inference

- Project: GPT-SoVITS_minimal_inference
- Upstream: https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference
- Role in this runtime: direct upstream of this local inference runtime.
- License: Apache License 2.0; full text: `GPT-SoVITS_minimal_inference_LICENSE.txt`.

This runtime is based on GPT-SoVITS_minimal_inference and includes modifications by **桌边工房 / DesksideWorks / デスクサイド工房**.

The modifications include:

- a local HTTP API for applications;
- packaging as a self-contained local runtime;
- backend selection and startup/status handling;
- DirectML compatibility work for non-NVIDIA Windows GPUs;
- ORT DirectML semantic inference, GPU adapter matching and session lifecycle;
- prompt-cache workflow so reference-side feature extraction can be prepared ahead of runtime;
- streaming synthesis and character-level timeline output for character display and synchronization;
- per-language semantic references for the shared GPT;
- removal of upstream training code and other modules the runtime does not load.

The bundled runtime does not claim ownership of GPT-SoVITS or GPT-SoVITS_minimal_inference, and it does not imply endorsement by the upstream projects.

本运行环境基于 GPT-SoVITS_minimal_inference，并包含 **桌边工房 / DesksideWorks / デスクサイド工房** 进行的修改。

这些修改包括：

- 面向应用的本地 HTTP API；
- 打包为自带依赖的本地运行环境；
- 后端选择与启动/状态处理；
- 面向非 NVIDIA Windows GPU 的 DirectML 兼容性处理；
- ORT DirectML 语义推理、GPU 设备匹配与会话生命周期管理；
- prompt-cache 工作流，使参考侧特征提取可以在运行前准备；
- 流式合成，以及用于角色显示和同步的逐字时间线输出；
- 为共享 GPT 提供的各语言语义参考；
- 删去上游训练代码及其他运行时不加载的模块。

同梱运行环境不主张拥有 GPT-SoVITS 或 GPT-SoVITS_minimal_inference，也不暗示获得上游项目认可。

## Other Runtime Dependencies / 其他运行依赖

CUDA semantic inference uses PyTorch. DirectML semantic inference uses ONNX Runtime DirectML; acoustic decoding uses torch-directml. The CPU ONNX Runtime provider is also used by the Chinese frontend. ONNX Runtime's LICENSE and ThirdPartyNotices.txt, including bundled native dependencies, are retained in each applicable `Runtime/Python_*/Lib/site-packages/onnxruntime/` directory and included in `PythonThirdPartyLicenseTexts.txt`.

CUDA 语义推理使用 PyTorch；DirectML 语义推理使用 ONNX Runtime DirectML，声学解码使用 torch-directml。中文前端还使用 ORT 的 CPU provider。ORT 与所附原生组件的许可原文保留在对应 Python 环境的 onnxruntime 目录，并纳入 `PythonThirdPartyLicenseTexts.txt`。

Model and dictionary provenance and licenses: `TTS_MODEL_AND_DATA_NOTICE.md`.
模型与词典的来源和许可另见该文件。

The Chinese text frontend is restored from GPT-SoVITS_minimal_inference and uses
Chinese RoBERTa, G2PW, ONNX Runtime (CPU), pypinyin, cn2an, jieba/jieba_fast,
OpenCC, fastText/fast-langdetect, and split-lang with their runtime dependencies.
The source retains the upstream copyright and license headers. The bundled
model origins and file hashes are recorded in `chinese-runtime-manifest.json`.

中文文本前端从 GPT-SoVITS_minimal_inference 恢复，使用 Chinese RoBERTa、G2PW、
ONNX Runtime（CPU）、pypinyin、cn2an、jieba/jieba_fast、OpenCC、
fastText/fast-langdetect 和 split-lang 及其运行依赖。源码保留上游版权及许可头，
同梱模型来源和文件哈希见 `chinese-runtime-manifest.json`。

The embedded Python runtime may include additional third-party packages such as PyTorch, torch-directml, FastAPI, Uvicorn, NLTK, pyopenjtalk, and other Python packages required for local inference.

Their license texts are listed in `PythonThirdPartyNotices.txt` and `PythonThirdPartyLicenseTexts.txt`.

嵌入式 Python 运行环境可能包含 PyTorch、torch-directml、FastAPI、Uvicorn、NLTK、pyopenjtalk 以及本地推理所需的其他 Python 包。

其许可证文本见 `PythonThirdPartyNotices.txt` 和 `PythonThirdPartyLicenseTexts.txt`。

## Character Voice Data / 角色声音数据

This file covers TTS engine and runtime code acknowledgments only.

Character voice data, voice model data, prompt caches, generated audio, character names, and character identities are covered by separate character voice notices, such as:

```text
resources/avatars/ami/AMITARO_VOICE_NOTICE.md
resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md
```

本文件仅说明 TTS 引擎和运行时代码相关的上游项目致谢。

角色声音数据、语音模型数据、prompt cache、生成音频、角色名称和角色身份等内容，由单独的角色声音说明文件覆盖，例如：

```text
resources/avatars/ami/AMITARO_VOICE_NOTICE.md
resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md
```
