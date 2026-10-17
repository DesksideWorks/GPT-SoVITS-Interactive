# GPT-SoVITS Interactive Runtime Notice

This notice applies to **GPT-SoVITS Interactive**: the source code in this repository and the Windows runtime packages released with it.

本说明适用于 **GPT-SoVITS Interactive**：本仓库中的源码，以及随之发布的 Windows 运行包。

## 1. What This Is / 定位

GPT-SoVITS Interactive is a local GPT-SoVITS runtime for games, desktop companions and other applications. It provides a local HTTP API, streaming synthesis and character-level timelines.

It was developed for and is used in **DesktopDaily / デスクトップ日常 / 桌面日常**, and is released as a standalone runtime. The runtime packages include voice synthesis data for two characters, so the runtime can generate speech right after extraction.

GPT-SoVITS Interactive 是面向游戏、桌宠和其他应用的本地 GPT-SoVITS 运行环境，提供本地 HTTP API、流式合成和逐字时间线。

它最初为 **DesktopDaily / デスクトップ日常 / 桌面日常** 开发并在其中使用，现作为独立运行环境发布。运行包随附两位角色的音声合成数据，解压后即可生成语音。

## 2. TTS Engine / TTS 引擎

The TTS engine used by this runtime is based on **GPT-SoVITS** and a modified local inference runtime based on **GPT-SoVITS_minimal_inference**.

- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- GPT-SoVITS_minimal_inference: https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference

This runtime includes modifications and integration work by **桌边工房 / DesksideWorks / デスクサイド工房**, such as the local API, runtime packaging, backend selection, DirectML compatibility work, the prompt-cache workflow, streaming synthesis and timeline output.

The model architecture, the original GPT-SoVITS synthesis method, and the upstream inference foundation come from the upstream projects listed above.

本运行环境使用的 TTS 引擎基于 **GPT-SoVITS**，并基于 **GPT-SoVITS_minimal_inference** 修改得到本地推理运行环境。

- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- GPT-SoVITS_minimal_inference: https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference

本运行环境包含 **桌边工房 / DesksideWorks / デスクサイド工房** 进行的修改和整合工作，例如本地 API、运行环境打包、后端选择、DirectML 兼容性处理、prompt cache 工作流、流式合成和时间线输出。

模型架构、原始 GPT-SoVITS 合成方法以及上游推理基础来自上述上游项目。

## 3. Code Licenses / 代码许可

The TTS engine/runtime code and the voice synthesis data are separate.

Original code added to this repository by DesksideWorks is released under the Apache License, Version 2.0, available in the repository's `LICENSE` file. Code copied or derived from GPT-SoVITS, GPT-SoVITS_minimal_inference, or other third-party projects remains under the applicable original license listed in the repository notices.

The code derived from GPT-SoVITS_minimal_inference is provided under the license of GPT-SoVITS_minimal_inference. The upstream GPT-SoVITS project is provided under its own license. The corresponding license texts are `GPT-SoVITS_minimal_inference_LICENSE.txt` and `GPT-SoVITS_LICENSE.txt`; other third-party license texts are under `Licenses/` and in `PythonThirdPartyLicenseTexts.txt`.

Voice model data, character voice data, prompt caches, character names, voice identities, generated audio, and other resources derived from third-party character or voice materials are **not** covered by the TTS engine code license.

TTS 引擎/运行时代码与音声合成数据是不同层面的内容。

桌边工房为本仓库新增的原创代码以 Apache License 2.0 发布，完整文本见仓库的 `LICENSE` 文件。复制或派生自 GPT-SoVITS、GPT-SoVITS_minimal_inference 或其他第三方项目的代码，继续适用仓库声明中列出的对应原始许可证。

来自 GPT-SoVITS_minimal_inference 的代码适用 GPT-SoVITS_minimal_inference 的许可证。上游 GPT-SoVITS 项目适用其自身许可证。相应许可证文本为 `GPT-SoVITS_minimal_inference_LICENSE.txt` 和 `GPT-SoVITS_LICENSE.txt`；其他第三方许可证文本见 `Licenses/` 和 `PythonThirdPartyLicenseTexts.txt`。

语音模型数据、角色声音数据、prompt cache、角色名称、声音身份、生成音频，以及其他由第三方角色或声音素材派生的资源，**不属于** TTS 引擎代码许可证的授权范围。

## 4. Character Voice Data / 角色声音数据

The runtime packages include voice synthesis data for 小春音アミ and つくよみちゃん.

Each character voice has its own credit and notice file. The applicable voice-source terms, character terms, attribution requirements, and usage restrictions are described in those files:

```text
resources/avatars/ami/AMITARO_VOICE_NOTICE.md
resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md
```

The included character voice data is not official voice synthesis data created, distributed, or endorsed by the original character or voice-material providers unless explicitly stated otherwise in the corresponding character notice.

运行包随附小春音アミ与つくよみちゃん的音声合成数据。

每位角色的声音都有各自的署名与说明文件。适用的声音来源规约、角色规约、署名要求和使用限制，以对应文件为准：

```text
resources/avatars/ami/AMITARO_VOICE_NOTICE.md
resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md
```

除非对应角色说明文件中另有明确说明，随附的角色音声合成数据并非由原角色或声音素材提供方制作、分发或认可的官方音声合成数据。

## 5. Bundled Runtime Data / 随附运行数据

The runtime packages include model files, runtime configuration files, precomputed reference caches, semantic references, and other files required for local inference.

These files are provided so that the runtime works with its two bundled characters. They are not provided as a voice-material collection, sound library, training dataset, or third-party AI voice service.

This repository and the runtime packages do not provide original third-party voice recordings or reference-audio files as independently reusable audio materials.

运行包包含本地推理所需的模型文件、运行配置文件、预计算的参考缓存、语义参考，以及其他运行所需文件。

这些文件的用途是让运行环境可以直接使用两位随附角色，并非作为声音素材集、声音素材库、训练数据集或第三方 AI 语音服务提供。

本仓库和运行包不将第三方原始声音录音或参考音频文件作为可独立复用的音频素材提供。

## 6. No Standalone Redistribution / 不作为独立声音模型再分发

The restrictions in this section concern the bundled character voice synthesis data. They do not restrict rights granted by the original licenses of third-party software or independently licensed models and dictionaries. See `TTS_MODEL_AND_DATA_NOTICE.md` for those components.

本节限制针对随附的角色音声合成数据，不限制第三方软件、独立授权模型及词典的原始许可证所授予的权利。相关组件见 `TTS_MODEL_AND_DATA_NOTICE.md`。

本節の制限は同梱のキャラクター音声合成データを対象とし、第三者ソフトウェアや個別に許諾されたモデル・辞書の原ライセンスが認める権利を制限しません。

The bundled character voice synthesis data is provided for use with this runtime.

Do not redistribute, resell, mirror, re-upload, repack, or publish the bundled voice synthesis data, prompt caches, character voice data, or related runtime data separately from this runtime as a standalone voice model, voice pack, sound library, training dataset, voice-material collection, or third-party AI voice service.

Using these files with other engines or tools, separately from this runtime, is outside the intended scope of this release. DesksideWorks does not provide support, warranty, or separate permission for such use. DesksideWorks grants no permission for the character voice data beyond the official terms of the character and voice-material providers. Any use remains subject to the corresponding upstream code licenses, character voice notices, original material-provider terms, platform rules, and applicable laws.

随附的角色音声合成数据是为配合本运行环境使用而提供的。

不得将随附的音声合成数据、prompt cache、角色声音数据或相关运行数据脱离本运行环境，作为独立语音模型、voice pack、声音素材库、训练数据集、声音素材集或第三方 AI 语音服务进行再分发、转卖、镜像、再上传、重新打包或公开发布。

脱离本运行环境、将这些文件用于其他引擎或工具，不属于本次发布的预期用途范围。DesksideWorks 不为此类使用提供支持、保证或单独许可，也不在角色及声音素材提供方的官方规约之外对角色声音数据作任何授权。任何使用仍需遵守对应的上游代码许可证、角色声音说明、原素材方规约、平台规则及适用法律。

## 7. No Official Endorsement / 非官方声明

GPT-SoVITS Interactive and its runtime packages are produced by **桌边工房 / DesksideWorks / デスクサイド工房**.

GPT-SoVITS Interactive is not an official project of GPT-SoVITS, GPT-SoVITS_minimal_inference, any character source, any voice-material provider, or any corpus provider, unless explicitly stated otherwise.

GPT-SoVITS Interactive 及其运行包由 **桌边工房 / DesksideWorks / デスクサイド工房** 制作和整理。

除非另有明确说明，GPT-SoVITS Interactive 并非 GPT-SoVITS、GPT-SoVITS_minimal_inference、任何角色来源方、声音素材提供方或语料库提供方的官方项目。

## 8. No Warranty / 免责声明

For the Microsoft C++ runtime libraries bundled beside the TTS Python executables,
see `MSVC_RUNTIME_NOTICE.md` and the original Microsoft license included with it.
TTS Python 可执行文件旁同梱的微软 C++ 运行库，见 `MSVC_RUNTIME_NOTICE.md` 及其中所列原始微软许可。
TTS 用 Python に同梱される Microsoft C++ ランタイムについては、`MSVC_RUNTIME_NOTICE.md` と原文ライセンスをご確認ください。

The runtime and the bundled voice synthesis data are provided as is, without warranty of any kind. Use is at the user's own risk and remains subject to applicable licenses, material-provider terms, platform rules, and applicable laws.

本运行环境及随附的音声合成数据按现状提供，不附带任何保证。使用风险由用户自行承担，并需遵守适用的许可证、素材方规约、平台规则及适用法律。
