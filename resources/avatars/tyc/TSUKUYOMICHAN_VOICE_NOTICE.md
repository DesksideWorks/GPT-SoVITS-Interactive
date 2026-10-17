# Tsukuyomi-chan Voice Notice

This notice applies to the local TTS voice synthesis data for **つくよみちゃん / Tsukuyomi-chan** included in **GPT-SoVITS Interactive**.

本说明适用于 **GPT-SoVITS Interactive** 随附的つくよみちゃん / Tsukuyomi-chan 本地 TTS 音声合成数据。

## 1. Scope of This Notice / 本说明的范围

GPT-SoVITS Interactive is a local TTS runtime for character voice output.

The TTS engine used by the runtime is **GPT-SoVITS**. The GPT-SoVITS engine/code is covered by its own open-source license, which is provided separately with the code or third-party license notices.

This file does **not** license the GPT-SoVITS engine/code. This file only explains the voice source credit, official terms, and usage restrictions for the つくよみちゃん / Tsukuyomi-chan voice synthesis data included in GPT-SoVITS Interactive.

GPT-SoVITS Interactive 是用于角色语音输出的本地 TTS 运行环境。

本运行环境使用的 TTS 引擎为 **GPT-SoVITS**。GPT-SoVITS 引擎/代码适用其自身的开源许可证，该许可证已随代码或第三方开源许可说明另行提供。

本文件不用于授权 GPT-SoVITS 引擎/代码。本文件仅说明 GPT-SoVITS Interactive 随附的つくよみちゃん / Tsukuyomi-chan 音声合成数据所涉及的声音来源署名、官方规约和使用限制。

## 2. Voice Credit / 声音署名

```text
本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

■つくよみちゃんコーパス（CV.夢前黎）
使用音源：Vol.1 声優統計コーパス（JVSコーパス準拠）
https://tyc.rei-yumesaki.net/material/corpus/

■つくよみちゃん公式サイト
https://tyc.rei-yumesaki.net/
```

The つくよみちゃん / Tsukuyomi-chan voice synthesis data included in GPT-SoVITS Interactive uses voice data from **つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV.夢前黎）** as training/source voice data.

GPT-SoVITS Interactive and the included voice synthesis data are produced by **桌边工房 / DesksideWorks / デスクサイド工房**.

In GPT-SoVITS Interactive, the つくよみちゃん synthesis path also uses data from other speakers. This is how GPT-SoVITS Interactive is built, not something that comes from the official つくよみちゃん materials: the shared semantic GPT is trained with 小春音アミ voice material, and the Chinese and English semantic references come from a Qwen3-TTS generated Mandarin sentence and a CMU ARCTIC English recording. The timbre comes from the つくよみちゃん SoVITS model and reference. See `TTS_MODEL_AND_DATA_NOTICE.md`.

GPT-SoVITS Interactive 随附的つくよみちゃん / Tsukuyomi-chan 音声合成数据使用了 **つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV.夢前黎）** 的声音数据作为训练/来源声音数据。

GPT-SoVITS Interactive 及其随附音声合成数据由 **桌边工房 / DesksideWorks / デスクサイド工房** 制作和整理。

在 GPT-SoVITS Interactive 中，つくよみちゃん的合成流程也用到了其他说话人的数据。这是 GPT-SoVITS Interactive 自身的做法，并非来自つくよみちゃん官方素材：共用的语义 GPT 以小春音アミ的声音素材训练；中文和英文的语义参考分别来自 Qwen3-TTS 生成的一句中文和 CMU ARCTIC 的一句英文录音。音色来自つくよみちゃん自己的 SoVITS 模型和参考。详见 `TTS_MODEL_AND_DATA_NOTICE.md`。

## 3. Unofficial Notice / 非官方声明

GPT-SoVITS Interactive is not an official product of Rei Yumesaki or the つくよみちゃん Project.

The bundled つくよみちゃん / Tsukuyomi-chan voice synthesis data is **not** official voice synthesis data created, distributed, or endorsed by Rei Yumesaki or the つくよみちゃん Project.

Generated audio from this data must not be presented as speech created, recorded, published, or endorsed by Rei Yumesaki or the つくよみちゃん Project.

GPT-SoVITS Interactive 并非 夢前黎 或 つくよみちゃんプロジェクト 的官方产品。

同梱的つくよみちゃん / Tsukuyomi-chan 音声合成数据并非由 夢前黎 或 つくよみちゃんプロジェクト 制作、分发或认可的官方音声合成数据。

不得将该数据生成的音频表述为 夢前黎 或 つくよみちゃんプロジェクト 制作、录制、发表或认可的内容。

## 4. Official Terms / 官方规约

Use of this bundled voice synthesis data and audio generated from it is subject to the official terms for つくよみちゃんコーパス and the つくよみちゃん character license.

This notice is a supplementary notice for GPT-SoVITS Interactive. It does not replace the official terms of the つくよみちゃん Project. If there is any difference or conflict between this notice and the official terms, the official terms take precedence.

Official pages:

- つくよみちゃん公式サイト: https://tyc.rei-yumesaki.net/
- つくよみちゃんコーパス: https://tyc.rei-yumesaki.net/material/corpus/
- つくよみちゃんキャラクターライセンス: https://tyc.rei-yumesaki.net/about/terms/
- クレジット表記について: https://tyc.rei-yumesaki.net/about/terms/credit/

使用本同梱音声合成数据及其生成音频时，必须遵守つくよみちゃんコーパス及つくよみちゃんキャラクターライセンス的官方规约。

本说明是 GPT-SoVITS Interactive 随附的补充说明，不取代 つくよみちゃんプロジェクト 的官方规约。如本说明与官方规约存在差异或冲突，以官方规约为准。

官方页面：

- つくよみちゃん公式サイト: https://tyc.rei-yumesaki.net/
- つくよみちゃんコーパス: https://tyc.rei-yumesaki.net/material/corpus/
- つくよみちゃんキャラクターライセンス: https://tyc.rei-yumesaki.net/about/terms/
- クレジット表記について: https://tyc.rei-yumesaki.net/about/terms/credit/

## 5. Bundled Runtime Data / 同梱运行数据

The voice synthesis data included in GPT-SoVITS Interactive is provided as one of the runtime's bundled character voices.

The bundled files may include model data, runtime data, and precomputed or pre-inferred cache data prepared for local TTS inference. These files are not provided as a standalone voice model package, voice material collection, sound library, training dataset, or third-party voice model service.

GPT-SoVITS Interactive does not provide the original つくよみちゃんコーパス recordings, original voice clips, or reference-audio files as independently reusable audio materials.

GPT-SoVITS Interactive 随附的音声合成数据，作为本运行环境的随附角色声音提供。

同梱文件可能包含为本地 TTS 推理准备的模型数据、运行数据，以及预先计算或预先推理得到的缓存数据。这些文件并非作为独立语音模型包、声音素材集、声音素材库、训练数据集或第三方音声模型服务提供。

GPT-SoVITS Interactive 不将つくよみちゃんコーパス的原始录音、原始声音素材或参考音频文件作为可独立复用的音频素材提供。

## 6. Generated Audio and User Works / 生成音频与用户作品

Audio generated by the bundled つくよみちゃん / Tsukuyomi-chan voice synthesis data may be used only within the scope of the official つくよみちゃんコーパス terms, the つくよみちゃん character license, and applicable platform rules and laws.

The official つくよみちゃんコーパス terms allow developers to decide whether users of a voice synthesis software must credit generated audio. For GPT-SoVITS Interactive, audio generated with this voice carries no credit obligation on the つくよみちゃん side; the official terms make an exception for the output of voice synthesis software. In GPT-SoVITS Interactive, the つくよみちゃん synthesis path also uses the semantic GPT trained with 小春音アミ voice material. This is how GPT-SoVITS Interactive is built, not something that comes from the official つくよみちゃん materials. The credit and content rules of Amitaro's Voice Material Studio therefore apply to audio generated with this voice in GPT-SoVITS Interactive as well. See `resources/avatars/ami/AMITARO_VOICE_NOTICE.md`.

If a screenshot, video, livestream, recording, social media post, or other work includes the つくよみちゃん character design, 3D model, visual assets, or other つくよみちゃん materials, the relevant character/material credits and official terms also apply.

使用同梱的つくよみちゃん / Tsukuyomi-chan 音声合成数据生成的音频，仅可在つくよみちゃんコーパス官方规约、つくよみちゃんキャラクターライセンス、相关平台规则及适用法律允许的范围内使用。

つくよみちゃんコーパス官方规约允许音声合成软件开发者自行决定是否要求软件用户对生成音频进行署名。对 GPT-SoVITS Interactive，つくよみちゃん一方不要求为用该声音生成的音频署名（官方规约对音声合成软件的输出音频设有特例）。在 GPT-SoVITS Interactive 中，つくよみちゃん的合成流程也使用以小春音アミ声音素材训练的语义 GPT。这是 GPT-SoVITS Interactive 自身的做法，并非来自つくよみちゃん官方素材。因此 Amitaro's Voice Material Studio 关于署名和内容的规定，同样适用于用 GPT-SoVITS Interactive 生成的つくよみちゃん音频，见 `resources/avatars/ami/AMITARO_VOICE_NOTICE.md`。

如果截图、视频、直播、录像、社交媒体投稿或其他作品中包含つくよみちゃん角色设计、3D模型、视觉素材或其他つくよみちゃん相关素材，则相关角色/素材署名和官方规约同样适用。

## 7. Restrictions / 使用限制

The bundled voice synthesis data and generated audio must not be used for purposes prohibited by the official つくよみちゃんコーパス terms, the つくよみちゃん character license, platform rules, or applicable laws.

In particular, the bundled voice synthesis data and generated audio must not be used for:

- uses that imply GPT-SoVITS Interactive, the user, or any third party is the official creator or rights holder of つくよみちゃん;
- uses that imply official endorsement by Rei Yumesaki or the つくよみちゃん Project;
- criticism, attack, harassment, or defamation of real people, organizations, products, or groups;
- activities calling for support for or opposition to a specific political position, religion, or ideology;
- publishing highly sensitive or strong expressive content without appropriate warnings or zoning;
- distributing generated audio as reusable voice material, a sound library, a training dataset, or similar material collection;
- fraud, impersonation, deception, or misleading uses;
- deepfake audio that makes it appear that a real person said something they did not say;
- voice synthesis software, conversational AI, or similar software in which a character other than つくよみちゃん speaks with this voice, unless Rei Yumesaki has given prior permission. Dubbing other characters with generated audio is allowed.

同梱音声合成数据及其生成音频不得用于つくよみちゃんコーパス官方规约、つくよみちゃんキャラクターライセンス、相关平台规则或适用法律禁止的用途。

尤其不得用于：

- 暗示 GPT-SoVITS Interactive、用户或任何第三方是つくよみちゃん官方创作者或权利人的用途；
- 暗示获得 夢前黎 或 つくよみちゃんプロジェクト 官方认可的用途；
- 对现实中的个人、组织、商品或团体进行批判、攻击、骚扰或诽谤的用途；
- 呼吁支持或反对特定政治立场、宗教或思想的活动；
- 未进行适当提示或分区而公开刺激性较强的表现内容；
- 将生成音频作为可二次利用的声音素材、声音素材库、训练数据集或类似素材集合进行分发；
- 诈骗、冒充、欺骗或误导性用途；
- 让人误以为现实人物说过未说内容的 deepfake 音频；
- 未事先取得夢前黎许可，制作让つくよみちゃん以外的角色用该声音说话的音声合成软件、会话 AI 或类似软件。用生成音频给其他角色配音是允许的。

## 8. No Standalone Redistribution / 不作为独立素材或模型再分发

The bundled voice synthesis data, runtime data, and precomputed cache data are provided as part of GPT-SoVITS Interactive's bundled character voices.

Do not redistribute, resell, mirror, re-upload, repack, or publish these files as a standalone voice model, voice model package, voice material collection, sound library, training dataset, or similar collection.

Do not upload, register, or provide these files to third-party voice model libraries, AI voice services, voice changer services, model-sharing platforms, or similar services that allow third parties to use the voice data or model separately from GPT-SoVITS Interactive.

Using these files with other engines or tools, separately from GPT-SoVITS Interactive, is outside the intended scope of this release. DesksideWorks does not provide support, warranty, or separate permission for such use. Any use remains subject to the official terms of the つくよみちゃん Project.

Where modification or redistribution is allowed, for example when an application bundles GPT-SoVITS Interactive or part of it (including only part of the code or only the voice model), the parts derived from つくよみちゃんコーパス remain subject to the つくよみちゃんコーパス terms. Anyone who modifies or redistributes them must accept this condition and pass it on to anyone who receives their derivative software or redistributed data. An application that bundles this voice also publishes software that can use the つくよみちゃんコーパス voice quality, and must meet the official credit and prohibition requirements itself.

同梱音声合成数据、运行数据和预计算缓存数据作为 GPT-SoVITS Interactive 随附角色声音的一部分提供。

不得将这些文件作为独立语音模型、语音模型包、声音素材集、声音素材库、训练数据集或类似集合进行再分发、转卖、镜像、再上传、重新打包或公开发布。

不得将这些文件上传、注册或提供给第三方音声模型库、AI 语音服务、变声服务、模型分享平台，或其他允许第三方脱离 GPT-SoVITS Interactive 使用该声音数据或模型的类似服务。

脱离 GPT-SoVITS Interactive、将这些文件用于其他引擎或工具，不属于本次发布的预期用途范围。DesksideWorks 不为此类使用提供支持、保证或单独许可。任何使用仍需遵守つくよみちゃんプロジェクト官方规约。

在允许改动或再分发的情况下（例如应用随附 GPT-SoVITS Interactive 或其一部分，包括只取部分代码或只取声音模型），来自つくよみちゃんコーパス的部分继续适用つくよみちゃんコーパス规约。进行改动或再分发的人须接受这一条件，并把它传给接收其派生软件或再分发数据的人。随附该声音的应用，本身也是公开了可使用つくよみちゃんコーパス声质的软件，须自行满足官方的署名和禁止事项要求。

## 9. No Warranty / 免责声明

This bundled voice synthesis data is provided as part of GPT-SoVITS Interactive, as is and without warranty of any kind. Use is at the user's own risk and remains subject to the official terms of the つくよみちゃん Project, applicable platform rules, and applicable laws.

本随附音声合成数据作为 GPT-SoVITS Interactive 的一部分按现状提供，不附带任何保证。使用风险由用户自行承担，并需遵守つくよみちゃんプロジェクト官方规约、相关平台规则及适用法律。
