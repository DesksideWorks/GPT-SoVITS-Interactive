# Credits: つくよみちゃん Voice

This file lists the voice-related credits for the つくよみちゃん voice included in **GPT-SoVITS Interactive**.

本文件列出 **GPT-SoVITS Interactive** 随附的つくよみちゃん声音的署名信息。

## つくよみちゃん / Tsukuyomi-chan Voice

```text
本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。

■つくよみちゃんコーパス（CV.夢前黎）
使用音源：Vol.1 声優統計コーパス（JVSコーパス準拠）
https://tyc.rei-yumesaki.net/material/corpus/

■つくよみちゃん公式サイト
https://tyc.rei-yumesaki.net/
```

The local TTS voice data for つくよみちゃん / Tsukuyomi-chan included in GPT-SoVITS Interactive was prepared by **桌边工房 / DesksideWorks / デスクサイド工房**.

The voice synthesis data uses voice data from **つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV.夢前黎）** as training/source voice data.

In GPT-SoVITS Interactive, the つくよみちゃん synthesis path also uses data from other speakers. This is how GPT-SoVITS Interactive is built, not something that comes from the official つくよみちゃん materials: the shared semantic GPT is trained with 小春音アミ voice material, and the Chinese and English semantic references come from a Qwen3-TTS generated Mandarin sentence and a CMU ARCTIC English recording. The timbre comes from the つくよみちゃん SoVITS model and reference. See `TTS_MODEL_AND_DATA_NOTICE.md`.

The TTS engine used by the runtime is **GPT-SoVITS**. The GPT-SoVITS engine/code is covered by its own open-source license, which is provided separately with the code or third-party license notices.

GPT-SoVITS Interactive and the included voice synthesis data are **not** official software or official voice synthesis data created, distributed, or endorsed by Rei Yumesaki or the つくよみちゃん Project.

GPT-SoVITS Interactive 随附的つくよみちゃん / Tsukuyomi-chan 本地 TTS 用音声合成数据，由 **桌边工房 / DesksideWorks / デスクサイド工房** 制作和整理。

该音声合成数据使用了 **つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV.夢前黎）** 的声音数据作为训练/来源声音数据。

在 GPT-SoVITS Interactive 中，つくよみちゃん的合成流程也用到了其他说话人的数据。这是 GPT-SoVITS Interactive 自身的做法，并非来自つくよみちゃん官方素材：共用的语义 GPT 以小春音アミ的声音素材训练；中文和英文的语义参考分别来自 Qwen3-TTS 生成的一句中文和 CMU ARCTIC 的一句英文录音。音色来自つくよみちゃん自己的 SoVITS 模型和参考。详见 `TTS_MODEL_AND_DATA_NOTICE.md`。

本运行环境使用的 TTS 引擎为 **GPT-SoVITS**。GPT-SoVITS 引擎/代码适用其自身的开源许可证，该许可证已随代码或第三方开源许可说明另行提供。

GPT-SoVITS Interactive 及其随附的音声合成数据并非由 夢前黎 或 つくよみちゃんプロジェクト 制作、分发或认可的官方软件/官方音声合成数据。

## Related Notice / 相关说明

For details about the bundled voice synthesis data, official terms, and usage restrictions, see:

```text
TSUKUYOMICHAN_VOICE_NOTICE.md
```

关于随附音声合成数据、官方规约和使用限制，请参阅：

```text
TSUKUYOMICHAN_VOICE_NOTICE.md
```
