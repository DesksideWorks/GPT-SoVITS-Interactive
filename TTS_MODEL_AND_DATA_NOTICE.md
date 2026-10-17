# TTS model and data notices / TTS 模型与数据说明

This index separates software licenses from model/data and character rights.
本索引区分软件许可、模型/数据许可与角色权利。源码的许可证不会自动覆盖所有权重与数据。
この索引はソフトウェア、モデル・データ、キャラクターの権利を区別します。

## Character synthesis models / 角色语音模型

- Shared semantic GPT: `resources/models/ami/gpt.ckpt`, trained using 小春音アミ (Ami Koharune) voice material; used for both characters.
- Tsukuyomi-chan SoVITS: `resources/models/tyc/sovits.pth`, trained using [つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV：夢前黎）](https://tyc.rei-yumesaki.net/material/corpus/) — Tsukuyomi-chan Corpus Vol.1 (CV: Rei Yumesaki), © Rei Yumesaki.
- Ami SoVITS: `resources/models/ami/sovits.pth`, trained using the selected normal-reading recordings from Amitaro's Voice Material Studio.
- Release hashes: `resources/model-release.json`. ORT graphs in `resources/models/ami/gpt.ort-dml/` are converted forms of the shared GPT, not independently licensed voice models.
- Character voice source terms, credits and restrictions:
  - `resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md`
  - `resources/avatars/tyc/CREDITS.md`
  - `resources/avatars/ami/AMITARO_VOICE_NOTICE.md`
  - `resources/avatars/ami/CREDITS.md`

两位角色共用以 Ami 素材训练的语义 GPT，各自使用对应角色的 SoVITS 和参考特征。因此，つくよみちゃん的合成流程也使用 Ami 来源的模型组件。两位声音来源方的署名均保留；转换成 ONNX 不会改变模型的来源或适用条款。模型并非素材作者提供或认可的官方语音模型。

## Semantic references / 语义参考

Each asset in `resources/semantic_prompts/` holds the phoneme IDs, text features and 25 Hz semantic tokens of one short reference. It conditions the shared GPT only. These assets contain no waveform and are not an acoustic condition for either character voice. A Japanese asset is bound to one character and is never applied to the other.
`resources/semantic_prompts/` 中的每份资源保存一条短参考的音素、文本特征和 25 Hz 语义 token，只作为共享 GPT 的条件。这些资源不含波形，也不作为任何角色的声学音色条件。日文资源各自绑定一位角色，不会用于另一位角色。
`resources/semantic_prompts/` の各アセットは短い参照の音素・テキスト特徴・25 Hz セマンティックトークンのみを含み、共有 GPT の条件としてだけ使われます。波形は含まれません。日本語のアセットはそれぞれ一人のキャラクターに紐づけられ、他のキャラクターには適用されません。

- `zh-greeting-v1` (Chinese, both characters): encoded from the first sentence (3.2 s) of a synthetic Mandarin speech clip that was generated with Qwen3-TTS and provided to DesksideWorks for this project. The Qwen3-TTS model is released under the Apache License 2.0. Modification notice: the clip was trimmed to its first sentence, resampled to 16 kHz and converted to semantic tokens.
- `ja-ami-v1` (Japanese, 小春音アミ only): the semantic tokens of that character's own bundled reference recording, taken from `resources/avatars/ami/references/default.cache.pt`. Terms: `resources/avatars/ami/AMITARO_VOICE_NOTICE.md`.
- `ja-tyc-v1` (Japanese, つくよみちゃん only): encoded from the first sentence (2.95 s) of that character's own bundled reference recording. Terms: `resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md`. Neither Japanese asset is used for the other character.
- `en-arctic-clb-v1` (English, both characters): encoded from utterance `arctic_b0108` of the CMU ARCTIC `cmu_us_clb_arctic` database, http://www.festvox.org/cmu_arctic/ . Copyright (c) 2005 Carnegie Mellon University. The database is free for any purpose, commercial or otherwise; its copyright notice, conditions and disclaimer are retained in `Licenses/CMU-ARCTIC-COPYING.txt`. Modification notice: the recording was resampled to 16 kHz and converted to semantic tokens.

The source and license of each asset are also recorded in the `provenance` field of its JSON manifest.
各资源的来源与许可同时记录在对应 JSON 清单的 `provenance` 字段。

## Chinese text frontend / 中文文本前端

### Chinese RoBERTa

- Model: `hfl/chinese-roberta-wwm-ext-large`.
- Bundled location: `resources/models/chinese-roberta-wwm-ext-large/`.
- Authors/project: HFL, Chinese-BERT-wwm; https://github.com/ymcui/Chinese-BERT-wwm
- Model card: https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
- Project license: Apache License 2.0; original text: `Licenses/Chinese-BERT-wwm-LICENSE.txt`.

### g2pW

- Project: https://github.com/GitYCC/g2pW
- Bundled model/data: `resources/models/g2pw/`.
- Upstream project license: Apache License 2.0; original text: `Licenses/G2PW-LICENCE.txt`.
- Reference: Yi-Chang Chen, Yu-Chuan Steven, Yen-Cheng Chang and Yi-Ren Yeh, *g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin*, Interspeech 2022, pp. 1926–1930, https://doi.org/10.21437/Interspeech.2022-216.
- The frontend includes adaptations through PaddleSpeech and GPT-SoVITS_minimal_inference; original source headers are retained.

The bundled frontend files were copied from the optimized GPT-SoVITS checkout. Local provenance and hashes are in `chinese-runtime-manifest.json`; that manifest records an import snapshot, not a live hash of every subsequently edited source file.
中文前端来自优化版仓库，导入来源和当时的哈希见上述清单；该清单不是所有后续修改文件的实时哈希表。

### fastText language identification

- Model: `resources/models/fast_langdetect/lid.176.bin` (176-language identification).
- Provider: Facebook Research / fastText; https://fasttext.cc/docs/en/language-identification.html
- Original model: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
- Model license: **Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)**; https://creativecommons.org/licenses/by-sa/3.0/ ; full text: `Licenses/CC-BY-SA-3.0.txt`.
- Training sources credited by the provider: Wikipedia, Tatoeba and SETimes.
- Model is bundled for language identification, not character speech generation; this model is not trained or adapted for this runtime.
- fastText software is separately MIT-licensed: `Licenses/fastText-LICENSE.txt`.

`lid.176.bin` 的许可与 fastText 代码不同。其许可不应被误写为 MIT，也不应套用角色声音数据的再分发限制。本声明不将该模型与本运行环境的其他独立组件视为同一授权作品。

## Dictionaries and installed libraries / 词典与运行库

Python distributions, pyopenjtalk and its bundled license files, NLTK tagger resource notices and CMUdict notices are collected in `PythonThirdPartyNotices.txt` and `PythonThirdPartyLicenseTexts.txt`.
The original files remain in the runtimes. Frontend source files also retain their own notices, including PaddleSpeech-derived normalization code.

Third-party open-source software and independently licensed models/data remain governed by their original licenses. The character-data restrictions in this repository's notices do not remove rights granted under those licenses.
第三方开源软件和独立授权模型/数据仍适用各自原始许可。本仓库说明中对角色声音数据的限制不缩减这些原始许可授予的权利。
第三者のオープンソースソフトウェアと個別に許諾されたモデル・データには、それぞれの原ライセンスが適用されます。
