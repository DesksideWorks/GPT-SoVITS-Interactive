<h1 align="center">GPT-SoVITS Interactive</h1>

<p align="center">
  <a href="README.md">中文</a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="docs/DISTRIBUTION.md">Distribution</a>
</p>

<p align="center"><strong>A local GPT-SoVITS voice service for games and desktop companions: audio plays while it is generated, with a timeline for every character of the text</strong></p>

<p align="center">
  <img src="docs/assets/interactive-banner.png" width="960" alt="GPT-SoVITS Interactive banner: an easy way to add character voices to applications and games, with Chinese, Japanese and English, CUDA, DirectML, CPU, streaming audio and character timelines. The characters shown are つくよみちゃん and 小春音アミ.">
</p>

<p align="center">
  <a href="#desktopdaily">Our game</a> ·
  <a href="#difference">Compared with the official API</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#voices">Bundled voices</a> ·
  <a href="#api">API</a> ·
  <a href="#scope">Scope</a> ·
  <a href="#performance">Performance</a> ·
  <a href="docs/TECHNICAL.en.md">Technical design</a> ·
  <a href="#license">License</a>
</p>

Send a line of dialogue to the local service and it streams the audio back while it is still generating, telling you when each character of the text is spoken. Subtitle highlighting, expressions and gestures can follow the same playback clock: no waiting for the whole sentence, and no separate alignment pass afterwards.

- **Audio and text stream together**: each audio chunk carries its global sample range and the characters newly spoken in it.
- **Three languages right after extraction**: the Windows package includes two characters, 小春音アミ and つくよみちゃん, together with the language frontends and Python runtimes, so Chinese, Japanese, English and mixed text work out of the box.
- **Not just NVIDIA**: CUDA on NVIDIA, DirectML on AMD / Intel, or CPU only.

It is an inference service for applications to call; it has no training and no WebUI. Characters are trained with upstream GPT-SoVITS (v2ProPlus) and then added here.

<a id="desktopdaily"></a>

## Our game: DesktopDaily

<!-- Game screenshot or GIF: as a character speaks, nods, gestures and expressions land on the matching words. Asset to be decided. -->

**[DesktopDaily](https://store.steampowered.com/app/4724470/_/)** is a 3D desktop companion we are making, coming free on Steam. つくよみちゃん and 小春音アミ live on your desktop: while you work or browse, the two of them stay in a transparent window, do their own thing, and now and then react to you.

- **Chat with the characters**: connect your own API key and a compatible LLM. The characters answer in Chinese, Japanese or English, and this repository's TTS reads their lines aloud locally.
- **Acting while they speak**: the LLM plans gestures and expressions together with each line. With this repository's streaming timeline, nods, gestures and expressions land on the right word of the line, without waiting for the whole line to be generated.

This TTS was put together for the game's character dialogue, and the banner at the top shows the game's two characters. If you like it, add the game to your Steam wishlist, or follow us for development updates:

- **[Steam store page (add to wishlist)](https://store.steampowered.com/app/4724470/_/)** · **[Website](https://www.desksideworks.com)**
- **[Bilibili](https://space.bilibili.com/32459108)** · **[Xiaoheihe](https://www.xiaoheihe.cn/app/user/profile/95841191)** · **[X / Twitter](https://x.com/DesksideWorks)**

<a id="difference"></a>

## How it differs from the official GPT-SoVITS API

The model architecture, weight format and synthesis method follow [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), and the inference code builds directly on [GPT-SoVITS_minimal_inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference). Foundations such as autoregressive caching, ONNX graphs, I/O binding, streaming inference and an API are already provided upstream. What this repository does is put them to work inside applications. In the official column, the interface rows were checked against `api_v2.py` on the upstream main branch as of October 2026:

| | Official GPT-SoVITS | This repository |
|---|---|---|
| Purpose | A complete tool for training, WebUI and inference | Inference only, running alongside your application |
| Streaming output | Audio stream (`streaming_mode` 1–3; 0 turns streaming off) | NDJSON: audio chunks + sample ranges + per-character timeline |
| Which text is playing | Not returned | Start and end time of every character, on the same sample clock as the audio |
| Characters speaking other languages | GPT conditioning comes from the reference recording in the request | One native semantic reference per language; the voice still comes from the character |
| Each request | Carries a reference recording path | Reference features are precomputed; only the character ID is sent |
| Service management | Restart / exit, switch weights | Character switching, GPU hibernation and wake-up, serialized requests, cleanup after disconnects |

<a id="quick-start"></a>

## Quick start

### Download

The package comes in two editions, one per GPU type. Code and characters are the same; only the runtime differs (the CUDA edition leaves out the GPT graphs that only DirectML uses). Download one:

- **CUDA** (`…-win-x64-cuda`): NVIDIA GPUs.
- **DirectML** (`…-win-x64-directml`): compatible AMD / Intel / NVIDIA GPUs.

Each package is split into volumes (`.7z.001`, `.7z.002`, …). Download all volumes of one edition into the same folder and extract `.7z.001` with [7-Zip](https://www.7-zip.org/). Both editions can also run on the CPU. GPU inference needs the matching graphics driver; Python and the inference dependencies come from the bundled environment.

### 1. Start the service

Open PowerShell in the extracted directory (the one containing this README) and run the command for your GPU. The command creates and prints an access token for this service session; keep it, because requests need it.

**NVIDIA CUDA:**

```powershell
$env:GSV_TTS_TOKEN = [guid]::NewGuid().ToString('N')
Write-Output $env:GSV_TTS_TOKEN
& "./Runtime/Python_CUDA/python.exe" api_server.py --device cuda --cpu_threads 4 --avatar ami --reference default --resources resources --port 8001
```

**DirectML (compatible AMD / Intel / NVIDIA GPUs):**

```powershell
$env:GSV_TTS_TOKEN = [guid]::NewGuid().ToString('N')
Write-Output $env:GSV_TTS_TOKEN
& "./Runtime/Python_DirectML/python.exe" api_server.py --device dml --cpu_threads 4 --avatar ami --reference default --resources resources --port 8001
```

- For CPU inference, change `--device cuda` or `--device dml` to `--device cpu`. `--device auto` picks a backend available in the current environment.
- `--avatar tyc` starts with the other character; you can also choose the character in later requests.
- `--device_name` selects a device by name, and `--voices_config` selects a voice configuration.

Keep the service terminal open and wait for the `GSV_TTS_STARTUP` JSON; `"ok": true` in it means the service is ready. A failed startup prints the same line with `ok` set to `false`, and `code` and `message` (sometimes also `detail`) give the reason. The service listens only on `127.0.0.1`, on port 8001 by default (change it with `--port`), and every request needs `Authorization: Bearer <token>`.

### 2. Generate your first line

Open another PowerShell and replace the placeholder with the token printed by the service. The audio is saved as `speech.wav` in the current directory:

```powershell
$ttsHeaders = @{ Authorization = 'Bearer <token printed by the service>' }
$ttsBody = @{
    text = 'Hello, what would you like to talk about today?'
    modelId = 'ami'
    textLang = 'en'
} | ConvertTo-Json -Compress
Invoke-WebRequest -Uri 'http://127.0.0.1:8001/api/tts/predict' -Method Post -Headers $ttsHeaders -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($ttsBody)) -OutFile 'speech.wav'
```

Set `modelId` to `tyc` to use つくよみちゃん. `textLang` accepts `zh`, `ja`, `en` and `auto` for automatic detection; change the text to try other languages or mixed dialogue.

### 3. Play as it arrives: streaming output

The client below uses only the Python standard library and runs with the bundled Python. Before running it, set `$env:GSV_TTS_TOKEN = '<token printed by the service>'` in the client terminal. It saves the PCM chunk by chunk and prints the timeline; hand `pcm` to your application's playback queue to play the audio as it arrives. The resulting `.f32` file has no WAV header.

```python
import base64
import json
import os
import urllib.request

payload = {
    "text": "Let's have a cup of tea together, and then talk about what has been happening lately.",
    "modelId": "ami",
    "textLang": "en",
    "chunkLength": 12,
}
request = urllib.request.Request(
    "http://127.0.0.1:8001/api/tts/predict-stream",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["GSV_TTS_TOKEN"],
    },
)

cursor = 0
finished = False
with urllib.request.urlopen(request, timeout=180) as response, open("speech.f32", "wb") as output:
    for line in response:
        event = json.loads(line)
        if event["type"] == "error":
            raise RuntimeError(event["error"])
        if event["type"] == "chunk":
            pcm = base64.b64decode(event["audio"])
            assert event["audio_start_sample"] == cursor
            assert len(pcm) % 4 == 0
            cursor += len(pcm) // 4
            assert event["audio_end_sample"] == cursor
            output.write(pcm)
            print("chunk", event["chunk_index"], event["audio_start_sample"], event["audio_end_sample"])
            for item in event["timestamps"]:
                print("   ", item["char_idx"], item["char"], f"{item['start']:.2f}", f"{item['end']:.2f}")
        elif event["type"] == "done":
            assert event["audio_end_sample"] == cursor
            finished = True
if not finished:
    raise RuntimeError("Stream ended before the done event")
```

Output from one run (RTX 4060 Ti, CUDA; sampling is random, so the numbers differ a little every time). Each `chunk` is a piece of PCM that can play right away; the indented lines below it are the characters newly spoken in that chunk: index, character, start and end in seconds. In English every letter, space and punctuation mark gets its own entry, and letters that are not pronounced separately (the e of "have", for example) have zero length. The middle of the last chunk is left out here.

```text
chunk 1 0 15360
    0 L 0.12 0.20
    1 e 0.20 0.28
    2 t 0.28 0.34
    3 ' 0.34 0.34
    4 s 0.34 0.42
    5   0.42 0.42
    6 h 0.42 0.46
    7 a 0.46 0.52
chunk 2 15360 30720
    8 v 0.52 0.60
    9 e 0.60 0.60
    10   0.60 0.60
    11 a 0.60 0.62
    12   0.62 0.62
    13 c 0.62 0.74
    14 u 0.74 0.80
    15 p 0.80 0.90
    16   0.90 0.90
    17 o 0.90 0.94
    18 f 0.94 1.02
chunk 3 30720 46080
    19   1.02 1.02
    20 t 1.02 1.14
    21 e 1.14 1.28
    22 a 1.28 1.28
    23   1.28 1.28
    24 t 1.28 1.30
    25 o 1.30 1.32
    26 g 1.32 1.44
chunk 4 46080 61440
    27 e 1.44 1.44
    28 t 1.44 1.48
    29 h 1.50 1.60
    30 e 1.60 1.72
    31 r 1.72 1.72
    32 , 1.72 1.98
chunk 5 61440 76800
    33   1.98 1.98
    34 a 1.98 2.12
    35 n 2.12 2.16
    36 d 2.16 2.18
    37   2.18 2.18
    38 t 2.18 2.24
    39 h 2.24 2.30
    40 e 2.30 2.38
    41 n 2.38 2.38
    42   2.38 2.38
    43 t 2.38 2.50
chunk 6 76800 145920
    44 a 2.50 2.64
    45 l 2.64 2.72
    46 k 2.72 2.72
    47   2.72 2.72
    48 a 2.72 2.74
    49 b 2.74 2.82
    …
    80 t 4.20 4.24
    81 e 4.24 4.34
    82 l 4.34 4.52
    83 y 4.52 4.52
    84 . 4.52 4.56
```

<a id="voices"></a>

## Bundled characters and voice material

| Character | ID | Languages | Default reference |
|---|---|---|---|
| 小春音アミ (Ami Koharune) | `ami` | Chinese, Japanese, English and mixed text | `default`, precomputed |
| つくよみちゃん | `tyc` | Chinese, Japanese, English and mixed text | `default`, precomputed |

Both characters are configured in [voices.json](resources/voices.json). They share the GPT semantic model while using separate SoVITS weights and acoustic reference caches. The per-language semantic references are included, and the DirectML edition also includes the GPT graphs DirectML needs. Set `modelId` in a request to select a character; adding your own character is described [later](#custom).

### What can these voice assets be used for?

The bundled voices are suitable for local character dialogue, desktop companion responses, game NPC lines, subtitle-synchronized performances and prototypes. The models, acoustic references and runtime configuration are already integrated, so you can audition Chinese, Japanese, English and mixed text through the API before deciding how to use them.

If you are looking for **anime-style voices available for commercial use free of charge under specified conditions**, 小春音アミ and つくよみちゃん are worth exploring. Both have public voice sources and terms. Possessing the models does not grant unconditional commercial use. Check attribution, character-use rules, content restrictions and redistribution restrictions before publishing a work, using the voices commercially or including character artwork.

- [小春音アミ voice notice](resources/avatars/ami/AMITARO_VOICE_NOTICE.md) · [official terms](https://amitaro.net/voice/voice_rule/)
- [つくよみちゃん voice notice](resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md) · [character license](https://tyc.rei-yumesaki.net/about/terms/) · [corpus terms](https://tyc.rei-yumesaki.net/material/corpus/)

The rights for character voice data, reference caches and generated audio are separate from the code license; see the [license section](#license) at the end.

<a id="3d-models"></a>

### 3D models to use with the voices

We also offer SD-style 3D models of both characters, **free to download and use commercially under the model licenses and official character terms, with no model usage or license fees**. These are the designs shown in the banner at the top of this page. You can pair them with this repository's character voices to build desktop companions, game NPCs or other interactive applications:

- [Tsukuyomi-chan SD Dress Ver.](https://www.desksideworks.com/en/models/tsukuyomi-chan)
- [Ami Koharune SD Everyday Outfit Ver.](https://www.desksideworks.com/en/models/ami-koharune)

Specifications for the currently available **v1.0** releases:

| Specification | Tsukuyomi-chan SD Dress Ver. | Ami Koharune SD Everyday Outfit Ver. |
|---|---|---|
| Triangle count | Approx. 96,000 tris | Approx. 130,000 tris |
| MMD-style expressions | 64 | 64 |
| Intermediate / corrective expressions | 20 | 20 |
| ARKit expressions | 52 | 52 |
| Available formats | VRM / Warudo / UnityPackage | VRM / Warudo / UnityPackage |

After integrating a model into your application, you can use this repository's audio and per-character timeline to coordinate subtitles, expressions and gestures as the character speaks.

Each model's page provides previews, downloads, credits and usage terms. The 3D models and character voices have separate licenses; follow both when using them together.

<a id="credits"></a>

### Voice credits

This repository publishes models trained on the two characters' voices. Under the [つくよみちゃんコーパス terms](https://tyc.rei-yumesaki.net/material/corpus/) and the [terms of あみたろの声素材工房](https://amitaro.net/voice/faq/), the following must be shown prominently:

> 本ソフトウェアの音声合成には、フリー素材キャラクター「つくよみちゃん」（© Rei Yumesaki）が無料公開している音声データを使用しています。
>
> ■つくよみちゃんコーパス（CV.夢前黎）<br>
> https://tyc.rei-yumesaki.net/material/corpus/
>
> ■3Dモデル：つくよみちゃん SD ドレスVer.（3D化参考図：橘子tg / 3D制作：CK-迭代型）
>
> ■つくよみちゃん公式サイト<br>
> https://tyc.rei-yumesaki.net/
>
> つくよみちゃんの声質を使用する場合は、出力した音声を次の目的で使用することを禁止します。<br>
> ■人を批判・攻撃すること。（「批判・攻撃」の定義は、[つくよみちゃんキャラクターライセンス](https://tyc.rei-yumesaki.net/about/terms/#condition3)に準じます）<br>
> ■特定の政治的立場・宗教・思想への賛同または反対を呼びかけること。<br>
> ■刺激の強い表現をゾーニングなしで公開すること。<br>
> ■他者に対して二次利用（素材としての利用）を許可する形で公開すること。

> 音声：小春音アミ（CV：あみたろ）<br>
> 学習データ：あみたろの声素材工房（<https://amitaro.net/>）<br>
> あみたろの声素材工房が作成した公式モデルではありません。この音声モデルを利用する場合も、「音声合成モデルに、あみたろの声素材工房（<https://amitaro.net/>）の音声データを使用しています」などのクレジット表記が必要で、[あみたろの声素材工房の利用規約](https://amitaro.net/voice/voice_rule/)の範囲内でのみ使用できます。
>
> ■3Dモデル：小春音アミ SD ふだんぎVer.（3D化参考図：橘子tg / 3D制作：CK-迭代型）

- In this repository, つくよみちゃん's synthesis also uses data from other speakers. This is this repository's own design, not something that comes from the official つくよみちゃん side: the semantic GPT shared by both characters is trained on 小春音アミ's material, and the Chinese and English pronunciation references come from one Mandarin sentence generated with Qwen3-TTS and one English recording from CMU ARCTIC. The voice itself comes from つくよみちゃん's own SoVITS model and reference.
- Neither character's voice model, nor the 3D models in the banner, are official works made or endorsed by the material providers.
- Credit for generated audio follows each provider's terms:
  - 小春音アミ: works must credit her, for example 「音声合成モデルに、あみたろの声素材工房（<https://amitaro.net/>）の音声データを使用しています」, and use must stay within the あみたろ terms (all-ages and so on).
  - つくよみちゃん: her side does not require credit for generated audio (the corpus terms make an exception for audio output by voice-synthesis software). When her artwork, 3D model or other character material appears on screen, credit it under the [character license](https://tyc.rei-yumesaki.net/about/terms/). However, this repository has her share the GPT trained on Ami's material with 小春音アミ (this repository's design, not an official one), so つくよみちゃん audio generated with this repository also needs the あみたろ credit above and must follow the same use restrictions.

<a id="redistribute"></a>

### Distributing the character voices in your own application

Bundling the included つくよみちゃん model in your game or app means you are also publishing software that uses her voice (「③声質を使用した音声合成ソフト等を公開する」 in the corpus terms), so you need to meet the points below yourself. This is only a summary; the [official terms](https://tyc.rei-yumesaki.net/material/corpus/) prevail:

- **Credit**: before release, email 夢前黎 to agree on how to credit; or show the official credit text prominently in a sufficiently large font (paid software must also show it before the user pays). The official side provides three templates, depending on whether users must credit generated audio: [required](https://tyc.rei-yumesaki.net/wp-content/uploads/tsukuyomi-chan_credit_template1.html), [optional](https://tyc.rei-yumesaki.net/wp-content/uploads/tsukuyomi-chan_credit_template2.html), [not required](https://tyc.rei-yumesaki.net/wp-content/uploads/tsukuyomi-chan_credit_template3.html).
- **Prohibited uses**: your terms of use must prohibit the four uses listed above; if you leave them out, consult 夢前黎 first.
- **Mixing disclosure**: the bundled つくよみちゃん uses data from other speakers (see above), so your documentation must state this as well.
- **Modification and redistribution**: if you let others modify or redistribute your software (including taking only part of the code or only the voice model), they must agree that the parts derived from the つくよみちゃんコーパス follow the corpus terms. This requirement must also be passed on downstream.
- **Voicing other characters**: using her voice as the voice of a character other than つくよみちゃん (your own character or a conversational AI, for example) needs prior permission from 夢前黎. [A past answer the official side has published](https://tyc.rei-yumesaki.net/material/corpus/conversational-ai/) asks that it be made clear to everyone that another character is speaking with her voice; one way it gives is a credit kept visible on screen, and if that is not possible, consult her again. Simply using generated audio to dub other characters is fine.

For 小春音アミ's voice, the [terms](https://amitaro.net/voice/voice_rule/) and [FAQ](https://amitaro.net/voice/faq/) of あみたろの声素材工房 apply:

- **Consult first**: bundling the model in software or an app, or distributing it publicly in any other way, needs prior consultation with あみたろの声素材工房.
- **Free only**: a model made only from her voice may only be distributed free of charge. You may not add a paid feature outside the software that unlocks this voice (except with revenue shared with あみたろ), and you may not register it with a service that lets third parties use voice models.
- **Include the terms**: include a link to the terms, or the Readme from あみたろの声素材工房, when distributing.

<a id="api"></a>

## API

| Method and path | Response / behavior |
|---|---|
| `POST /api/tts/predict` | Complete WAV |
| `POST /api/tts/predict-timeline` | JSON: Base64 WAV, character timeline, sample rate and duration |
| `POST /api/tts/predict-stream` | NDJSON: incremental PCM and timeline |
| `GET /api/tts/models` | Available characters and reference configurations |
| `POST /api/tts/switch-model` | Switch character / reference |
| `GET /api/tts/status` | Current state, backend, device and startup stage timings |
| `POST /api/tts/hibernate` | Release GPU models / sessions held by the engine, retaining recoverable state |
| `POST /api/tts/wake` | Wake or load a specified character |
| `POST /api/tts/benchmark` | Built-in short-dialogue benchmark |
| `POST /v1/audio/speech` | WAV endpoint accepting `input` / `voice` |
| `GET /v1/models` | OpenAI-compatible model list (authentication required) |
| `GET /v1/voices` | Voice configuration list (authentication required) |
| `POST /v1/voices/reload` | Reload voice configuration (authentication required) |

`/v1/audio/speech` always outputs WAV; use `/api/tts/predict-stream` to play audio as it arrives.

### Synthesis requests

The three `/api/tts/predict*` endpoints share the main fields:

```json
{
  "text": "今天也辛苦了。お疲れさま。Have a good rest!",
  "modelId": "ami",
  "referenceId": "default",
  "textLang": "auto",
  "speed": 1.0,
  "noiseScale": 0.35,
  "topK": 15,
  "topP": 1.0,
  "temperature": 1.0,
  "chunkLength": 12
}
```

- `modelId` selects a bundled character with `ami` or `tyc`; when the character and reference are omitted, the current active configuration is used. `textLang` accepts `auto`, `zh`, `ja` and `en`.
- `speed` sets the speaking rate. `1.0` is the pace the model generates on its own; larger values are faster and smaller values slower. It works in all three languages and with streaming.
- `chunkLength` applies only to streaming and sets how many GPT semantic tokens are produced per chunk. The HTTP default is 24, limited to 10–100. It is **not a duration in milliseconds or a fixed PCM packet size**; first-chunk latency also depends on look-ahead, decoding context and device speed.
- Pauses between segments are read from the character's `defaults.pause_length`.
- The legacy request fields `g2pPriorityMode`, `languages` and `generateTimeline` are still accepted but have no effect: the endpoint decides the response form, and `textLang` sets the language.

### Streaming events

The response type is `application/x-ndjson`, with one JSON object per line and the event sequence `start → chunk… → done`. An error after the stream has started produces an `error` event; errors before it starts, such as failed authentication, empty text, an unknown character or a model that fails to load, return an HTTP error status instead, with no NDJSON events.

| `chunk` field | Meaning |
|---|---|
| `audio` | Base64-encoded **little-endian float32, mono raw PCM**, without a WAV header |
| `audio_start_sample` / `audio_end_sample` | Global half-open sample interval; successive intervals should be contiguous |
| `sample_rate` / `channels` | 32000 / 1 in the current streaming implementation |
| `chunk_index` | Chunk number starting at 1 |
| `timestamps` | Character timestamps newly delivered in this chunk; may be empty |
| `gpt_finished` | Whether GPT has finished the current sentence, not the whole request |
| `is_final` | False for audio chunks; `done` confirms completion |
| `server_first_chunk_latency_seconds` | Seconds the server took from starting generation to sending the first chunk |

The `start` event gives `sample_rate` and `channels`. The `done` event gives the total sample count `audio_end_sample`, the chunk count `chunk_count`, and the server-measured first-chunk latency, total time and audio duration (in seconds). The `error` field of an `error` event describes the error.

Timeline entries contain `char_idx`, `char`, `start`, `end`, `phones`, `ph_start`, `ph_end` and `approximate`:

- `start` / `end` are in seconds, counted from the start of the whole request's audio. The first character's `start` is usually above zero, and neighbouring characters can leave a gap: time without voice belongs to no character unless it is the pause of a punctuation mark.
- Each character is delivered once, in order, and entries already sent are never rewritten. Characters removed by normalization (for example `……`, which is read as a single pause) also get entries, with zero duration. Only whitespace dropped when the text is split into sentences has no entry: whitespace at the start and end of the text, and between two sentences synthesized separately.
- `char_idx` is a **Unicode code-point index into the original text**. C# / JavaScript clients must not use it directly as a UTF-16 index, especially with emoji or supplementary characters.

The first audio chunk and the `done` event from another run of the same sentence (`audio` is shortened; each event is actually one JSON object on a single line):

```json
{
  "type": "chunk",
  "audio": "TEcqOcdREjn1vQw5…",
  "audio_start_sample": 0,
  "audio_end_sample": 15360,
  "sample_rate": 32000,
  "channels": 1,
  "is_final": false,
  "chunk_index": 1,
  "timestamps": [
    {"char_idx": 0, "ph_start": 0, "ph_end": 1, "phones": ["L"], "start": 0.12, "end": 0.2, "char": "L", "approximate": true},
    {"char_idx": 1, "ph_start": 1, "ph_end": 2, "phones": ["EH1"], "start": 0.2, "end": 0.28, "char": "e", "approximate": true},
    {"char_idx": 2, "ph_start": 2, "ph_end": 3, "phones": ["T"], "start": 0.28, "end": 0.34, "char": "t", "approximate": true},
    {"char_idx": 3, "ph_start": 3, "ph_end": 3, "phones": [], "start": 0.34, "end": 0.34, "char": "'", "approximate": true},
    {"char_idx": 4, "ph_start": 3, "ph_end": 4, "phones": ["S"], "start": 0.34, "end": 0.38, "char": "s", "approximate": true},
    {"char_idx": 5, "ph_start": 4, "ph_end": 4, "phones": [], "start": 0.38, "end": 0.38, "char": " ", "approximate": true},
    {"char_idx": 6, "ph_start": 4, "ph_end": 5, "phones": ["HH"], "start": 0.38, "end": 0.44, "char": "h", "approximate": true},
    {"char_idx": 7, "ph_start": 5, "ph_end": 6, "phones": ["AE1"], "start": 0.44, "end": 0.5, "char": "a", "approximate": true}
  ],
  "gpt_finished": false,
  "server_first_chunk_latency_seconds": 0.44544230005703866
}
```

```json
{
  "type": "done",
  "sample_rate": 32000,
  "channels": 1,
  "audio_end_sample": 142080,
  "is_final": true,
  "chunk_count": 6,
  "server_first_chunk_latency_seconds": 0.44544230005703866,
  "server_total_seconds": 1.114124899962917,
  "audio_duration_seconds": 4.44
}
```

The HTTP streaming endpoint always computes and returns the timeline. Calling the Python engine's `LatentStream.stream(..., timeline=False)` directly turns this extra output off. How the timeline comes out of the same inference pass is explained in [Technical design](docs/TECHNICAL.en.md#streaming-timeline).

### Service lifecycle

- **Requests run one at a time.** POST operations that change engine state run in sequence, and the lock is held until the streaming response ends, so character switching, hibernation and later synthesis never change the model while an earlier stream is still using it. When a client disconnects, the service lets the current compute step finish, then closes the generator, restores precision and lets later operations through.
- **Queries don't wake it.** `/api/tts/status` only reads state.
- **Hibernation releases GPU resources.** PyTorch models and caches move back to the CPU, and ORT DirectML sessions are released. While the process is alive, the CUDA context, drivers and libraries may still hold some VRAM.

<a id="integration"></a>

### Integration tips

[DesktopDaily](#desktopdaily) drives its characters' acting with the streaming timeline, puts the TTS into hibernation to free VRAM after a while without speech, and wakes it again before the next line. The game's player follows the rules below, which can also serve as a reference for other applications:

- **Queue audio for playback as soon as it arrives.** The audio thread only consumes the PCM queue; the receiver checks the sample format, that sample intervals are contiguous, and the total sample count at completion.
- **Pause the content clock when data runs out.** Silence the playback device inserts to keep its output going does not count toward line progress, so subtitles, expressions and gestures don't run ahead while waiting for the next chunk. Pauses inside the synthesized audio itself count as usual.
- **Keep playing the queue after `done`.** Generation finishing and playback finishing are judged separately: the performance ends only after the last real sample plays, so the end of the line isn't cut off.
- **On interruption, cancel both the request and the old playback queue**: stop the old audio, discard unplayed PCM, and isolate late callbacks from the old request.
- **Drive subtitles and acting by the sample position actually played, divided by `sample_rate`.** One character can span several audio chunks and its timeline entry can arrive slightly late, so the player should keep a little buffer.
- **Prepare with the language you will actually use.** The Chinese, Japanese and English frontends load the first time they are used. After the user chooses a voice language, send one real streaming request with a short line in that language (with an explicit `textLang`), and allow normal synthesis only after its audio chunks and `done` arrive; show this wait as a preparing state. Track readiness per process: it stays valid across hibernation and wake-up, but has to be redone after a service restart.

Also treat a non-2xx HTTP status, a stream that sends `error`, or a stream that ends without `done` as a failure; don't use audio containing only the first half of the line as a complete result.

<a id="scope"></a>

## Scope and limitations

- **Model version**: GPT-SoVITS **v2ProPlus** (32 kHz, 25 Hz semantic frame rate) is currently supported. In-sentence streaming decoding checks the model structure; other versions need adapted decoding parameters.
- **No training**: there is no training code, WebUI or fine-tuning tool. Train your own character with upstream GPT-SoVITS first, then add it as described in [Custom characters](#custom).
- **Platform**: the package is Windows x64 only and has only been verified on Windows. The source directory has no models or runtimes; fill them in from the package of the same version, as described in [Distribution](docs/DISTRIBUTION.md).
- **One application at a time**: the service listens only on the local machine and requests need the token. It runs one inference task at a time, which suits a single application that manages character synthesis and switching.
- **The timeline is an approximate alignment**: it is derived from SoVITS attention and suits subtitle highlighting, text reveal and rough gesture sync. Precise lip sync or phoneme labelling needs a separate alignment. Character boundaries inside Japanese kanji words and English words are approximate.
- **Where semantic references apply**: only to single-language requests (Chinese with only a few Japanese words in it counts as Chinese). Other mixed Chinese / Japanese / English requests use no semantic reference.
- **DirectML**: needs a compatible Windows GPU and driver. The GPT graphs have a fixed capacity (256 text positions, 512 KV positions); long passages are split automatically by a phoneme budget, and generation that reaches the limit without finishing raises an error. AMD has only one user measurement so far, and Intel has no measurement yet.

<a id="performance"></a>

## Performance

RTF means synthesis time divided by output audio duration; lower is faster. For example, **RTF 0.2 means roughly five seconds of audio per second of synthesis**.

| GPU | Route | Short dialogue RTF ↓ | Long Chinese text RTF ↓ | Peak dedicated VRAM |
|---|---|---:|---:|---:|
| NVIDIA RTX 4060 Ti 16GB | CUDA | **0.1955** | **0.1648** | **about 0.78 GiB full synthesis; about 1.09 GiB streaming** |
| NVIDIA RTX 4060 Ti 16GB | DirectML | **0.2165** | **0.2637** | **about 1.98 GiB** |
| AMD Radeon RX 9070 XT (user measurement) | DirectML | **0.2** | — | **about 1.98 GiB** |

- **Precision**: CUDA uses PyTorch FP16, with the streaming endpoint's acoustic decoding (flow and vocoder) temporarily in FP32; DirectML uses ORT GPT + torch-directml SoVITS, both in FP32. The RTF in the table is for full synthesis.
- **Test environment**: the RTX 4060 Ti RTF used the bundled 小春音アミ model on Windows with an i5-12600KF and four CPU threads, measured on 2026-10-11. The AMD result comes from a user's measurement.
- **Short dialogue**: four built-in Japanese lines in one request, about 14 seconds of audio, timeline enabled and no inter-segment pause; two runs after a short warm-up, RTF averaged.
- **Long Chinese text**: 1,278 characters and about four minutes of audio, with the older Chinese prefix of that time (97 tokens), a 0.3-second inter-segment pause and the timeline disabled; three runs after a full-text warm-up, measured as total time divided by total audio duration.
- **Measured version**: the RTF values in the table predate the current per-language semantic references: the short dialogue (Japanese) had no semantic prefix then, and the long Chinese text used the older Chinese prefix above. DirectML feeds prefix tokens one at a time, so DirectML RTF for prefixed requests is higher in the current version; on CUDA the measured time to first audio is essentially unchanged.
- **VRAM**: the CUDA figures were measured with the current version on the RTX 4060 Ti across both characters, multilingual short lines and long Chinese lines, covering full synthesis and a resident streaming service. The NVIDIA and AMD DirectML figures are both user measurements of the process peak for non-streaming short lines. The figures are the dedicated VRAM peak of the whole TTS process, including model loading and synthesis; the CUDA test also used up to about 0.07 GiB of shared GPU memory. `—` means no result was provided.
- **Timing scope**: includes text processing, synthesis and returning the waveform to the CPU, but excludes model loading, HTTP, file writing and playback. Audio duration includes configured inter-segment pauses. Actual performance varies by device, language and text.

Use `/api/tts/benchmark` to measure short-dialogue RTF on the target device, and `/api/tts/status` to see which GPT backend and GPU are actually in use. Streaming also depends on first-chunk latency and player buffering.

<a id="custom"></a>

## Custom characters and development deployment

### Layout and resources

The package uses **Python 3.10**, with CUDA and DirectML in `Runtime/Python_CUDA/` and `Runtime/Python_DirectML/` and their dependencies kept separate. If you set up your own environment, [requirements-cuda-frontend.txt](requirements-cuda-frontend.txt), [requirements-directml.txt](requirements-directml.txt) and [requirements-chinese.txt](requirements-chinese.txt) contain **incremental dependencies** for the backends or frontends; use them with an already configured base inference environment.

| Resource | Purpose |
|---|---|
| GPT `.ckpt` and compatible SoVITS `.pth` | Semantic generation and character acoustic model |
| `.cache.pt` reference cache | Reference spectrum, speaker embedding and version metadata; see [prompt_cache.py](prompt_cache.py) for the structure |
| Chinese BERT, G2PW and language identification models | Local Chinese processing and language segmentation; paths and sources in [chinese-runtime-manifest.json](chinese-runtime-manifest.json) |
| NLTK resources, Japanese dictionary and frontend dependencies | English and Japanese text processing |
| `resources/semantic_prompts/` manifest and assets | Per-language semantic references, applied by GPT hash; character-bound references also check the SoVITS hash |
| GPT's adjacent `.ort-dml/` directory, DML only | Prefill / Decode ONNX graphs and hash manifest |
| `Runtime/Python_CUDA/` and `Runtime/Python_DirectML/` | Two isolated Python runtimes and inference dependencies |

Some frontend and semantic-reference assets load from `resources/` in this directory, and `--resources` does not redirect every frontend resource path. Keep their relative locations when deploying on your own or reorganizing the directory.

Load external models or caches only from trusted sources: PyTorch weight files may contain executable serialized content.

### Add your own character

Keep the existing `ami` and `tyc` entries in `resources/voices.json` and add your character using the structure below. `character` is an example custom ID; the models and references must be data you have the right to use:

```json
{
  "character": {
    "name": "My character",
    "gpt_path": "models/character/gpt.ckpt",
    "sovits_path": "models/character/sovits.pth",
    "default_reference_id": "default",
    "references": {
      "default": {
        "name": "Default",
        "cache_path": "avatars/character/references/default.cache.pt"
      }
    },
    "defaults": {
      "text_lang": "auto",
      "speed": 1.0,
      "top_k": 15,
      "top_p": 1.0,
      "temperature": 1.0,
      "noise_scale": 0.35,
      "pause_length": 0.3
    }
  }
}
```

Models and references can differ by character, or several characters can point to the same GPT weights. The service switches one active engine on demand, releasing the previous engine's model resources when it switches. Start the service as in the quick start, with `--avatar` set to your new character's ID.

### Build a reference cache

A reference cache is made from one reference recording; the runtime never reads the recording itself. Add the character to `voices.json` first, then run in a preparation environment:

```powershell
python tools/build_prompt_cache.py --voice character --audio reference.wav --text "Exact transcript of the recording" --language en --hubert path/to/chinese-hubert-base --sv path/to/pretrained_eres2netv2w24s4ep4.ckpt --sample check.wav
```

- Use a clean 3–10 second mono recording. `--text` must match the recording word for word, and `--language` is the recording's language (`zh`, `ja` or `en`).
- `--hubert` and `--sv` are upstream GPT-SoVITS pretrained models (`chinese-hubert-base/` and `sv/pretrained_eres2netv2w24s4ep4.ckpt` in [lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS)). They are only needed to build caches and are not distributed with the package.
- The cache is written to the reference's `cache_path` in `voices.json`; pass `--overwrite` to replace an existing file. Before writing, the script loads the new cache with the runtime engine and synthesizes a short sentence; `--sample` saves that audio for listening.
- For a character-specific Japanese semantic reference, generate one from this cache with `--prompt-cache` and `--character-only` of [build_semantic_prompt.py](tools/build_semantic_prompt.py). A longer recording gives a longer semantic reference and a slower first packet.

### Prepare DirectML GPT graphs

In the DirectML edition, the bundled GPT already includes working DirectML graphs. If you replace the GPT weights, export new graphs in a preparation environment with PyTorch, ONNX and ONNX Runtime:

```powershell
python tools/export_gpt_dml.py --gpt resources/models/character/gpt.ckpt
```

By default this creates an adjacent `gpt.ort-dml/` containing `prefill-p256.onnx`, `step.onnx` and `manifest.json`. The exporter checks the ONNX structure and compares fixed samples numerically on the CPU; the runtime verifies the graph and GPT weight hashes. After export, you can run the synthesis and acoustic comparison scripts on the target device to check the actual audio.

<a id="technical"></a>

## Technical design

Implementation details are in [docs/TECHNICAL.en.md](docs/TECHNICAL.en.md):

- [How streaming timelines work](docs/TECHNICAL.en.md#streaming-timeline): synthesis starts while GPT is still generating; the character timeline is taken directly from SoVITS's MRTE attention and mapped back to each character of the original text with a monotonic alignment.
- [Cross-language pronunciation, voice identity and speed](docs/TECHNICAL.en.md#cross-language): one native semantic reference per language, given to GPT only, reduces accents, drawn-out sounds, long silences and dropped words when a character speaks another language; the voice comes only from the character's own acoustic reference.
- [Backend choices and optimization](docs/TECHNICAL.en.md#backends): how CUDA, DirectML and CPU are each implemented, device matching on multi-GPU systems, and memory optimizations on the reference side and in the Chinese frontend.
- [Code navigation](docs/TECHNICAL.en.md#code-map) and [verification](docs/TECHNICAL.en.md#verification).

<a id="license"></a>

## License and third-party resources

This repository treats **original code, upstream code, character voice data and other third-party resources** separately:

| Content | Applicable terms |
|---|---|
| Original code added in this repository | Apache License 2.0; see [LICENSE](LICENSE) |
| GPT-SoVITS and GPT-SoVITS_minimal_inference code | Their own MIT / Apache-2.0 licenses still apply |
| The two characters' models, acoustic references, prompt caches, character names and voice data | Their official terms, character notices and provider terms still apply; this repository's Apache license grants no new authorization |
| Other third-party models, dictionaries, runtimes and resources | Their original licenses or terms still apply |

### Voice sources of the bundled characters

- **小春音アミ (Ami Koharune):** the local SoVITS voice model uses voice data from [Amitaro's Voice Material Studio](https://amitaro.net/) as training/source audio. Generated audio and related data must follow the site's official terms.
- **つくよみちゃん:** the local SoVITS voice model uses voice data from **つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV：夢前黎）**, or Tsukuyomi-chan Corpus Vol.1 (CV: Rei Yumesaki), as training/source audio. For the character license, corpus terms and credit requirements, the [official character license](https://tyc.rei-yumesaki.net/about/terms/), the [つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/) page and the [credit guidance](https://tyc.rei-yumesaki.net/about/terms/credit/) prevail.
- The semantic GPT shared by both characters is trained on 小春音アミ source material, and the Chinese and English semantic references come from other speakers (see [TTS_MODEL_AND_DATA_NOTICE.md](TTS_MODEL_AND_DATA_NOTICE.md)). The sources and terms of each SoVITS voice model stay the same; but the shared GPT is this repository's design, so つくよみちゃん audio generated with this repository must also follow the credit and use rules of あみたろの声素材工房 (see [Voice credits](#credits)).
- For the credit text and distribution requirements, see [Voice credits](#credits) and [Distributing the character voices in your own application](#redistribute).

### License files and notices

- **GPT-SoVITS:** MIT; see [GPT-SoVITS_LICENSE.txt](GPT-SoVITS_LICENSE.txt).
- **GPT-SoVITS_minimal_inference:** Apache-2.0; see [GPT-SoVITS_minimal_inference_LICENSE.txt](GPT-SoVITS_minimal_inference_LICENSE.txt).
- **Original code in this repository:** Apache-2.0; see [LICENSE](LICENSE). Unless a file states otherwise, code added in this repository is released under it; upstream and other third-party code keep their original licenses.
- Changes and dependency sources: [NOTICE.txt](NOTICE.txt), [TTS_ENGINE_THIRD_PARTY_NOTICE.md](TTS_ENGINE_THIRD_PARTY_NOTICE.md).
- Models and dictionaries: [TTS_MODEL_AND_DATA_NOTICE.md](TTS_MODEL_AND_DATA_NOTICE.md).
- The bundled character models and caches are additionally covered by [TTS_RUNTIME_NOTICE.md](TTS_RUNTIME_NOTICE.md) and the character notices: [小春音アミ](resources/avatars/ami/AMITARO_VOICE_NOTICE.md), [つくよみちゃん](resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md).

Character sources: [小春音アミ / Amitaro's Voice Material Studio](https://amitaro.net/) (© Amitaro's Voice Material Studio), [つくよみちゃん](https://tyc.rei-yumesaki.net/) (© Rei Yumesaki).

**Code and character voice data are licensed separately.** Follow each authorization when using or distributing character models, reference recordings, reference caches, character names or voice material.
