<h1 align="center">GPT-SoVITS Interactive</h1>

<p align="center">
  <a href="README.md">中文</a> ·
  <a href="README.en.md">English</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="docs/DISTRIBUTION.md">分发说明</a>
</p>

<p align="center"><strong>给游戏和桌宠用的本地 GPT-SoVITS 语音服务：边生成边播放，逐字给出时间线</strong></p>

<p align="center">
  <img src="docs/assets/interactive-banner.png" width="960" alt="GPT-SoVITS Interactive 横幅：为应用与游戏轻松接入角色语音，支持中日英三语、CUDA、DirectML、CPU、流式音频与字符时间线。图中为つくよみちゃん与小春音アミ。">
</p>

<p align="center">
  <a href="#desktopdaily">我们的游戏</a> ·
  <a href="#difference">和官方 API 的区别</a> ·
  <a href="#quick-start">快速开始</a> ·
  <a href="#voices">随包角色</a> ·
  <a href="#api">API</a> ·
  <a href="#scope">适用范围</a> ·
  <a href="#performance">性能</a> ·
  <a href="docs/TECHNICAL.md">技术方案</a> ·
  <a href="#license">许可</a>
</p>

把台词发给本地服务，它一边生成一边把音频送回来，同时告诉你每个字在什么时候说出口。字幕高亮、表情和动作可以跟着同一个播放时钟走：不用等整句合成完，也不用事后再跑一遍对齐。

- **声音和文字一起流式返回**：每个音频块带着全局采样区间，以及这一块里新说出的字。
- **解压就能说三种语言**：Windows 整合包自带小春音アミ与つくよみちゃん两位角色、语言前端和 Python 环境，中文、日文、英文及混合文本都能直接生成。
- **不只 N 卡**：NVIDIA 用 CUDA，AMD / Intel 用 DirectML，也可以只用 CPU。

它是给应用调用的推理服务，不含训练和 WebUI。角色用上游 GPT-SoVITS（v2ProPlus）训练后放进来使用。

<a id="desktopdaily"></a>

## 我们的游戏：《桌面日常》

<!-- 游戏截图或动图：角色说话时，点头、手势和表情落在对应的字上。素材待定。 -->

**[《桌面日常》](https://store.steampowered.com/app/4724470/_/)** 是我们正在制作的 3D 桌宠，将在 Steam 免费推出。つくよみちゃん和小春音アミ住在你的桌面上：你工作、上网的时候，两位角色在透明窗口里待着、做自己的事，偶尔对你做出反应。

- **和角色聊天**：接入你自己的 API key 和兼容的大模型，角色用中文、日文或英文回答，由本仓库的 TTS 在本地念出来。
- **边说边演**：动作和表情由大模型随台词一起编排。借助本仓库的流式时间线，点头、手势和表情能落在台词里对应的那个字上，而且不用等整句生成完。

这套 TTS 就是为它的角色对话整理出来的，页首就是游戏里的两位角色。喜欢的话，欢迎在 Steam 加入愿望单，或者关注我们，获取开发进度：

- **[Steam 商店页（加入愿望单）](https://store.steampowered.com/app/4724470/_/)** · **[官网](https://www.desksideworks.com)**
- **[哔哩哔哩](https://space.bilibili.com/32459108)** · **[小黑盒](https://www.xiaoheihe.cn/app/user/profile/95841191)** · **[X / Twitter](https://x.com/DesksideWorks)**

<a id="difference"></a>

## 和 GPT-SoVITS 官方 API 有什么不同

模型结构、权重格式和合成方法沿用 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)，推理代码的直接基础是 [GPT-SoVITS_minimal_inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference)；自回归缓存、ONNX 图、I/O binding、流式推理和 API 等基础能力，上游也已经提供。本仓库做的是把它用进应用。下表官方一列中接口相关的几行，对照的是 2026 年 10 月上游 main 分支的 `api_v2.py`：

| | GPT-SoVITS 官方 | 本仓库 |
|---|---|---|
| 定位 | 训练、WebUI、推理一体的完整工具 | 只做推理，常驻在应用旁边 |
| 流式输出 | 音频流（`streaming_mode` 1–3，0 为不流式） | NDJSON：音频块 + 采样区间 + 逐字时间线 |
| 播放到哪个字 | 不返回 | 每个字的起止时间，和音频共用采样时钟 |
| 角色说其他语言 | GPT 条件来自请求里的参考录音 | 每种语言一份母语语义参考，音色仍来自角色本身 |
| 每次请求 | 带参考录音路径 | 参考特征已预计算，只传角色 ID |
| 服务管理 | 重启 / 退出、切换权重 | 角色切换、GPU 休眠与唤醒、请求串行、断流清理 |

<a id="quick-start"></a>

## 快速开始

### 下载

整合包按显卡分为两个版本，代码和角色相同，只差运行环境（CUDA 版不含 DirectML 专用的 GPT 图），选一个下载即可：

- **CUDA 版**（`…-win-x64-cuda`）：NVIDIA 显卡。
- **DirectML 版**（`…-win-x64-directml`）：AMD / Intel / NVIDIA 兼容显卡。

整合包切成多个分卷（`.7z.001`、`.7z.002`……）。下载同一版本的全部分卷，放在同一个文件夹，用 [7-Zip](https://www.7-zip.org/) 解压 `.7z.001` 即可。两个版本都可以用 CPU 推理。GPU 推理需要安装对应的显卡驱动，Python 与推理依赖使用包内环境。

### 1. 启动服务

在解压出的目录（本 README 所在目录）打开 PowerShell，按显卡选一条命令。命令会生成并输出本次服务的访问 token，请保存，调用时要用。

**NVIDIA CUDA：**

```powershell
$env:GSV_TTS_TOKEN = [guid]::NewGuid().ToString('N')
Write-Output $env:GSV_TTS_TOKEN
& "./Runtime/Python_CUDA/python.exe" api_server.py --device cuda --cpu_threads 4 --avatar ami --reference default --resources resources --port 8001
```

**DirectML（AMD / Intel / NVIDIA 兼容显卡）：**

```powershell
$env:GSV_TTS_TOKEN = [guid]::NewGuid().ToString('N')
Write-Output $env:GSV_TTS_TOKEN
& "./Runtime/Python_DirectML/python.exe" api_server.py --device dml --cpu_threads 4 --avatar ami --reference default --resources resources --port 8001
```

- 使用 CPU 时，把 `--device cuda` 或 `--device dml` 改为 `--device cpu`；`--device auto` 按当前环境可用的后端选择。
- `--avatar tyc` 用另一位角色启动，也可以在之后的请求里选择角色。
- `--device_name` 指定设备名称，`--voices_config` 指定角色配置。

保留服务终端，等待输出 `GSV_TTS_STARTUP` JSON，其中 `"ok": true` 表示已就绪。启动失败时也会输出这一行，此时 `ok` 为 `false`，原因见 `code` 和 `message`（有时还有 `detail`）。服务只监听本机 `127.0.0.1`，端口默认 8001（`--port` 可改），所有请求使用 `Authorization: Bearer <token>` 鉴权。

### 2. 生成第一段语音

另开一个 PowerShell，把占位符换成服务终端输出的 token，音频会保存为当前目录下的 `speech.wav`：

```powershell
$ttsHeaders = @{ Authorization = 'Bearer <服务终端输出的 token>' }
$ttsBody = @{
    text = '你好，今天想聊些什么？'
    modelId = 'ami'
    textLang = 'zh'
} | ConvertTo-Json -Compress
Invoke-WebRequest -Uri 'http://127.0.0.1:8001/api/tts/predict' -Method Post -Headers $ttsHeaders -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($ttsBody)) -OutFile 'speech.wav'
```

把 `modelId` 改为 `tyc` 即可使用つくよみちゃん。`textLang` 支持 `zh`、`ja`、`en` 和自动识别的 `auto`；换一段文本就能试其他语言或混合对话。

### 3. 边收边播：流式输出

下面的客户端只依赖 Python 标准库，可以用包内 Python 运行。运行前在客户端终端设置 `$env:GSV_TTS_TOKEN = '<服务终端输出的 token>'`。它逐块保存 PCM 并打印时间线；把 `pcm` 交给应用的播放队列，就能边接收边播放。输出的 `.f32` 文件没有 WAV 头。

```python
import base64
import json
import os
import urllib.request

payload = {
    "text": "今天我们一起喝杯茶，然后聊一聊最近发生的事情吧。",
    "modelId": "ami",
    "textLang": "zh",
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

某次运行打印的内容如下（RTX 4060 Ti、CUDA；采样有随机性，每次的数值略有不同）。每个 `chunk` 都是一段可以立刻播放的 PCM，下面缩进的是这一块里新说出的字：序号、字、开始和结束的秒数。

```text
chunk 1 0 15360
    0 今 0.08 0.26
    1 天 0.26 0.40
    2 我 0.42 0.54
chunk 2 15360 30720
    3 们 0.54 0.60
    4 一 0.60 0.72
    5 起 0.72 0.86
    6 喝 0.86 0.98
chunk 3 30720 46080
    7 杯 1.00 1.14
    8 茶 1.14 1.46
chunk 4 46080 61440
    9 ， 1.46 1.80
    10 然 1.80 1.96
chunk 5 61440 134400
    11 后 1.96 2.10
    12 聊 2.10 2.30
    13 一 2.30 2.40
    14 聊 2.40 2.64
    15 最 2.82 3.00
    16 近 3.02 3.20
    17 发 3.20 3.36
    18 生 3.36 3.52
    19 的 3.52 3.60
    20 事 3.60 3.76
    21 情 3.76 3.94
    22 吧 3.94 4.08
    23 。 4.08 4.20
```

<a id="voices"></a>

## 随包角色与音声素材

| 角色 | 调用 ID | 可生成的语言 | 默认参考 |
|---|---|---|---|
| 小春音アミ（Ami Koharune） | `ami` | 中文、日文、英文及混合文本 | `default`，已预计算 |
| つくよみちゃん | `tyc` | 中文、日文、英文及混合文本 | `default`，已预计算 |

两位角色已在 [voices.json](resources/voices.json) 中配置完成，共用 GPT 语义模型，分别使用自己的 SoVITS 权重与声学参考缓存。各语言的语义参考已随包准备，DirectML 版还附带 DirectML 所需的 GPT 图。请求中设置 `modelId` 即可选择角色；添加自有角色的方式见[后文](#custom)。

### 两种音声素材适合做什么？

随包的两种音声都适合用作本地角色对话、桌宠回应、游戏 NPC 台词、字幕同步演出和原型开发的角色声线。它们已经完成模型、声学参考和运行配置的整合，使用者可以先通过 API 试听三语和混合文本，再决定是否用于自己的应用。

如果你正在寻找**有条件免费商用的二次元声线**，小春音アミ和つくよみちゃん可以作为值得了解的方向：两者都有公开的声音来源和使用规约，适合在遵守署名、角色使用范围、内容限制和再分发限制的前提下用于项目。它们不是“拿到模型即可无条件商用”的素材；发布作品、商业项目或包含角色视觉素材时，请先阅读各自的官方条款和本仓库随附的说明：

- [小春音アミ音声说明](resources/avatars/ami/AMITARO_VOICE_NOTICE.md) · [官方规约](https://amitaro.net/voice/voice_rule/)
- [つくよみちゃん音声说明](resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md) · [角色许可](https://tyc.rei-yumesaki.net/about/terms/) · [语料规约](https://tyc.rei-yumesaki.net/material/corpus/)

角色声音数据、参考缓存和生成音频的许可范围与本仓库代码许可分开适用，具体规则见文档末尾的[许可说明](#license)。

<a id="credits"></a>

### 声音署名

本仓库公开了用两位角色的声音训练的模型，按[つくよみちゃんコーパス规约](https://tyc.rei-yumesaki.net/material/corpus/)和[あみたろの声素材工房的规约](https://amitaro.net/voice/faq/)，须在醒目位置写明：

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

- 本仓库里的つくよみちゃん，合成时也用到了其他说话人的数据。这是本仓库自己的做法，不是つくよみちゃん官方的情况：两位角色共用的语义 GPT 用小春音アミ的素材训练；中文和英文的发音参考分别来自 Qwen3-TTS 生成的一句中文和 CMU ARCTIC 的一句英文录音。音色来自つくよみちゃん自己的 SoVITS 模型和参考。
- 两位角色的声音模型和横幅里的 3D 模型都不是素材方制作或认可的官方作品。
- 生成音频的署名按两边各自的规约：
  - 小春音アミ：作品里必须署名，例如「音声合成モデルに、あみたろの声素材工房（<https://amitaro.net/>）の音声データを使用しています」，用途也只能在あみたろ规约范围内（全年龄等）。
  - つくよみちゃん：她这边不要求给生成音频署名（语料规约对音声合成软件的输出音频设了特例）；画面里出现她的立绘、3D 模型等角色素材时，按[角色许可](https://tyc.rei-yumesaki.net/about/terms/)署名。不过本仓库让她和小春音アミ共用以アミ素材训练的 GPT（这是本仓库的做法，不是官方的），所以用本仓库生成的つくよみちゃん音频，同样要按上面あみたろ那条署名、守她的用途限制。

<a id="redistribute"></a>

### 在自己的应用里分发角色声音

把随包的つくよみちゃん模型打包进你的游戏或应用，就等于你也公开了一个能用她声质的软件（语料规约中的"③声質を使用した音声合成ソフト等を公開する"），需要自己满足下面几条。这里只是摘要，以[官方规约](https://tyc.rei-yumesaki.net/material/corpus/)为准：

- **署名**：发布前给夢前黎发邮件商定署名方式；或者把官方给的署名文字放在醒目位置、用足够大的字号（收费软件还要让用户在付款前看到）。官方按"是否要求用户给生成音频署名"准备了三份模板：[必须](https://tyc.rei-yumesaki.net/wp-content/uploads/tsukuyomi-chan_credit_template1.html)、[任意](https://tyc.rei-yumesaki.net/wp-content/uploads/tsukuyomi-chan_credit_template2.html)、[不需要](https://tyc.rei-yumesaki.net/wp-content/uploads/tsukuyomi-chan_credit_template3.html)。
- **禁止事项**：你的使用条款里要禁止上面四种用途，不写的话要先找夢前黎商量。
- **混合说明**：随包的つくよみちゃん用到了其他说话人的数据（见上），你的说明里也要写明。
- **改动和再分发**：如果允许别人改动或再分发你的软件（包括只拿部分代码或只拿声音模型），要让对方承诺：来自つくよみちゃんコーパス的部分遵守语料规约。这条要求也要继续传给下游。
- **给其他角色用**：要让她的声质给つくよみちゃん以外的角色当声音（比如你自己的角色、会话 AI），须事先取得夢前黎的许可。[官方公开的过往答复](https://tyc.rei-yumesaki.net/material/corpus/conversational-ai/)要求让人清楚知道是其他角色在用她的声质说话，给出的一种做法是在画面上持续显示署名，做不到时再和她商量。只是拿生成的音频给其他角色配音，则可以直接用。

小春音アミ的声音按あみたろの声素材工房的[规约](https://amitaro.net/voice/voice_rule/)和[常见问题](https://amitaro.net/voice/faq/)：

- **事先商量**：把模型打包进软件或应用，或者以其他方式公开分发，都要事先找あみたろの声素材工房商量。
- **免费**：只用她的声音做的模型只能免费分发；不能在软件之外另设付费功能来解锁这个声音（和あみたろ分成的除外）；也不能登记到让第三方使用声音模型的服务里。
- **附上规约**：分发时附上规约链接，或者附上她的 Readme。

<a id="api"></a>

## API

| 方法与路径 | 返回 / 行为 |
|---|---|
| `POST /api/tts/predict` | 完整 WAV |
| `POST /api/tts/predict-timeline` | JSON：Base64 WAV、字符时间线、采样率与时长 |
| `POST /api/tts/predict-stream` | NDJSON：增量 PCM 与时间线 |
| `GET /api/tts/models` | 可用角色与参考配置 |
| `POST /api/tts/switch-model` | 切换角色 / 参考 |
| `GET /api/tts/status` | 当前状态、后端、设备与启动阶段耗时 |
| `POST /api/tts/hibernate` | 释放引擎持有的 GPU 模型 / 会话资源，保留可恢复状态 |
| `POST /api/tts/wake` | 唤醒或加载指定角色 |
| `POST /api/tts/benchmark` | 内置短对话测试 |
| `POST /v1/audio/speech` | 接受 `input` / `voice` 的 WAV 接口 |
| `GET /v1/models` | OpenAI 兼容的角色模型列表（需要鉴权） |
| `GET /v1/voices` | 角色配置列表（需要鉴权） |
| `POST /v1/voices/reload` | 重新加载角色配置（需要鉴权） |

`/v1/audio/speech` 的输出格式固定为 WAV；需要边收边播时使用 `/api/tts/predict-stream`。

### 合成请求

三个 `/api/tts/predict*` 端点共用主要字段：

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

- `modelId` 用 `ami` 或 `tyc` 选择随包角色；省略角色与参考时使用当前活动配置。`textLang` 可用 `auto`、`zh`、`ja`、`en`。
- `speed` 调整语速，`1.0` 是模型生成的原始节奏，较大值加快、较小值减慢，三种语言和流式输出都支持。
- `chunkLength` 只用于流式接口，表示 GPT 语义 token 的产出块长，HTTP 层默认 24、限制在 10–100。它**不是音频毫秒数，也不是固定的 PCM 包大小**；首块延迟还取决于前瞻、解码上下文和设备速度。
- 段间停顿从角色的 `defaults.pause_length` 读取。
- 旧请求字段 `g2pPriorityMode`、`languages`、`generateTimeline` 仍可接收，但不影响行为：返回形式由端点决定，语言由 `textLang` 指定。

### 流式事件

响应类型为 `application/x-ndjson`，每行一个 JSON 对象，事件顺序为 `start → chunk… → done`。流开始以后出错，会发出 `error` 事件；鉴权失败、文本为空、角色不存在、模型加载失败等发生在流开始之前的错误，直接返回 HTTP 错误状态码，没有 NDJSON 事件。

| `chunk` 字段 | 含义 |
|---|---|
| `audio` | Base64 编码的 **小端 float32、单声道原始 PCM**，没有 WAV 头 |
| `audio_start_sample` / `audio_end_sample` | 全局采样区间，左闭右开；按顺序应连续 |
| `sample_rate` / `channels` | 当前流式实现为 32000 / 1 |
| `chunk_index` | 从 1 开始的块编号 |
| `timestamps` | 本块新交付的字符时间戳，可以为空 |
| `gpt_finished` | 当前分句的 GPT 是否结束；不表示整个请求已经结束 |
| `is_final` | 音频块为 false，完成由 `done` 事件确认 |
| `server_first_chunk_latency_seconds` | 服务端从开始生成到发出第一块用了多少秒 |

`start` 事件给出 `sample_rate` 和 `channels`。`done` 事件给出总样本数 `audio_end_sample`、块数 `chunk_count`，以及服务端统计的首块延迟、总耗时和音频时长（秒）。`error` 事件的 `error` 字段是错误说明。

时间线条目包含 `char_idx`、`char`、`start`、`end`、`phones`、`ph_start`、`ph_end` 和 `approximate`：

- `start` / `end` 的单位是秒，以整个请求音频的起点为零。第一个字的 `start` 一般大于零，相邻的字之间也可以有空档：没有人声的时间不属于任何字，除非它是某个标点的停顿。
- 每个字符按顺序交付一次，已经发出的条目不会改写。被规范化去掉的字符（例如 `……` 只读作一个停顿）也有条目，时长为零。只有分句时去掉的空白没有条目：文本开头和结尾的空白，以及分开合成的两句之间的空白。
- `char_idx` 是原始文本的 **Unicode code point 索引**；C# / JavaScript 接入时不能直接当作 UTF-16 下标，尤其是有 emoji 或扩展字符时。

上面那句话另一次运行的第一个音频块和结束事件如下（`audio` 已截短，实际每行是一个不换行的 JSON）：

```json
{
  "type": "chunk",
  "audio": "I8jjOHvooji40iY4…",
  "audio_start_sample": 0,
  "audio_end_sample": 15360,
  "sample_rate": 32000,
  "channels": 1,
  "is_final": false,
  "chunk_index": 1,
  "timestamps": [
    {"char_idx": 0, "ph_start": 0, "ph_end": 2, "phones": ["j", "in1"], "start": 0.08, "end": 0.24, "char": "今", "approximate": true},
    {"char_idx": 1, "ph_start": 2, "ph_end": 4, "phones": ["t", "ian1"], "start": 0.24, "end": 0.4, "char": "天", "approximate": true},
    {"char_idx": 2, "ph_start": 4, "ph_end": 6, "phones": ["w", "o3"], "start": 0.4, "end": 0.52, "char": "我", "approximate": true}
  ],
  "gpt_finished": false,
  "server_first_chunk_latency_seconds": 0.6759327999316156
}
```

```json
{
  "type": "done",
  "sample_rate": 32000,
  "channels": 1,
  "audio_end_sample": 143360,
  "is_final": true,
  "chunk_count": 6,
  "server_first_chunk_latency_seconds": 0.6759327999316156,
  "server_total_seconds": 1.4494603999191895,
  "audio_duration_seconds": 4.48
}
```

HTTP 流式端点总是计算并返回时间线。直接调用 Python 引擎的 `LatentStream.stream(..., timeline=False)` 可以关闭这项附加输出。时间线怎样从同一次推理里得到，见[技术方案](docs/TECHNICAL.md#streaming-timeline)。

### 服务生命周期

- **请求串行执行。** 会改变引擎状态的 POST 操作依次执行，锁一直持有到流式响应结束；角色切换、休眠和后续合成不会在前一个流还在用模型时抢先修改模型。客户端断开后，服务等当前计算步骤完成，再关闭生成器、恢复精度并放行后续操作。
- **查询不唤醒。** `/api/tts/status` 只读取状态。
- **休眠释放 GPU 资源。** PyTorch 模型与缓存移回 CPU，ORT DirectML 会话被释放；进程还在时，CUDA 上下文、驱动和库仍可能占用部分显存。

<a id="integration"></a>

### 接入建议

[《桌面日常》](#desktopdaily)用流式时间线驱动角色的演出，一段时间不说话就让 TTS 休眠、释放显存，下次开口前再唤醒。游戏的播放器按以下规则接入，也可作为其他应用的参考：

- **收到音频即可入队播放**，音频线程只消费 PCM 队列；接收端检查采样格式、样本区间是否连续，以及完成时的总样本数。
- **缺数据时暂停内容时钟。** 播放设备为维持输出补入的静音不计入台词进度，字幕、表情和动作不会在等下一块时提前推进；合成结果自己的停顿照常计入。
- **收到 `done` 后继续播完队列。** 生成结束和播放结束分开判断，最后一个真实样本播完才结束演出，保留尾音。
- **打断时同时取消请求和旧的播放队列**：停止旧音频、丢弃未播的 PCM，并隔离旧请求的延迟回调。
- **用实际播放的采样位置除以 `sample_rate` 驱动字幕和演出。** 一个字可能跨越多个音频块，时间线也可能稍晚到达，播放器要留一点缓冲。
- **按实际语言做首次准备。** 中文、日文、英文的前端在第一次用到时才加载。用户选定语音语言后，用该语言的短句走一次真实的流式接口（显式传 `textLang`），收到音频块和 `done` 后再开放正式合成，并把这段等待显示为准备状态。准备状态按进程记录：休眠唤醒后仍有效，服务重启后要重新准备。

另外，HTTP 状态码不是 2xx、收到 `error`，或者没收到 `done` 就断开的流，都应按失败处理，不要把只读出前半句的音频当成完整结果。

<a id="scope"></a>

## 适用范围与限制

- **模型版本**：当前支持 GPT-SoVITS **v2ProPlus**（32 kHz、25 Hz 语义帧率）。句内流式解码会检查模型结构，接入其他版本需要适配解码参数。
- **不含训练**：没有训练代码、WebUI 和微调工具。自有角色需要先用上游 GPT-SoVITS 训练，再按[自定义角色](#custom)接入。
- **平台**：整合包只有 Windows x64 版，也只在 Windows 上验证过。源码目录不含模型和运行环境，需要从同版本整合包补齐，见[分发说明](docs/DISTRIBUTION.md)。
- **一个应用独占**：服务只监听本机，请求需要 token；同一时间只执行一个推理任务，适合由一个应用统一管理角色合成与切换。
- **时间线是近似对齐**：由 SoVITS 的注意力推导，适合字幕高亮、文字推进和粗略的动作同步；精确口型或音素标注需要另做对齐。日文汉字词和英文单词内部的字符边界是近似的。
- **语义参考的适用范围**：只用于单一语言的请求（只夹了日文词的中文按中文处理）；其他中日英混合请求不使用语义参考。
- **DirectML**：需要兼容的 Windows GPU 与驱动。GPT 图的容量固定（256 个文本位置、512 个 KV 位置），长段按音素预算自动拆分；生成到上限仍未结束会报错。AMD 目前只有一份用户实测，Intel 还没有实测数据。

<a id="performance"></a>

## 性能

RTF = 合成耗时 / 输出音频时长，越低越快；例如 **RTF 0.2 表示约 1 秒生成 5 秒音频**。

| 显卡 | 推理路线 | 短对话 RTF ↓ | 中文长文 RTF ↓ | 专用显存峰值参考 |
|---|---|---:|---:|---:|
| NVIDIA RTX 4060 Ti 16GB | CUDA | **0.1955** | **0.1648** | **完整合成约 0.78 GiB；流式约 1.09 GiB** |
| NVIDIA RTX 4060 Ti 16GB | DirectML | **0.2165** | **0.2637** | **约 1.98 GiB** |
| AMD Radeon RX 9070 XT（用户实测） | DirectML | **0.2** | — | **约 1.98 GiB** |

- **精度**：CUDA 使用 PyTorch FP16，流式接口的声学解码（flow 和声码器）临时改用 FP32；DirectML 使用 ORT GPT + torch-directml SoVITS，均为 FP32。表中 RTF 测的是完整合成。
- **测试环境**：RTX 4060 Ti 的 RTF 使用随包小春音アミ模型，Windows、i5-12600KF、CPU 线程数 4，测试日期 2026-10-11；AMD 成绩来自用户实测反馈。
- **短对话**：内置四段日语作为一次请求，约 14 秒音频，开启时间线、段间停顿为 0；短句预热后测两轮，取 RTF 算术平均。
- **中文长文**：1,278 字符，约 4 分钟音频，启用当时的旧版中文前缀（97 tokens）、段间停顿 0.3 秒、关闭时间线；全文预热后测三轮，以总耗时 / 总音频时长计算。
- **测量版本**：表中 RTF 测于现行的各语言语义参考加入之前：当时短对话（日语）没有语义前缀，中文长文用的是上面那个旧版中文前缀。DirectML 需要逐个送入前缀 token，现行版本带前缀请求的 DirectML RTF 会高于表中数值；CUDA 上实测的首包延迟基本不变。
- **显存**：CUDA 数值为当前版本在 RTX 4060 Ti 上对两位角色、多语短句和中文长句的实测，分别覆盖完整合成与流式常驻服务；NVIDIA 与 AMD 的 DirectML 数值均为用户实测的非流式短句进程峰值。统计的是整个 TTS 进程的专用显存峰值，包含模型加载与合成；CUDA 测试另有约 0.07 GiB 的共享 GPU 内存峰值。`—` 表示未提供该项数据。
- **计时范围**：包含文本处理、合成和 CPU 波形返回，不含模型加载、HTTP、写盘与播放；音频时长包含设置的段间停顿。实际性能随设备、语言和文本变化。

可以用 `/api/tts/benchmark` 在目标设备上测短对话 RTF，用 `/api/tts/status` 查看实际使用的 GPT 后端和显卡。流式体验还取决于首个音频块的延迟和播放器缓冲。

<a id="custom"></a>

## 自定义角色与开发部署

### 目录与资源

随包使用 **Python 3.10**，CUDA 与 DirectML 分别位于 `Runtime/Python_CUDA/` 和 `Runtime/Python_DirectML/`，依赖彼此隔离。自行配置环境时，[requirements-cuda-frontend.txt](requirements-cuda-frontend.txt)、[requirements-directml.txt](requirements-directml.txt) 和 [requirements-chinese.txt](requirements-chinese.txt) 提供后端或前端的**增量依赖**，需要配合已配置好的基础推理环境使用。

| 资源 | 用途 |
|---|---|
| GPT `.ckpt` 与兼容的 SoVITS `.pth` | 语义生成与角色声学模型 |
| `.cache.pt` 参考缓存 | 参考频谱、说话人嵌入和版本元数据；结构见 [prompt_cache.py](prompt_cache.py) |
| 中文 BERT、G2PW、语言识别模型 | 本地中文处理与语言分段，路径和来源见 [chinese-runtime-manifest.json](chinese-runtime-manifest.json) |
| NLTK 资源、日语字典与前端依赖 | 英文与日文文本处理 |
| `resources/semantic_prompts/` 清单及对应资源 | 各语言的语义参考；通过 GPT 哈希判断是否应用，绑定角色的参考另核对 SoVITS 哈希 |
| GPT 同名 `.ort-dml/` 目录，仅 DML 需要 | Prefill / Decode ONNX 图及哈希清单 |
| `Runtime/Python_CUDA/` 与 `Runtime/Python_DirectML/` | 两套独立的 Python 运行环境与推理依赖 |

部分前端与语义参考资源从本目录的 `resources/` 加载，`--resources` 不会重定向所有前端资源路径。自行部署或重新组织目录时，请保留这些资源的相对位置。

加载外部模型或缓存时，只使用可信来源的文件：PyTorch 权重文件可能包含可执行的序列化内容。

### 添加自有角色

在 `resources/voices.json` 中保留现有的 `ami`、`tyc` 配置，按下面的结构添加自己的角色。`character` 是自定义角色 ID 的示例，模型与参考需要是你有权使用的数据：

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

模型和参考可以按角色分别配置，也可以让多个角色指向同一份 GPT 权重。服务按需切换一个活动引擎，切换时释放上一个引擎的模型资源。启动方式与"快速开始"相同，把 `--avatar` 改为新角色的 ID 即可。

### 制作参考缓存

参考缓存由一条参考录音生成，运行时不再读取录音本身。先在 `voices.json` 中添加角色条目，再在资源准备环境中运行：

```powershell
python tools/build_prompt_cache.py --voice character --audio reference.wav --text "录音的逐字文本" --language ja --hubert path/to/chinese-hubert-base --sv path/to/pretrained_eres2netv2w24s4ep4.ckpt --sample check.wav
```

- 录音使用 3–10 秒、干净的单声道语音。`--text` 必须与录音逐字一致，`--language` 是录音的语言（`zh`、`ja`、`en`）。
- `--hubert` 与 `--sv` 是上游 GPT-SoVITS 的预训练模型（[lj1995/GPT-SoVITS](https://huggingface.co/lj1995/GPT-SoVITS) 中的 `chinese-hubert-base/` 与 `sv/pretrained_eres2netv2w24s4ep4.ckpt`），只在制作缓存时使用，不随整合包分发。
- 缓存写入该参考在 `voices.json` 中的 `cache_path`，替换已有文件需要加 `--overwrite`。写入前，脚本用运行时引擎加载新缓存并合成一句短句；`--sample` 保存这句音频以供试听。
- 需要角色专属的日文语义参考时，可以再用 [build_semantic_prompt.py](tools/build_semantic_prompt.py) 的 `--prompt-cache` 和 `--character-only` 从这份缓存生成。录音越长，语义参考越长，首包越慢。

### 准备 DirectML GPT 图

DirectML 版整合包里的随包 GPT 已附带可用的 DirectML 图。更换 GPT 权重后，需要在具备 PyTorch、ONNX 和 ONNX Runtime 的资源准备环境中重新导出：

```powershell
python tools/export_gpt_dml.py --gpt resources/models/character/gpt.ckpt
```

默认在相邻位置生成 `gpt.ort-dml/`，包含 `prefill-p256.onnx`、`step.onnx` 和 `manifest.json`。导出脚本会做 ONNX 结构检查和固定样例的 CPU 数值对照，运行时核对图文件与 GPT 权重的哈希。导出后可以在目标设备上运行合成与声学对照脚本，检查实际音频输出。

<a id="technical"></a>

## 技术方案

实现细节写在 [docs/TECHNICAL.md](docs/TECHNICAL.md)：

- [流式时间线是怎样工作的](docs/TECHNICAL.md#streaming-timeline)：GPT 生成过程中就开始合成；字符时间线直接取自 SoVITS 的 MRTE 注意力，用单调对齐还原到原文的每个字。
- [跨语言发音、角色音色与语速](docs/TECHNICAL.md#cross-language)：每种语言一份母语语义参考只提供给 GPT，减少说外语时的口音以及拖长、长静音和漏读；音色只来自角色自己的声学参考。
- [后端与优化取舍](docs/TECHNICAL.md#backends)：CUDA、DirectML、CPU 各自的实现，多显卡的设备匹配，以及参考侧和中文前端的内存优化。
- [代码导航](docs/TECHNICAL.md#code-map)与[验证入口](docs/TECHNICAL.md#verification)。

<a id="license"></a>

## 许可与第三方资源

本仓库把**原创代码、上游代码、角色音声数据和其他第三方资源**分开处理：

| 内容 | 适用范围 |
|---|---|
| 本仓库新增的原创代码 | Apache License 2.0，见 [LICENSE](LICENSE) |
| GPT-SoVITS 与 GPT-SoVITS_minimal_inference 代码 | 继续适用各自的 MIT / Apache-2.0 许可证 |
| 两位角色的模型、声学参考、prompt cache、角色名称和声音数据 | 继续适用各自的官方规约、角色声明和素材方条款，不因本仓库的 Apache 许可获得新授权 |
| 其他第三方模型、字典、运行库和资源 | 继续适用各自的原始许可证或使用条款 |

### 随包角色的声音来源

- **小春音アミ（Ami Koharune）**：本地 SoVITS 声音模型使用 [Amitaro's Voice Material Studio](https://amitaro.net/) 的声音数据作为训练/来源声音；生成音频和相关数据需遵守该站官方规约。
- **つくよみちゃん**：本地 SoVITS 声音模型使用 **つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠 / CV：夢前黎）** 的声音数据作为训练/来源声音。相关角色许可、语料规约和署名要求以 [官方角色许可](https://tyc.rei-yumesaki.net/about/terms/)、[つくよみちゃんコーパス](https://tyc.rei-yumesaki.net/material/corpus/) 和 [署名说明](https://tyc.rei-yumesaki.net/about/terms/credit/) 为准。
- 两位角色共用的语义 GPT 使用小春音アミ来源素材训练，中文、英文的语义参考来自其他说话人（见 [TTS_MODEL_AND_DATA_NOTICE.md](TTS_MODEL_AND_DATA_NOTICE.md)）。各自 SoVITS 声音模型的来源和条款不变；但共用 GPT 是本仓库的做法，所以用本仓库生成的つくよみちゃん音频也要遵守あみたろの声素材工房的署名和用途规定（见[声音署名](#credits)）。
- 署名文字和分发要求见[声音署名](#credits)与[在自己的应用里分发角色声音](#redistribute)。

### 许可文件与说明

- **GPT-SoVITS**：MIT，见 [GPT-SoVITS_LICENSE.txt](GPT-SoVITS_LICENSE.txt)。
- **GPT-SoVITS_minimal_inference**：Apache-2.0，见 [GPT-SoVITS_minimal_inference_LICENSE.txt](GPT-SoVITS_minimal_inference_LICENSE.txt)。
- **本仓库原创代码**：Apache-2.0，见 [LICENSE](LICENSE)。除文件内另有说明外，本仓库新增的代码按此发布；上游代码和其他第三方代码不因本文件而改变原许可证。
- 修改说明与依赖来源见 [NOTICE.txt](NOTICE.txt)、[TTS_ENGINE_THIRD_PARTY_NOTICE.md](TTS_ENGINE_THIRD_PARTY_NOTICE.md)。
- 模型和字典来源见 [TTS_MODEL_AND_DATA_NOTICE.md](TTS_MODEL_AND_DATA_NOTICE.md)。
- 随包的角色模型与缓存另适用 [TTS_RUNTIME_NOTICE.md](TTS_RUNTIME_NOTICE.md) 及对应的角色声明：[小春音アミ](resources/avatars/ami/AMITARO_VOICE_NOTICE.md)、[つくよみちゃん](resources/avatars/tyc/TSUKUYOMICHAN_VOICE_NOTICE.md)。

角色来源：[小春音アミ / あみたろの声素材工房](https://amitaro.net/)（© Amitaro's Voice Material Studio）、[つくよみちゃん](https://tyc.rei-yumesaki.net/)（© Rei Yumesaki）。

**代码许可与角色声音数据许可分别适用。** 使用或分发角色模型、参考录音、参考缓存、角色名称及声音素材时，请遵循各自的授权。
