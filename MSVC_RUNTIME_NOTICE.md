# Microsoft Visual C++ Runtime / 微软 C++ 运行库 / Microsoft C++ ランタイム

Version / 版本 / バージョン: **14.51.36247.0**, x64.

Copyright Microsoft Corporation. All rights reserved.

The runtime packages include unmodified Microsoft Visual C++ runtime DLLs beside
each bundled TTS Python executable. These libraries support local voice synthesis.
They are Microsoft components and are not covered by the GPT-SoVITS code license.
This application-local deployment does not install or replace Windows system DLLs.
Updates to these bundled copies are delivered with new runtime package releases.

运行包在两套 TTS Python 可执行文件旁同梱未经修改的微软 C++ 运行库，用于支持本地语音合成。
这些文件属于微软组件，不适用 GPT-SoVITS 的代码许可证。本地同梱不会安装或替换 Windows 系统 DLL；
这些随包副本随运行包的新版本更新。

ランタイムパッケージには、ローカル音声合成を支える未改変の Microsoft C++ ランタイム DLL が、
各 TTS 用 Python 実行ファイルと同じフォルダーに同梱されています。これらは Microsoft の
コンポーネントであり、GPT-SoVITS のコードライセンスの対象ではありません。
Windows のシステム DLL のインストールや置き換えは行いません。同梱版はランタイムパッケージの新しいリリースで更新します。

The original Microsoft runtime license is included at
`Redistributables/VC_Runtime_x64/MICROSOFT_LICENSE.docx`.
原始微软许可文件 / Microsoft の原文ライセンスは上記ファイルに同梱しています。

- Official runtime license: https://aka.ms/VCRedistLicense
- Official redistribution guidance: https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files
- Visual Studio redistribution list: https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution
- Original package: https://aka.ms/vc14/vc_redist.x64.exe

Provenance, exact file versions, and SHA-256 values are recorded in
`msvc-runtime-manifest.json`. Redistribution permissions derive from the applicable
Visual Studio license and Microsoft's redistribution list; the runtime end-user
license is not a standalone grant to redistribute it. No debug runtime is included.
