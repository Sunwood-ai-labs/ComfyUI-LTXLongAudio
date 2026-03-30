---
layout: home

hero:
  name: ComfyUI-LTXLongAudio
  text: 長尺音声チャンク、ネイティブループ、静止画動画プレビュー
  tagline: アップロード UI から最終 MP4 出力まで、ComfyUI の表面をネイティブなまま保ちます。
  image:
    src: /logo.svg
    alt: ComfyUI-LTXLongAudio ロゴ
  actions:
    - theme: brand
      text: はじめに
      link: /ja/guide/getting-started
    - theme: alt
      text: サンプルワークフロー
      link: /ja/guide/usage
    - theme: alt
      text: GitHub
      link: https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio

features:
  - title: ネイティブな LTX 表面
    details: ループ制御、チャンク計画、メディア結合、最終プレビューまでを外部ノードパックなしでまとめます。
  - title: App mode 対応スモーク
    details: 4 つの公開入力だけで使えるワークフローと、決定的なフレーム選択を含む静止画動画パスを同梱します。
  - title: 公開前 QA
    details: レイアウト契約、App mode メタデータ、Prompt API 実行をリポジトリ内スクリプトで確認できます。
---

## 同梱内容

- 音声アップロード、画像アップロード、バッチフレーム、フォルダ選択のためのネイティブ入力ノード
- セグメント計画、決定的フレーム選択、音声スライス、音声連結のための長尺音声ヘルパー
- 同梱 LTX グラフで使うネイティブループ制御と補助ノード置換
- `samples/workflows/` 配下のスモークワークフローと、`samples/input/` 配下の軽量サンプル画像群

## サンプルの見どころ

同梱ワークフローの App mode 入力は最小限です。

| App mode 入力 | 用途 |
| --- | --- |
| `Frames Folder` | チャンクごとの静止画を選ぶフォルダを指定します。 |
| `Source Audio Upload` | ComfyUI 上で元音声を直接アップロードします。 |
| `Segment Seconds` | チャンク長を指定します。既定値は 20 秒です。 |
| `Random Seed` | フレーム選択を再実行時も決定的に保ちます。 |

現在のワークフローは `3 groups / 9 nodes` として解析され、App mode の最終出力ノードは 1 つです。

## 検証入口

- `uv run pytest`
- `uv run python scripts/check_workflow_layout.py samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json --require-all-nodes-in-groups --require-app-mode`
- `uv run python scripts/run_comfyui_api_smoke.py --workflow samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json --comfy-root /path/to/ComfyUI`

::: tip
`run_comfyui_api_smoke.py` の追跡済み既定値は `ltx-demo-tone.wav` です。ローカルに長い `HOWL AT THE HAIRPIN2.wav` があれば、それを優先しつつ追跡済みウィジェット名へコピーして使用します。
:::
