# トラブルシュート

## App mode に古い入力が残る

カスタムノード更新後は ComfyUI バックエンドやデスクトップアプリを完全再起動してください。ホットリロードでは古い schema が残ることがあります。

## プレビューが短すぎる

同梱グラフは先頭チャンクだけでなく元音声の全長を描画する想定です。短く見える場合は、まず backend の stale state を疑ってください。

## dropdown に音声が出ない

入力解決の前提を確認してください。

- repository sample は folder 系では `samples/input/...` という形で見えます
- 対話 UI での音声利用は、アップロード済み音声か runtime input 側の可視ファイルに依存します
- smoke script は Prompt API 用に fallback 音声をステージできますが、通常 UI ではアップロードか実ファイルが必要です
- repo が `ComfyUI/custom_nodes` 配下に無い場合、smoke script は `--comfy-root` を明示しないと runtime path を解決できません

## `ffmpeg` が見つからない

最終 mux は `ffmpeg` 依存です。ComfyUI 起動前にランタイムから見えるようにするか、必要なら API smoke 側へ明示パスを渡してください。

## 公開前に回すもの

```bash
uv run pytest

uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

Prompt API smoke は実 ComfyUI 環境がある場合だけ実行してください。
