# はじめに

## 利用パス

このリポジトリは主に 2 つの流れを想定しています。

1. Google Colab やローカル ComfyUI の `ComfyUI/custom_nodes` に clone して使う
2. リポジトリ自体を QA ワークスペースとして開き、テストやレイアウト検査を回す

## ComfyUI への導入

```bash
cd /content/ComfyUI/custom_nodes
git clone https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio.git
uv pip install -r ComfyUI-LTXLongAudio/requirements.txt
```

その後、ComfyUI を再起動してネイティブ `LTX*` ノードの入力スキーマを更新してください。

## ローカル QA

開発者向けチェックは `uv` でそろえています。

```bash
uv run pytest

uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

## 実行前提

- 最終動画プレビューや API smoke には `ffmpeg` が必要です。
- 音声読込は `torchaudio` を使うため、ランタイム依存は `requirements.txt` に残しています。
- Prompt API smoke は到達可能な ComfyUI、本体の入出力ディレクトリ、利用可能なデータベースパスを前提にしています。

## 最初の確認手順

1. 同梱スモークワークフローを開く
2. フォルダ一覧に `samples/input/frames_pool` が見えることを確認する
3. `LoadAudio` から音声をアップロードする
4. 初回は 20 秒チャンクのまま動かす
5. `LTXVideoCombine` が出力パネルへプレビューを返すことを確認する
