# 使い方

## スモークワークフローの既定値

同梱ワークフローは `samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json` にあります。

現在のチェック済み前提は次のとおりです。

| 項目 | 値 |
| --- | --- |
| フレームフォルダ既定値 | `samples/input/frames_pool` |
| 音声ウィジェット既定値 | `ltx-demo-tone.wav` |
| App mode 入力 | `Frames Folder`, `Source Audio Upload`, `Segment Seconds`, `Random Seed` |
| App mode 出力ノード | `LTXVideoCombine` |

## サンプル資産

追跡対象のサンプルは軽量なものに絞っています。

- `samples/input/frames_pool` は `688x384` の still frame 群
- `samples/input/demo_frames` は `192x108` の小さなデバッグ frame 群
- `samples/input/ltx-demo-tone.wav` はスクリプト用の追跡済み fallback 音声

追跡済みの既定値は `ltx-demo-tone.wav` です。ローカルに長い `HOWL AT THE HAIRPIN2.wav` を置いている場合、API smoke はそれを優先しつつ、追跡済みアップロード名へコピーして利用します。

## 入力解決の挙動

入力系ノードは ComfyUI の input ディレクトリとリポジトリの sample ディレクトリを両方見ます。

- 音声 dropdown は runtime input と `samples/input` の両方を候補にできます
- 画像 dropdown も同じです
- フォルダ dropdown は ComfyUI input 配下と `samples/input/*` を列挙します
- upload 系 COMBO は App mode で扱いやすいように blank-first option を持ちます

## 公開前チェック

まず静的チェッカーを通します。

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

実 ComfyUI がある場合だけ、live smoke を回します。

```bash
uv run python scripts/run_comfyui_api_smoke.py \
  --workflow samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --comfy-root /path/to/ComfyUI
```
