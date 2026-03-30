# 構成

## 長尺音声パイプライン

リポジトリ内の長尺音声パスは次のように明示されています。

1. `LTXAudioDuration` と `LTXLongAudioSegmentInfo` が元音声の長さからセグメントを計画する
2. `LTXBuildChunkedStillVideo` が音声をチャンク単位へ分割する
3. 各チャンクで選択フォルダから決定的に 1 枚のフレームを選ぶ
4. 内部のダミーセグメント描画で still-video 断片を作る
5. 画像バッチと音声片を 1 本へ戻す
6. `LTXVideoCombine` が最終 MP4 プレビューを mux する

フレーム数は LTX 系 downstream 前提に合わせ、8 フレーム単位へ量子化されます。

## ノード群

### 入力とステージング

- `LTXLoadAudioUpload`
- `LTXLoadImageUpload`
- `LTXLoadImages`
- `LTXBatchUploadedFrames`
- `LTXRepeatImageBatch`

### チャンク計画とメディア組み立て

- `LTXAudioDuration`
- `LTXLongAudioSegmentInfo`
- `LTXRandomImageIndex`
- `LTXAudioSlice`
- `LTXBuildChunkedStillVideo`
- `LTXDummyRenderSegment`
- `LTXAppendImageBatch`
- `LTXAppendAudio`
- `LTXEnsureImageBatch`
- `LTXEnsureAudio`
- `LTXAudioConcatenate`
- `LTXVideoCombine`

### ループ制御と補助ノード

- `LTXWhileLoopStart`, `LTXWhileLoopEnd`
- `LTXForLoopStart`, `LTXForLoopEnd`
- `LTXIfElse`
- `LTXCompare`
- `LTXSimpleMath`
- `LTXSimpleCalculator`
- `LTXIntConstant`
- `LTXIndexAnything`
- `LTXBatchAnything`
- `LTXSeedList`
- `LTXShowAnything`

### LTX ユーティリティ置換

- `LTXVAELoader`
- `LTXImageResize`
- `LTXChunkFeedForward`
- `LTXSamplingPreviewOverride`
- `LTXNormalizedAttentionGuidance`

## リポジトリ配置

```text
docs/                         VitePress docs と共有 SVG アセット
samples/input/                軽量なサンプル画像群と fallback 音声
samples/workflows/            スモークワークフロー JSON
scripts/check_workflow_layout.py
scripts/run_comfyui_api_smoke.py
tests/                        import と workflow の回帰テスト
nodes.py                      カスタムノード本体とマッピング
```
