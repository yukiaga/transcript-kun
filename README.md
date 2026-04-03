# transcript-kun

ローカル音声ファイル（M4A等）の文字起こし＋話者分離ツール。
[WhisperX](https://github.com/m-bain/whisperX)（OpenAI Whisper + pyannote-audio）をバックエンドに使用。

## 特徴

- M4A, MP3, WAV, FLAC, OGG, WebM, MP4 対応
- 話者分離（Speaker Diarization）対応
- TXT / SRT / JSON / TSV 4種類の出力形式
- クリーンアーキテクチャ設計（ドメイン層・アプリケーション層・インフラ層・インターフェース層）
- ポリシーによるガードレール（ファイル検証・設定検証）

## 前提条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（パッケージマネージャー）
- ffmpeg（音声ファイルの読み込みに必要）
- GPU (NVIDIA, VRAM 8GB+) 推奨。CPU でも動作可能
- Hugging Face アカウント（話者分離を使う場合）
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) の利用規約に同意が必要
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) の利用規約に同意が必要

## セットアップ

### uv のインストール

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### ffmpeg のインストール

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### プロジェクトのセットアップ

```bash
# リポジトリをクローン
git clone https://github.com/yukiaga/transcript-kun.git
cd transcript-kun

# 依存関係のインストール（仮想環境の作成も自動）
uv sync

# GPU 環境の場合（WhisperX を含む）
uv sync --extra gpu

# 開発用（テスト・リンター込み）
uv sync --dev
```

## 使い方

### 基本（話者分離なし）

```bash
uv run transcript-kun audio.m4a --no-diarize
```

### 話者分離あり（推奨）

```bash
# 環境変数に Hugging Face トークンを設定
export HF_TOKEN="hf_your_token_here"

# 実行
uv run transcript-kun audio.m4a
```

### GPU を使う場合

```bash
uv run transcript-kun audio.m4a -d cuda --compute-type float16
```

### 出力形式を指定

```bash
# SRT字幕形式
uv run transcript-kun audio.m4a -f srt --no-diarize

# JSON（プログラムで後処理しやすい）
uv run transcript-kun audio.m4a -f json --no-diarize

# TSV（スプレッドシートにインポート）
uv run transcript-kun audio.m4a -f tsv --no-diarize
```

### モデルサイズを変更

精度とメモリ/速度のトレードオフ：

```bash
# 高精度（デフォルト、VRAM 10GB+推奨）
uv run transcript-kun audio.m4a -m large-v3 --no-diarize

# 中程度（VRAM 5GB程度）
uv run transcript-kun audio.m4a -m medium --no-diarize

# 軽量（VRAM 2GB程度、精度は落ちる）
uv run transcript-kun audio.m4a -m small --no-diarize
```

### 話者数を指定（精度向上）

事前に話者の人数がわかっている場合：

```bash
uv run transcript-kun audio.m4a --min-speakers 2 --max-speakers 3
```

### python -m でも実行可能

```bash
uv run python -m transcript_kun audio.m4a --no-diarize
```

## 全オプション一覧

```
usage: transcript-kun [-h] [-o OUTPUT] [-f {json,srt,tsv,txt}]
                      [-m {base,large-v2,large-v3,medium,small,tiny}]
                      [-l LANGUAGE] [-d {cpu,cuda}]
                      [--compute-type {float16,float32,int8}]
                      [--batch-size BATCH_SIZE] [--no-diarize]
                      [--hf-token HF_TOKEN]
                      [--min-speakers MIN_SPEAKERS]
                      [--max-speakers MAX_SPEAKERS] [-v]
                      audio_file

positional arguments:
  audio_file            Path to the audio file

options:
  -o, --output          Output file path (default: <stem>_transcript.<fmt>)
  -f, --format          Output format: txt, srt, json, tsv (default: txt)
  -m, --model           Whisper model size (default: large-v3)
  -l, --language        Language code (default: ja)
  -d, --device          Compute device: cpu, cuda (default: cpu)
  --compute-type        Compute type: int8, float32, float16 (default: int8)
  --batch-size          Batch size for transcription (default: 16)
  --no-diarize          Disable speaker diarization
  --hf-token            Hugging Face token (or set HF_TOKEN env var)
  --min-speakers        Minimum number of speakers
  --max-speakers        Maximum number of speakers
  -v, --verbose         Enable verbose logging
```

## 出力例

### TXT形式

```
[00:00:00] [SPEAKER_00] えー、本日はですね、プロジェクトの進捗について話し合いたいと思います。
[00:00:05] [SPEAKER_01] はい、よろしくお願いします。
[00:00:07] [SPEAKER_00] まず、フロントエンドの方ですが、進捗はいかがでしょうか。
[00:00:12] [SPEAKER_02] フロントエンドは予定通り進んでいます。
```

### JSON形式

```json
{
  "audio_path": "meeting.m4a",
  "language": "ja",
  "duration_seconds": 5400.0,
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "segments": [
    {
      "start": 0.0,
      "end": 4.8,
      "text": "えー、本日はですね、プロジェクトの進捗について話し合いたいと思います。",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

## 開発

### 開発環境のセットアップ

```bash
uv sync --dev
```

### テスト

```bash
uv run pytest
uv run pytest -v              # 詳細出力
uv run pytest --cov           # カバレッジ付き
```

### リント・フォーマット

```bash
uv run ruff check src/ tests/       # リントチェック
uv run ruff check --fix src/ tests/  # 自動修正
uv run ruff format src/ tests/       # フォーマット
```

## プロジェクト構成

```
transcript-kun/
├── src/transcript_kun/
│   ├── domain/          # ドメイン層：エンティティ、ポリシー、ポート（インターフェース）
│   │   ├── entities.py  # Segment, TranscriptionResult, TranscriptionConfig
│   │   ├── policies.py  # バリデーション・ガードレール
│   │   └── ports.py     # 抽象インターフェース（TranscriptionService等）
│   ├── application/     # アプリケーション層：ユースケース
│   │   └── transcribe.py
│   ├── infrastructure/  # インフラ層：外部ライブラリの実装
│   │   ├── whisperx_service.py
│   │   ├── audio_prober.py
│   │   └── writers.py
│   └── interface/       # インターフェース層：CLI
│       └── cli.py
├── tests/               # テスト
├── pyproject.toml       # プロジェクト設定・依存関係
├── uv.lock              # ロックファイル（uv が自動管理）
└── README.md
```

## ポリシー（ガードレール）

以下のポリシーにより、不正な入力や設定を事前にブロックします：

| ポリシー | 説明 |
|---------|------|
| `FILE_NOT_FOUND` | ファイルが存在しない |
| `NOT_A_FILE` | ディレクトリ等、通常ファイルでない |
| `UNSUPPORTED_FORMAT` | サポート外の拡張子 |
| `EMPTY_FILE` | ファイルサイズが0 |
| `FILE_TOO_LARGE` | 4GBを超えるファイル |
| `INVALID_MODEL` | 不正なモデル名 |
| `INVALID_DEVICE` | 不正なデバイス指定 |
| `INCOMPATIBLE_COMPUTE` | CPUでfloat16は使用不可 |
| `MISSING_HF_TOKEN` | 話者分離にはHFトークンが必要 |
| `INVALID_OUTPUT_FORMAT` | 不正な出力形式 |

## ライセンス

MIT
