# Qwen-Image-2.1 Colab Skill

A standalone Codex and Antigravity skill repository for generating and editing images via Google Colab GPU using Alibaba's **Qwen-Image-2.1** model.

- [繁體中文文件](README.zh-TW.md)

## What is included

- **Unified Generation & Editing**: Supports text-to-image (T2I) and image-to-image editing (1–10 reference images) within a single pipeline.
- **Native RGBA Transparency**: Generates transparent PNG assets directly without external background-removal tools.
- **Colab GPU support**: Uses an L4 (with bundled CPU offload) or A100 Colab GPU; runtime depends on the selected GPU, image size, and inference steps.
- **Batch Session Management**: Reuses a single live Colab session across multiple jobs to avoid redundant downloads, and terminates cleanly upon completion or error.
- `SKILL.md` — agent instructions.
- `scripts/runner.py` — Colab session and generation runner.
- `assets/Qwen_Image_2_1_Colab.ipynb` — remote inference notebook (with L4 CPU-offload patch applied).
- `references/l4-memory-20260930.md` — validated L4 22 GB VRAM memory notes.
- `install.sh` — installer that preserves an existing skill unless `--force` is used.
- `tests/test_runner.py` — offline tests using a mock Colab CLI.

The repository does not include model weights or generated images. Prompts and reference images are sent to the Colab runtime for generation.

## Quick Start

```bash
git clone https://github.com/killkli/qwen-image-colab-skill.git
cd qwen-image-colab-skill
./install.sh
colab --auth=oauth2 usage
```

The installer places the skill under `$CODEX_HOME/skills/qwen-image-colab`, or `~/.codex/skills/qwen-image-colab` when `CODEX_HOME` is not set. Use `./install.sh --dest /absolute/path/to/skills` to select another skills directory. Existing installations are preserved unless `--force` is passed; replacement moves the previous directory to a timestamped backup.

The local runner requires Python 3.10 or newer, Bash, and an installed, authenticated Google Colab CLI. Colab GPU access and compute-unit availability are required for generation. For example, install the CLI with `uv` and Python 3.12:

```bash
uv python install 3.12
uv tool install --python 3.12 google-colab-cli
```

### Single Image Inference

```bash
./run_qwen_image.sh \
  --prompt "A cute magical cat wizard casting sparkles, anime style, 8k" \
  --output ./cat_wizard.png
```

### Batch Execution

```bash
python3 scripts/runner.py batch \
  --manifest jobs.json \
  --gpu L4 \
  --output-dir ./outputs
```

Run the offline checks without starting a Colab session:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/runner.py --help
bash -n install.sh run_qwen_image.sh
```
