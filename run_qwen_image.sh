#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: run_qwen_image.sh --prompt "Prompt text" [--image /path/to/ref.png ...] [--output result.png]

Options:
  -p, --prompt TEXT/FILE   Prompt text or path to a prompt text file (required)
  -i, --image PATH         Reference image (optional; repeatable 1–10 times for editing)
  -o, --output PATH        Output PNG image path (default: ./output_qwen.png)
  --gpu MODEL              Colab GPU model (default: L4; A100 also supported)
  --steps N                Inference steps (default: 40)
  --width N                Image width (default: 1024)
  --height N               Image height (default: 1024)
  -h, --help               Show this help
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
RUNNER="$SCRIPT_DIR/scripts/runner.py"

PROMPT=""
OUTPUT=""
GPU="${COLAB_GPU:-L4}"
STEPS="40"
WIDTH="1024"
HEIGHT="1024"
IMAGES=()

while (($#)); do
  case "$1" in
    -p|--prompt)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      PROMPT="$2"
      shift 2
      ;;
    -i|--image)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      IMAGES+=("$2")
      shift 2
      ;;
    -o|--output)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      OUTPUT="$2"
      shift 2
      ;;
    --gpu)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      GPU="$2"
      shift 2
      ;;
    --steps)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      STEPS="$2"
      shift 2
      ;;
    --width)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      WIDTH="$2"
      shift 2
      ;;
    --height)
      (($# >= 2)) || { echo "Missing argument for $1" >&2; exit 2; }
      HEIGHT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROMPT" ]]; then
  echo "Error: --prompt is required." >&2
  usage >&2
  exit 2
fi

ARGS=(single --prompt "$PROMPT" --gpu "$GPU" --steps "$STEPS" --width "$WIDTH" --height "$HEIGHT")
if [[ -n "$OUTPUT" ]]; then
  ARGS+=(--output "$OUTPUT")
fi
for img in "${IMAGES[@]}"; do
  ARGS+=(--image "$img")
done

exec python3 "$RUNNER" "${ARGS[@]}"
