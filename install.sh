#!/usr/bin/env bash
set -Eeuo pipefail

SKILL_NAME="qwen-image-colab"
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FORCE=0
SKILLS_ROOT=""

usage() {
  cat <<'EOF'
Install the Qwen-Image-2.1 Colab skill into a Codex/Gemini skills directory.

Usage:
  ./install.sh [--dest SKILLS_DIR] [--force]

Options:
  --dest PATH  Skills directory to use (default: $CODEX_HOME/skills or ~/.codex/skills)
  --force      Replace an existing install after moving it to a timestamped backup
  -h, --help   Show this help
EOF
}

while (($#)); do
  case "$1" in
    --dest)
      (($# >= 2)) || { echo "Missing path after --dest." >&2; usage >&2; exit 2; }
      SKILLS_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
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

if [[ -z "$SKILLS_ROOT" ]]; then
  if [[ -n "${CODEX_HOME:-}" ]]; then
    SKILLS_ROOT="$CODEX_HOME/skills"
  elif [[ -n "${HOME:-}" ]]; then
    SKILLS_ROOT="$HOME/.codex/skills"
  else
    echo "Set CODEX_HOME or HOME before installing the skill." >&2
    exit 2
  fi
fi

SKILLS_ROOT="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$SKILLS_ROOT")"
TARGET="$SKILLS_ROOT/$SKILL_NAME"

for required in SKILL.md scripts/runner.py assets/Qwen_Image_2_1_Colab.ipynb; do
  [[ -f "$REPO_DIR/$required" ]] || { echo "Repository is missing required file: $required" >&2; exit 2; }
done

mkdir -p "$SKILLS_ROOT"

if [[ -e "$TARGET" || -L "$TARGET" ]]; then
  if [[ "$FORCE" != 1 ]]; then
    if [[ -d "$TARGET" ]]; then
      echo "Skill already installed; preserving: $TARGET"
      echo "Use --force to replace it (the old directory will be backed up)."
      exit 0
    fi
    echo "Refusing to replace non-directory path: $TARGET" >&2
    exit 2
  fi
  [[ -d "$TARGET" && ! -L "$TARGET" ]] || { echo "Refusing to replace non-directory or symlink path: $TARGET" >&2; exit 2; }
fi

STAGING="$(mktemp -d "$SKILLS_ROOT/.${SKILL_NAME}.install.XXXXXXXX")"
BACKUP=""
cleanup() {
  if [[ -n "$STAGING" && -d "$STAGING" ]]; then
    rm -rf "$STAGING"
  fi
}
trap cleanup EXIT

cp -R "$REPO_DIR/SKILL.md" "$STAGING/SKILL.md"
cp -R "$REPO_DIR/scripts" "$STAGING/scripts"
cp -R "$REPO_DIR/assets" "$STAGING/assets"
if [[ -d "$REPO_DIR/references" ]]; then
  cp -R "$REPO_DIR/references" "$STAGING/references"
fi
find "$STAGING" -type d -name __pycache__ -prune -exec rm -rf {} +
chmod +x "$STAGING/scripts/runner.py"

if [[ -e "$TARGET" || -L "$TARGET" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  BACKUP="$TARGET.backup.$timestamp"
  suffix=0
  while [[ -e "$BACKUP" || -L "$BACKUP" ]]; do
    suffix=$((suffix + 1))
    BACKUP="$TARGET.backup.$timestamp.$suffix"
  done
  mv "$TARGET" "$BACKUP"
  echo "Moved existing install to backup: $BACKUP"
fi

mv "$STAGING" "$TARGET"
STAGING=""
echo "Installed $SKILL_NAME to $TARGET"
