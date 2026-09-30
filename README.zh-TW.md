# Qwen-Image-2.1 Colab 技能

**English:** [README.en.md](README.en.md)

這是一個完整、可獨立使用的 Codex / Antigravity 技能儲存庫，能將文字 Prompt 與本機參照圖片交給 Google Colab GPU，使用阿里巴巴 **Qwen-Image-2.1** 進行文生圖（T2I）與影像編輯（Image Editing）。

儲存庫包含：
- `SKILL.md`：Codex / Antigravity 選用此技能時讀取的自動化指示；
- `scripts/runner.py`：管理 Colab session、查詢點數餘額、批次上傳、執行推論與安全關機清理的 Python runner；
- `assets/Qwen_Image_2_1_Colab.ipynb`：在遠端 Colab GPU 執行的推論 Notebook（已套用 L4 22GB CPU offload 修正）；
- `references/l4-memory-20260930.md`：L4 GPU VRAM 限制驗證紀錄；
- `run_qwen_image.sh`：單張圖片推論的便利 Shell 啟動器；
- `install.sh`：可攜、可重複執行且預設不覆蓋舊檔的安裝程式；
- `tests/test_runner.py`：使用 Mock Colab CLI 執行、不耗費任何雲端點數的單元測試。

---

## 核心亮點與特色

1. **統一架構（Unified Generation & Editing）**：
   * 支援純文字生圖（Text-to-Image）。
   * 支援單張或多張（最高 10 張）參照圖片的影像編輯、換裝、風格轉移與主體一致性保留。
2. **原生透明通道（RGBA）**：
   * 可原生輸出帶 Alpha 通道的透明背景圖片，無需額外執行摳圖或去背演算法。
3. **支援 Colab GPU（L4 / A100）**：
   * 可使用 Colab L4 GPU（22GB VRAM，內建 CPU offload）或 A100；實際耗時依 GPU、圖片尺寸與推論步數而異。L4 VRAM 限制與驗證紀錄見 `references/l4-memory-20260930.md`。
4. **自動生命週期管理**：
   * 批次任務自動重用同一個 Colab Session，避免重複下載模型。
   * 執行完畢或中途例外自動安全關閉 Session，避免點數外洩。

---

## 快速開始

### 1. 安裝技能

先複製儲存庫，再執行安裝器（macOS、Linux 或 WSL 均可）：

```bash
git clone https://github.com/killkli/qwen-image-colab-skill.git
cd qwen-image-colab-skill
```

```bash
./install.sh
```

預設安裝至 `$CODEX_HOME/skills/qwen-image-colab`；若未設定 `CODEX_HOME`，則安裝至 `~/.codex/skills/qwen-image-colab`。既有安裝預設會保留；加上 `--force` 才會替換，原目錄會先移至帶有時間戳記的備份路徑。

如需安裝到其他技能目錄（例如 Antigravity 全域技能目錄）：

```bash
./install.sh --dest ~/.gemini/config/skills
```

### 2. 安裝與驗證 Colab CLI

Runner 需要 Python 3.10 以上版本；安裝 Colab CLI 可使用 Python 3.12：

```bash
uv python install 3.12
uv tool install --python 3.12 google-colab-cli
colab --auth=oauth2 usage
```

實際生圖需要已登入的 Google Colab 帳戶、可用的 GPU 與運算單位。Prompt 和參照圖片會傳送至 Colab 執行環境；本機安裝或執行離線測試不會啟動 GPU。

---

## 使用方式

### 單圖推論快捷指令
```bash
# 純文字生圖 (T2I)
./run_qwen_image.sh \
  --prompt "A cute magical cat wizard casting sparkles, anime style, 8k" \
  --output ./cat_wizard.png

# 影像風格編輯 (Edit / 參照圖)
./run_qwen_image.sh \
  --image ./my_photo.jpg \
  --prompt "Transform <Picture 1> into a cyberpunk neon portrait with glowing goggles" \
  --output ./cyberpunk_me.png
```

### 批次任務 (`jobs.json`)
```json
{
  "jobs": [
    {
      "id": "t2i_landscape",
      "mode": "t2i",
      "prompt": "Sunset over a tranquil Japanese Zen garden with cherry blossoms, photorealistic",
      "width": 1024,
      "height": 1024,
      "output_name": "zen_garden"
    },
    {
      "id": "edit_portrait",
      "mode": "edit",
      "reference_images": ["/absolute/path/character.png"],
      "prompt": "Change the background of <Picture 1> to an illuminated concert stage with laser lights",
      "width": 1024,
      "height": 1024,
      "output_name": "concert_character"
    }
  ]
}
```

執行批次任務：
```bash
python3 scripts/runner.py batch \
  --manifest jobs.json \
  --gpu L4 \
  --output-dir ./outputs
```

### 離線檢查

以下指令使用 mock Colab CLI，不會啟動 Colab session 或消耗運算單位：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/runner.py --help
bash -n install.sh run_qwen_image.sh
```
