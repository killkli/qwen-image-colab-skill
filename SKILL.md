---
name: qwen-image-colab
description: Generate and edit images using Alibaba Qwen-Image-2.1 on Google Colab CLI. Supports text-to-image, editing with up to 10 reference images, and native RGBA transparency.
---

# Qwen-Image-2.1 on Colab

Use the bundled notebook and `scripts/runner.py` to generate or edit high-quality images using Alibaba's Qwen-Image-2.1 on Google Colab. The Colab CLI must be installed and authenticated with available compute units.

## Workflow

1. Determine the generation mode:
   - **Text-to-Image (`t2i`)**: Pure text prompt describing the image to generate. Supports native transparent background (RGBA) when requested.
   - **Image Editing / Reference Guidance (`edit`)**: Provide 1–10 local reference images alongside a prompt describing desired transformations, identity preservation, or multi-subject composition.
2. When reference images are provided, image order corresponds to `<Picture 1>`, `<Picture 2>`, through `<Picture 10>` in the prompt.
3. Check Colab access and compute unit balance with:
   ```bash
   python3 scripts/runner.py usage --json
   ```
4. Before starting generation, make sure remote processing is appropriate for the supplied prompt and reference images. The runner uploads them to Google Colab, and the session consumes account compute units. Do not include credentials or secrets.
5. Define tasks in a UTF-8 JSON manifest (`jobs.json`) and run the batch. The runner reuses a single Colab session to prevent repeated model downloads, executes jobs sequentially, downloads resulting PNG images, and stops the session upon completion.
6. Verify output files and report local image paths.

### Example Manifest (`jobs.json`):

```json
{
  "jobs": [
    {
      "id": "cyberpunk_city",
      "mode": "t2i",
      "prompt": "A bustling cyberpunk night market with holographic signs, neon reflections on wet asphalt, 8k resolution, cinematic lighting",
      "width": 1024,
      "height": 1024,
      "steps": 40,
      "output_name": "cyberpunk_city"
    },
    {
      "id": "character_restyle",
      "mode": "edit",
      "reference_images": ["/absolute/path/photo.jpg"],
      "prompt": "Turn <Picture 1> into a vibrant 3D anime game character with glowing neon accents",
      "width": 1024,
      "height": 1024,
      "output_name": "character_restyle"
    }
  ]
}
```

### Running from Skill Directory:

```bash
python3 scripts/runner.py batch \
  --manifest /absolute/path/jobs.json \
  --gpu L4 \
  --output-dir /absolute/path/outputs
```

Recommended GPU: `L4` (24GB VRAM, highly cost-effective) or `A100`.

> [!NOTE]
> On Windows host environments, execute the runner inside WSL (Ubuntu) where the Colab CLI is installed and authenticated:
> ```bash
> wsl -d ubuntu bash -lc "python3 ~/.codex/skills/qwen-image-colab/scripts/runner.py batch --manifest /path/to/jobs.json --gpu L4 --output-dir /path/to/outputs"
> ```

## Operational Limits

- **Reference Images**: 0 for `t2i`, 1–10 for `edit`. Images must be valid, readable image files.
- **Dimensions**: Default `1024x1024`. Multiples of 16 recommended (e.g., 768x1024, 1024x768, 1280x720, 2048x2048).
- **Steps**: Typically 30–50 steps (default: 40).
- **Session Cleanup**: The runner automatically terminates Colab sessions upon batch completion or unexpected error.
