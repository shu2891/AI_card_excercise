# AI Card Exercise

Google Colab workflow for project-specific Style LoRA training and image generation.

This repo is structured for one notebook per project. Each project uses its own
Google Drive folder, its own dataset, its own LoRA output, and its own generated PNGs.

## Drive Layout

Each project should live under:

```text
/content/drive/MyDrive/AI_Card_Project/{PROJECT_NAME}/
  raw_images/
  dataset/
  lora/
  outputs/
```

## Repo Contents

- `colab_style_lora_workflow.py`
  - Main Colab workflow functions.
- `src/local_stagegen/ai_card_renderer.py`
  - Raw image generation with SDXL or SSD models and optional LoRA loading.
- `notebooks/project_style_lora_colab.ipynb`
  - Five-step Colab notebook template:
    1. Install
    2. Project Setup
    3. Prepare Dataset
    4. Train LoRA
    5. Generate Images

## Colab Usage

Open the notebook and edit:

- `PROJECT_NAME`
- `PROMPT`
- `SEED`
- `NUM_IMAGES`

Then run the cells in order.

## Notes

- This version does not render card frames.
- Output is PNG images only.
- Each project trains and uses its own LoRA.
