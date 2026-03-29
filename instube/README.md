# InsTube: VLN Instruction Generation Pipeline

Uses Gemini API to generate and augment natural language navigation instructions from rendered image sequences and goal metadata produced by [vistube](../vistube/).

## Pipeline Overview

```
Step 1: Image Sequence → Navigation Instruction
        │  Feed rendered RGB frames to Gemini to produce
        │  second-person imperative navigation instructions.
        ▼
Step 2: Goal Image → Targeted Caption
        │  Generate a focused description of the goal object's
        │  appearance, state, and spatial relationships from
        │  a reference image.
        ▼
Step 3: Fusion & Augmentation
        │  Fuse template-based text instructions (from vistube stage 2)
        │  with the image caption, then rewrite into three styles:
        │  formal, natural, and casual.
        ▼
    Output: per-scene JSON files with original + augmented instructions.
```

## Scripts

| Script | Purpose |
|---|---|
| `prompt.py` | Prompt templates for all Gemini calls (video/image-sequence instruction, caption generation, fusion rewrite) |
| `gemini_images_analyzer.py` | Step 1: Generates navigation instructions from rendered image sequences |
| `gemini_aug_goal_image_enhance.py` | Steps 2-3: Generates targeted captions from goal images, then fuses with text instructions to produce augmented instructions in three styles |

## Dependencies

### Third-party packages

```bash
pip install google-generativeai Pillow tqdm natsort
```

## Prerequisites

- **Gemini API key**: Set via `export GOOGLE_API_KEY='your_key'`
- **vistube outputs**: Rendered image sequences (`sequence_discrete/`) and goal instructions (`goal_inst.json`) from vistube stages 2-3

## Usage

```bash
# Step 1: Generate instructions from image sequences
python instube/gemini_images_analyzer.py

# Step 2-3: Augment instructions with goal image captions
python instube/gemini_aug_goal_image_enhance.py
```

Paths (`dataroot`, `task_dir`, etc.) are currently configured as variables at the bottom of each script.

## Output Structure

Step 1 produces per-scene:
```
<dataroot>/<scene_id>/<task_dir>/inst/
└── inst_img_sequence.json    # Gemini-generated instructions per sequence
```

Steps 2-3 produce per-scene:
```
<dataroot>/<scene_id>/<task_dir>/
└── goal_inst_aug_enhance.json  # Original instructions + augmented (formal/natural/casual)
```

## Notes

- **Rate limiting**: A 1.1s sleep is enforced between API calls to stay within Gemini rate limits.
- **Resumable**: Both scripts support checkpoint/resume — they skip already-processed entries on restart.
- **Frame sampling**: `gemini_images_analyzer.py` uniformly samples up to 30 frames when a sequence has more images.
- **Image quality check**: `gemini_aug_goal_image_enhance.py` skips problematic images (black/white/blurry) via stddev/mean thresholds before captioning.
