# Checkpoint

`checkpoint.pt` (~15 GB) is the released FaceAnything model. It is loaded by
default by `run_inference.py`. The weight file is git-ignored, so on a fresh
clone it is absent. Download it and place it here as `checkpoint.pt` using
either option below.

**Option 1: Google Drive**

```bash
gdown --fuzzy "https://drive.google.com/file/d/1PdQQxzm-tU50RmJhgeoMCYVRlEiW3f8p/view?usp=sharing" \
    -O checkpoint.pt
```

**Option 2: Hugging Face** ([UmutKocasari/FaceAnything](https://huggingface.co/UmutKocasari/FaceAnything))

```bash
huggingface-cli download UmutKocasari/FaceAnything checkpoint.pt --local-dir .
```

Or pass `--checkpoint /path/to/checkpoint.pt` to load it from elsewhere.
