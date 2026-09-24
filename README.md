# minimax-music3-nvidia-ai-hub

Runtime image for the **Spark AI Hub** recipe *MiniMax Music 3 Studio*, built native for the
NVIDIA DGX Spark (GB10, ARM64 + CUDA 13).

```
ghcr.io/hitechcloud-vietnam/minimax-music3-spark-ai-hub:1.1.0
```

## What is in it

CUDA 13.0.1 runtime (ARM64), Python 3.12, ffmpeg, and:

| | |
|---|---|
| torch | 2.11.0+cu130 (aarch64) |
| diffusers | `82319140` — the commit carrying `MiniMaxMusic3Pipeline` |
| transformers | 5.8.0 |
| gradio | 6.24.0 |

plus both upstream Spaces' interfaces, pinned by revision and adapted for the Spark:

- [`MiniMaxAI/MiniMax-Music3`](https://huggingface.co/spaces/MiniMaxAI/MiniMax-Music3) — the Studio composer
- [`MiniMaxAI/MiniMax-Music3-workflow`](https://huggingface.co/spaces/MiniMaxAI/MiniMax-Music3-workflow) — the node canvas

**No model weights.** They are baked in by the recipe's own weights layer, so this image stays
stable across weight changes and installs as a plain `docker pull`.

## Spark adaptations

1. **AoTI removed.** The Spaces load kernels compiled for `sm120` (RTX PRO 6000, which is what the
   ZeroGPU pool runs). GB10 is `sm121`, so the package cannot load — and fetching it would break
   offline launch. The eager LM path is used instead.
2. **Local component paths.** At this diffusers commit, `_get_checkpoint_shard_files` calls the
   Hub's `model_info()` to verify shard lists unless `local_files_only` is set — and
   `HF_HUB_OFFLINE=1` does *not* set it. A repo id therefore needs the network even with a fully
   populated cache. Components load from `/models/MiniMax-Music3` instead.
3. **One engine, two interfaces.** The two Spaces' `app.py` files are byte-identical for their
   first 472 lines — the entire engine, including the pipeline load. The canvas half is rewired to
   import that engine from the studio module, so one process serves both UIs off one resident copy
   of the model.
4. **The prompt composer runs on-device.** Upstream posts the user's description to
   `router.huggingface.co` so a hosted model can draft the lyrics and structured caption. Spark AI
   Hub talks only to your Spark, so that path is replaced with a local Qwen3.5-4B shipped in the
   recipe's weights layer. No token, no network.

## Licence

The MiniMax Music 3 model and the two Space interfaces are the property of their respective
authors and remain under their own licences. This repository packages them for the DGX Spark.
