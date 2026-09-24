# =============================================================================
# ghcr.io/hitechcloud-vietnam/minimax-music3-nvidia-ai-hub
#
# The runtime half of the Spark AI Hub "MiniMax Music 3 Studio" recipe: CUDA 13
# + torch 2.11 (aarch64) + the diffusers commit that carries MiniMaxMusic3, and
# the two upstream Spaces' interfaces adapted for the DGX Spark. No weights —
# those are baked in by the recipe's own weights stage, so this image is stable
# across weight changes and installs as a plain `docker pull`.
#
# Build:  docker build -t ghcr.io/hitechcloud-vietnam/minimax-music3-nvidia-ai-hub:1.0.0 runtime/
# =============================================================================

# ---- the two Spaces, pinned by revision and adapted for the Spark -----------
FROM python:3.13-slim AS source
RUN pip install --no-cache-dir huggingface_hub
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('MiniMaxAI/MiniMax-Music3', repo_type='space', local_dir='/src', \
  revision='929e40ed4e425518693ec955ced52aba5ea6c425'); \
snapshot_download('MiniMaxAI/MiniMax-Music3-workflow', repo_type='space', local_dir='/wf', \
  revision='1866d52ff1d11e0ad453cda752da482b5801fe21')"

RUN python3 - <<'PATCH'
import pathlib

studio = pathlib.Path("/src/app.py")
text = studio.read_text()

# (1) Drop the AoTI block. Its artifacts are compiled for sm120 (RTX PRO 6000,
#     which is what the ZeroGPU pool runs); the Spark's GB10 is sm121, so the
#     package cannot load — and fetching it would break offline launch anyway.
#     The eager LM path defined directly below it becomes the default.
start = text.index("# AoTI-compiled kernels")
end = text.index("PIPE._iter_frames = _iter_frames_aoti") + len("PIPE._iter_frames = _iter_frames_aoti")
text = text[:start] + "# AoTI block removed for the Spark (sm121); the eager LM path below is used instead." + text[end:]
text = text.replace("# eager fallback available as _iter_frames_eager",
                    "PIPE._iter_frames = _iter_frames_eager")
assert "PIPE._iter_frames = _iter_frames_eager" in text

# (2) Load every component from the image's own directory rather than the repo
#     id. At this diffusers commit `_get_checkpoint_shard_files` calls the Hub's
#     `model_info()` to verify shards unless `local_files_only` is set, and
#     HF_HUB_OFFLINE does not set it — a repo id therefore needs the network
#     even with a fully populated cache. A path takes the local-folder branch
#     and never touches the Hub.
text = text.replace('ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")',
                    'ModularPipeline.from_pretrained("/models/MiniMax-Music3")')
assert '"/models/MiniMax-Music3"' in text

# (3) Songs land on the user-data volume instead of the container's /tmp.
text = text.replace('_SONGS_DIR = "/tmp/mm3_songs"', '_SONGS_DIR = "/data/songs"')

# (4) The prompt composer runs ON THIS MACHINE. Upstream sends the user's
#     description to router.huggingface.co so a hosted DeepSeek can draft the
#     lyrics and structured caption; the Hub talks only to your Spark, so that
#     whole path is replaced with a local Qwen3.5-4B baked into the image.
#     Same signature, same JSON contract, no token, no network.
old_start = text.index("def _llm_json(system, user, required=()):")
old_end = text.index("def compose_song(")
text = text[:old_start] + \
"""_COMPOSER_DIR = "/models/composer"
_COMPOSER_LM = None
_COMPOSER_TOK = None


def _llm_json(system, user, required=()):
    # On-device composer. Loaded lazily: it is an optional convenience, so it must
    # not add ~9 GB and a load to every launch of the music pipeline.
    import json as _json

    global _COMPOSER_LM, _COMPOSER_TOK
    if _COMPOSER_LM is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[composer] loading", _COMPOSER_DIR, flush=True)
        _COMPOSER_TOK = AutoTokenizer.from_pretrained(_COMPOSER_DIR)
        _COMPOSER_LM = AutoModelForCausalLM.from_pretrained(
            _COMPOSER_DIR, dtype=torch.bfloat16
        ).to("cuda").eval()

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        prompt = _COMPOSER_TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = _COMPOSER_TOK.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    last_error = None
    # Two passes: greedy first for format adherence, then sampled, which usually
    # shakes loose a reply that greedy decoding truncated or malformed.
    for attempt in range(2):
        inputs = _COMPOSER_TOK(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = _COMPOSER_LM.generate(
                **inputs, max_new_tokens=2048,
                do_sample=bool(attempt), temperature=0.8, top_p=0.9,
                pad_token_id=_COMPOSER_TOK.eos_token_id,
            )
        reply = _COMPOSER_TOK.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        if "</think>" in reply:
            reply = reply.rsplit("</think>", 1)[1]
        try:
            # Tolerate fences/preambles and reject truncated replies: parse the
            # outermost {...} span, exactly as the upstream cloud path did.
            begin, finish = reply.find("{"), reply.rfind("}")
            if begin == -1 or finish <= begin:
                raise ValueError("no JSON object in composer reply")
            data = _json.loads(reply[begin:finish + 1])
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"composer reply missing keys: {missing}")
            return data
        except Exception as exc:
            print(f"[composer] attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            last_error = exc
    raise gr.Error(
        "The on-device composer could not produce a usable draft \u2014 try again, or "
        "write the lyrics and structured prompt yourself in Studio mode."
    ) from last_error


""" + text[old_end:]
assert "router.huggingface.co" not in text
studio.write_text(text)

# (5) The workflow Space's app.py is byte-identical to the studio's for its
#     first 472 lines — the whole engine, including the pipeline load. Rewire
#     its canvas half to import that engine from the studio module, so one
#     process serves both interfaces off one resident copy of the model.
wf = pathlib.Path("/wf/app.py").read_text().split("\n")
assert wf[472].startswith("def render_video("), wf[472]
pathlib.Path("/wf/app_canvas.py").write_text(
    "# Spark AI Hub: lines 1-472 of this file are byte-identical to the studio\n"
    "# Space's app.py, so the canvas reuses that module rather than loading a\n"
    "# second copy of the pipeline. Everything below is the workflow Space's code.\n"
    "from app import PIPE, _encode_prompt, _stream_windows, _to_int16, gr, np, os, random, spaces, time, torch\n"
    "\n" + "\n".join(wf[472:])
)
PATCH

# ---- runtime ----------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04
LABEL org.opencontainers.image.source="https://github.com/hitechcloud-vietnam/minimax-music3-nvidia-ai-hub
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3-pip ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      torch==2.11.0+cu130 \
      "diffusers @ https://github.com/huggingface/diffusers/archive/82319140e0456fd58beff0a251c38825bfc310de.tar.gz" \
      transformers==5.8.0 accelerate==1.14.0 huggingface-hub==1.24.0 \
      gradio==6.24.0 spaces==0.51.1 scipy numpy pillow "uvicorn[standard]" fastapi

WORKDIR /app
COPY --from=source /src /app
COPY --from=source /wf/app_canvas.py /wf/workflow.json /app/
