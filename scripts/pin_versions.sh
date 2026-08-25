#!/usr/bin/env bash
# Capture the exact environment for the reproducibility statement.
python - <<'PY'
import json, torch, transformers, datasets
meta = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
}
for mod in ("bitsandbytes", "peft", "trl", "accelerate"):
    try:
        meta[mod] = __import__(mod).__version__
    except Exception as e:
        meta[mod] = f"unavailable: {e}"
print(json.dumps(meta, indent=2))
open("results/environment.json", "w").write(json.dumps(meta, indent=2))
PY
