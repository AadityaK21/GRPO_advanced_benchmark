"""Settings for every test run. They take effect before any test imports torch or transformers."""
import os

# The tests build tiny models from configs and never need the Hugging Face Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# TRL's GRPOTrainer refuses to run when one process sees several GPUs, and Kaggle's "GPU T4 x2" shows two.
_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if _devices is None or "," in _devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = (_devices or "0").split(",")[0]
