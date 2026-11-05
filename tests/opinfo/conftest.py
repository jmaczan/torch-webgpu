import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TORCH_WEBGPU_FALLBACK_WARN", "0")
