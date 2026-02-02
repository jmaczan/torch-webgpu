#!/usr/bin/env python
"""
Export Qwen2.5-0.5B-Instruct to ONNX format using Optimum.
"""

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "qwen_onnx"


def export_model(output_dir: Path, optimize: bool = True):
    """Export Qwen model to ONNX format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_NAME} to ONNX...")
    print(f"Output directory: {output_dir}")

    # Export using Optimum
    # This will download the model, convert to ONNX, and optionally optimize
    model = ORTModelForCausalLM.from_pretrained(
        MODEL_NAME,
        export=True,
        trust_remote_code=True,
    )

    # Save the model
    model.save_pretrained(output_dir)

    # Also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_dir)

    print(f"Export complete! Model saved to {output_dir}")

    # List exported files
    print("\nExported files:")
    for f in output_dir.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen to ONNX")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for ONNX model"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip ONNX optimization"
    )
    args = parser.parse_args()

    export_model(Path(args.output), optimize=not args.no_optimize)


if __name__ == "__main__":
    main()
