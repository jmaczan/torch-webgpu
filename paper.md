---
title: 'torch-webgpu: PyTorch compiler and WebGPU runtime'
tags:
  - C++
  - WebGPU
  - Python
  - WGSL
  - PyTorch
  - Machine Learning
authors:
  - name: Jędrzej Maczan
    orcid: 0000-0003-1741-6064
    affiliation: 1
    corresponding: true
affiliations:
 - name: Independent Researcher, Poland
   index: 1
date: 14 April 2026
bibliography: paper.bib
---

# Summary

PyTorch [@paszke2019pytorch] is the dominant framework for machine learning research and deployment, providing tensor operations and automatic differentiation used across virtually all modern deep learning workflows. However, PyTorch's high-performance execution has historically been tied to specific hardware ecosystems — primarily NVIDIA GPUs via CUDA and Apple Silicon via MPS — requiring separate implementations for each target platform.

WebGPU [@webgpu2024] is a modern graphics and compute API that provides a unified abstraction over diverse GPU hardware, with implementations available across desktop, mobile, and browser environments. By targeting WebGPU, computations become portable across hardware that would otherwise require distinct backends.

torch-webgpu bridges these two ecosystems. It implements a PyTorch out-of-tree backend that compiles PyTorch models to WebGPU compute operations, enabling PyTorch tensors and models to run on any WebGPU-capable hardware. Tensor operations are implemented as WGSL (WebGPU Shading Language) compute shaders, and the backend integrates with PyTorch's `torch.compile` infrastructure via the `PrivateUse1` device extension mechanism introduced in PyTorch 2.0.

# Statement of need

High-performance machine learning research depends on PyTorch, but hardware portability remains a significant challenge. Running PyTorch models on hardware without CUDA support — including integrated GPUs, mobile devices, and non-NVIDIA discrete GPUs — currently requires either CPU fallback with substantial performance loss, or substantial engineering effort to port to a new backend.

WebGPU provides a hardware abstraction layer with broad support across GPU vendors and platforms, including browser-based execution. However, no existing tool provides a direct path from PyTorch models to WebGPU execution. Solutions such as TensorFlow.js [@abadi2016tensorflow] and ONNX Runtime Web operate on different model representations and require format conversion, breaking the native PyTorch workflow.

torch-webgpu fills this gap by implementing WebGPU as a native PyTorch device. Researchers can run existing PyTorch models on WebGPU hardware using the standard `torch.compile` interface and `device="webgpu"` semantics, without modifying model code or converting to intermediate formats.

# State of the field

PyTorch 2.0 introduced `torch.compile` with TorchDynamo and TorchInductor as its default compiler stack, targeting CUDA and CPU. The `PrivateUse1` mechanism allows out-of-tree backends to register as first-class PyTorch devices. Existing out-of-tree backends include implementations for Ascend NPUs [@ascend2024] and other vendor-specific hardware, demonstrating the viability of this approach.

On the WebGPU side, TensorFlow.js provides GPU-accelerated ML in browsers using WebGL and WebGPU, but operates independently of the PyTorch ecosystem. ONNX Runtime Web supports inference from ONNX models in browsers, but again requires format conversion from PyTorch. Neither provides the native PyTorch device experience that torch-webgpu targets.

# Software design

torch-webgpu is implemented primarily in C++ with a thin Python wrapper distributed as a PyPI package. The backend is structured around four components.

The WebGPU context (`webgpu_context.cpp`) manages adapter and device lifecycle using Dawn [@dawn2024], Google's native WebGPU implementation. On initialization, the context requests a high-performance adapter and falls back to any available adapter if none is found, then queries adapter limits to configure the device with maximum supported capabilities.

The WebGPU allocator (`webgpu_allocator.cpp`) implements PyTorch's allocator interface, mapping tensor memory to WebGPU buffer objects with `Storage | CopySrc | CopyDst` usage flags. Buffer sizes are aligned to 4-byte boundaries as required by WebGPU's `WriteBuffer` operation.

PyTorch operator bindings (`bindings.cpp`) register the backend under PyTorch's `PrivateUse1` device type using `TORCH_LIBRARY_IMPL`. Operators with native WebGPU implementations are registered directly; unsupported operators fall back to CPU via `at::native::cpu_fallback`.

Compute operations are implemented as WGSL compute shaders. Activation functions including ReLU, GeLU, and SiLU (`activation_functions.cpp`) are dispatched through PyTorch's `TensorIterator` infrastructure to WGSL kernels. Matrix multiplication (`mm.wgsl`) uses a 2D dispatch where each GPU thread computes one output element of C = A x B. The shader accepts a `Params` uniform buffer containing matrix dimensions, per-tensor storage offsets, and stride arrays of up to 8 dimensions, enabling correct indexing of non-contiguous and transposed tensors without requiring data copies. Workgroup size is fixed at 16x16 threads, with dispatch dimensions computed to cover the full M x K output space.

# Research impact statement

torch-webgpu has been used to conduct empirical benchmarking of PyTorch model inference across heterogeneous hardware via WebGPU, the results of which are reported in [@maczan2026webgpu]. That study demonstrates that WebGPU-based execution enables cross-platform ML inference on hardware not supported by existing PyTorch backends.

# AI usage disclosure

Anthropic's Claude Opus 4.6 was used to generate and optimize some WGSL shader code and to generate test scripts. All design decisions, architecture, and research were made by the author. Claude was also used to proof-read and improve the quality of this paper. All AI-assisted outputs were reviewed and validated by the author.

# Acknowledgements

The author thanks the PyTorch team for their documentation on out-of-tree backends; Elie Michel for the WebGPU C++ guide [@michel2024]; the W3C WebGPU working group for the WGSL specification; the WebGPU Fundamentals project [@webgpufundamentals2024] for educational resources; and the Ascend PyTorch team [@ascend2024] for a high-quality reference implementation of a PrivateUse1 backend.

# References