<!--
# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# vLLM Backend - ROCm Edition

## Architecture

The vLLM backend is a **pure Python backend** — it contains no C/C++ or HIP/CUDA
code. It acts as a thin wrapper that bridges Triton's
[Python backend](https://github.com/ROCm/triton-inference-server-python_backend)
interface with the [vLLM](https://github.com/vllm-project/vllm) inference
engine. All heavy lifting (inference, paged attention, continuous batching) is
performed by the vLLM engine.

### Repository contents

| File | Purpose |
|---|---|
| `src/model.py` | `TritonPythonModel` class — the entry point that Triton loads. Receives requests, forwards them to the vLLM `AsyncEngine`, and streams responses back. |
| `src/utils/metrics.py` | vLLM statistics and metrics integration with Triton. |
| `src/utils/request.py` | Request handling utilities for generate and embed operations. |

### How the backend is built and deployed

There is no CMake build or compilation step. The build process (driven by the
Triton server's `build.py`) is:

1. **Git clone** this repository.
2. **Copy** `src/model.py` and `src/utils/` into
   `/opt/tritonserver/backends/vllm/`.
3. **Install the vLLM engine** separately:

The Python `model.py` itself is hardware-agnostic — it calls vLLM's Python API
(`AsyncEngineArgs`, `build_async_engine_client_from_engine_args`), and vLLM
internally handles whether it is running on CUDA or ROCm.

### ROCm enablement

Since this backend is pure Python, ROCm support does **not** require
hipification or any C/C++ changes in this repository. The ROCm enablement
happens in two places outside this repo:

1. **vLLM engine** — vLLM has its own ROCm support. 
2. **Triton server** — the server's own C++ code (shared memory manager, gRPC/HTTP
   endpoints, etc.) has `#ifdef TRITON_ENABLE_ROCM` guards that swap CUDA API
   calls for HIP equivalents. Those changes live in the
   [server repository](https://github.com/ROCm/triton-inference-server-server), not
   here.