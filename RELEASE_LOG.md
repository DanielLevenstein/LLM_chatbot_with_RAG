
# Docker Releases

| Version | Image Size | Change                                        |
|---------|------------|-----------------------------------------------|
| v0.0.1  | 15.8 GB    | First working build onrender.com 15.8 GB      |
| v0.1.0  | 2.97 GB    | Downgraded to pytorch 2.7.1 to fix image size |
| v0.1.1  | 2.97 GB    | Implemented Lazy Loading  (not merged)        |
| v0.2.0  | 2.97 GB    | Implemented context aware chunking            |
| v0.2.4  | 3.66 GB    | Parameter Tuning (update image platform)      | 
| v0.2.5  | 4.09 GB    | Docker cache warm for LLM and embeddings      |
| v0.2.6  | 4.09 GB    | Safer RAG context cap for 2048-token window   |
| v0.2.7  | 1.22 GB    | CPU-only torch and production timing logs     |
| v0.2.8  | 3.97 GB    | Clean tag for corrected CPU-only image        |
| v0.3.0  | 3.54 GB    | Restored lazy model loading UI;               |
| v0.3.1  | 1.19 GB    | Split README from RELEASE_LOG Smaller Image   |

## Release Stats

| Version | Batch Size | Chunk Size | Context Window | Threads | max_tokens | temperature | repeat_penalty | GPU Layers |
|---------|------------|------------|----------------|---------|------------|-------------|----------------|------------|
| v0.1.0  | 512        | 500        | 4096           | 2       | 512        | 0.0         | 1.2            | 42         |
| v0.1.1  | 256        | 500        | 1096           | 4       | 256        | 0.3         | 1.1            | 0          |
| v0.1.2  | 512        | 500        | 2048           | 4       | 512        | 0.0         | 1.2            | 0          |
| v0.2.0  | 512        | 500        | 2048           | 4       | 512        | 0.0         | 1.2            | 50         |
| v0.2.4  | 100        | 500        | 2048           | auto    | 250        | 0.0         | 1.2            | 0          |
| v0.2.5  | 100        | 500        | 2048           | auto    | 250        | 0.0         | 1.2            | 0          |
| v0.2.6  | 100        | 500        | 2048           | auto    | 250        | 0.0         | 1.2            | 0          |
| v0.2.7  | 100        | 500        | 2048           | auto    | 250        | 0.0         | 1.2            | 0          |
| v0.2.8  | 100        | 500        | 2048           | auto    | 250        | 0.0         | 1.2            | 0          |
| v0.3.0  | 100        | 500        | 2048           | auto    | 250        | 0.0         | 1.2            | 0          |


# Release Notes

Runtime memory profile for `v0.2.5`: a real Docker RAG request peaked at approximately `1.08 GB` Python process RSS
inside the container. This does not include platform/container overhead, so deployment targets should leave extra
headroom above that.

Performance profile for `v0.2.6`: batch perf over 20 sample AWS questions averaged `4.837s` per question
(`1.458s` min, `9.741s` max). The single-question perf run for the S3 bucket query completed in `11.174s`.

Diagnostics added in `v0.2.7`: production requests now write `TIMING` log lines for Streamlit request handling,
LLM client cold start/cache hits, model download/init, RAG asset loading, retrieval, context formatting, and LLM
completion. Docker now installs the CPU-only PyTorch wheel explicitly so linux/amd64 builds do not pull the CUDA
dependency stack.

Clean image tag `v0.2.8`: same corrected image as `v0.2.7`, tagged fresh after pruning pip build/cache artifacts from
the Docker layers. `docker image inspect` reports `1.22 GB`; `docker images` may report the larger virtual layer
footprint of approximately `3.97 GB`.

Release `v0.3.0`: restores the lazy-loading model flow in the Streamlit app. The user can explicitly load the model
before asking a question, and the `Load Model` and `Ask Question` buttons appear on the same line. This release is
intended to be tagged from a stable local build; no Docker image is pushed for this tag. Docker installs
`llama-cpp-python` from the maintainer CPU-wheel index and requires a binary wheel to avoid accidental source builds.
