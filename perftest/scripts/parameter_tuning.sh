#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

VERSION="${VERSION:-v0.2.0}"

for top_k in 1 2 3 4; do
  for max_tokens in 200 250 300 350; do
    change="top_k_${top_k}_tokens_${max_tokens}"
    echo "Running ${change}"

    RUN_PERF_TESTS=1 \
      VERSION="${VERSION}" \
      CHANGE="${change}" \
      MODEL_TOP_K="${top_k}" \
      MODEL_MAX_TOKENS="${max_tokens}" \
      .venv/bin/python perftest/scripts/single_perf.py
  done
done
