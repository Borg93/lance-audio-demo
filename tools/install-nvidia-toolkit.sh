#!/usr/bin/env bash
#
# One-shot installer for the NVIDIA Container Toolkit on Ubuntu noble (24.04).
# Adds the apt repo, installs nvidia-container-toolkit, generates a CDI spec
# for the local GPUs, and runs a smoke test that prints `nvidia-smi` from a
# CUDA-base container.
#
# Usage:
#   bash tools/install-nvidia-toolkit.sh
#
# Re-runnable: each step is idempotent.

set -euo pipefail

KEY=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
ARCH=amd64
DISTRO_PATH="stable/deb/${ARCH}"

echo "→ 1/6  Importing NVIDIA GPG key into ${KEY}"
sudo rm -f "${KEY}"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o "${KEY}"
test -s "${KEY}" || { echo "FAIL: keyring not written"; exit 1; }

echo "→ 2/6  Writing apt source list to ${LIST}"
echo "deb [signed-by=${KEY}] https://nvidia.github.io/libnvidia-container/${DISTRO_PATH} /" \
    | sudo tee "${LIST}" > /dev/null
cat "${LIST}"

echo "→ 3/6  apt update"
sudo apt-get update -o Dir::Etc::sourcelist="${LIST}" \
                    -o Dir::Etc::sourceparts=- \
                    -o APT::Get::List-Cleanup=0

echo "→ 4/6  Installing nvidia-container-toolkit"
sudo apt-get install -y nvidia-container-toolkit

echo "→ 5/6  Generating CDI spec for local GPUs"
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list

echo "→ 6/6  Smoke-testing GPU access from a Docker container"
# Docker 27+ requires explicit CDI device names — `--gpus all` enumerates
# every vendor and fails on NVIDIA-only hosts with "AMD CDI spec not found".
docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

echo
echo "✓ Done. Next step:"
echo "    make embed-server-docker"
