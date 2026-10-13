# Official MoCha BF16 inference runtime. Third-party research source and model
# weights are fetched directly from their publishers when a disposable pod runs.
# No GPU is needed to build this image.
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOCHA_HOME=/workspace/mocha \
    HF_HUB_DISABLE_TELEMETRY=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system
COPY requirements-runtime.txt /opt/mocha/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /opt/mocha/requirements-runtime.txt \
    && python -c "import torch, torchvision, transformers, pandas, imageio, cupy, huggingface_hub; assert torch.__version__.startswith('2.2.0'),torch.__version__; assert torchvision.__version__.startswith('0.17.0'),torchvision.__version__; print('CPU import smoke passed, torch',torch.__version__)" \
    && python -m pip check
COPY scripts/ /opt/mocha/scripts/
COPY model-manifest.json /opt/mocha/model-manifest.json
RUN chmod 755 /opt/mocha/scripts/mocha-runtime \
    && python -m compileall -q /opt/mocha/scripts \
    && ln -s /opt/mocha/scripts/mocha-runtime /usr/local/bin/mocha-runtime
WORKDIR /workspace
# Intentionally inherit RunPod's original /start.sh configuration. This
# preserves SSH and JupyterLab; only the inference tooling is added.
