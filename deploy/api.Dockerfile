# Demo API (demo/api.py) on a CPU server: the pipeline in the demo's light mode, no GPU.
#
#   docker build -f deploy/api.Dockerfile -t asila-api .      (from the repository root)
#
# requirements.txt pins the CUDA build of torch for Linux (and demo/requirements.txt includes it):
# here the CPU build is installed first and the torch lines of both files are left out.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WIUT_DEVICE=cpu \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

COPY requirements.txt /tmp/req/requirements.txt
COPY demo/requirements.txt /tmp/req/demo.txt
RUN pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
 && grep -vE '^(torch|torchvision)==|^--extra-index-url' /tmp/req/requirements.txt > /tmp/req/base.txt \
 && grep -vE '^-r ' /tmp/req/demo.txt > /tmp/req/demo-only.txt \
 && pip install -r /tmp/req/base.txt -r /tmp/req/demo-only.txt \
 && python -c "import torch; assert torch.version.cuda is None, torch.version.cuda" \
 && rm -rf /tmp/req

COPY . /app

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "-m", "demo.api", "--host", "0.0.0.0", "--port", "8000"]
