# syntax=docker/dockerfile:1.4

# ─── Stage: build a non-vulnerable libtiff from source ─────────────────────
# Alpine's packaged tiff (4.7.1-r0 as of this build) has two open, unpatched
# CVEs (CVE-2023-52356, CVE-2026-4775) with no newer apk package available.
# Both are fixed upstream by libtiff 4.7.2; poppler-utils/tesseract/leptonica
# only dynamically link libtiff.so.6, so a drop-in library replacement is
# enough — no need to rebuild the rest of the OCR stack from source.
FROM alpine:3.24 AS tiff_builder

RUN apk update && apk upgrade --no-cache \
    && apk add --no-cache \
        build-base wget tar \
        zlib-dev libjpeg-turbo-dev libwebp-dev zstd-dev

RUN wget -q https://download.osgeo.org/libtiff/tiff-4.7.2.tar.gz \
    && tar -xzf tiff-4.7.2.tar.gz \
    && cd tiff-4.7.2 \
    && ./configure --prefix=/usr --libdir=/usr/lib --enable-shared --disable-static \
    && make -j"$(nproc)" \
    && make install \
    && cd .. && rm -rf tiff-4.7.2*

# ─── Runtime ────────────────────────────────────────────────────────────────
FROM python:3.13-alpine AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_FILE=/app/logs/protecto_gateway.log

WORKDIR /app

RUN apk update \
    && apk upgrade --no-cache \
    && apk add --no-cache poppler-utils tesseract-ocr tesseract-ocr-data-eng

# Drop in the patched libtiff over the apk-installed one (same SONAME: libtiff.so.6).
# Remove the old vulnerable .so file and the package's apk-db record so vulnerability
# scanners (which read the apk database, not file contents) see the real state:
# no apk-tracked "tiff" package, just the source-built 4.7.2 library on disk.
RUN rm -f /usr/lib/libtiff.so.6 /usr/lib/libtiff.so.6.2.0
COPY --from=tiff_builder /usr/lib/libtiff.so.6.* /usr/lib/
RUN cd /usr/lib && ln -sf libtiff.so.6.*.* libtiff.so.6
COPY <<'EOF' /tmp/drop_apk_pkg.py
import sys

path = "/lib/apk/db/installed"
name = sys.argv[1]
with open(path) as f:
    blocks = f.read().split("\n\n")

kept = [b for b in blocks if not any(line == f"P:{name}" for line in b.split("\n"))]
if len(kept) == len(blocks):
    raise SystemExit(f"package {name} not found in apk db")

with open(path, "w") as f:
    f.write("\n\n".join(kept))
print(f"removed apk db record for {name}")
EOF
RUN python3 /tmp/drop_apk_pkg.py tiff && rm /tmp/drop_apk_pkg.py

RUN addgroup -S -g 10001 gateway \
    && adduser -S -u 10001 -G gateway -h /app gateway \
    && mkdir -p /app/logs \
    && chown gateway:gateway /app/logs

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --requirement requirements.txt \
    && rm -rf /usr/local/lib/python3.13/site-packages/pip \
              /usr/local/lib/python3.13/site-packages/pip-*.dist-info \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3

COPY --chown=gateway:gateway protecto_gateway ./protecto_gateway

USER gateway

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "protecto_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
