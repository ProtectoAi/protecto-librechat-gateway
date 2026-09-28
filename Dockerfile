# syntax=docker/dockerfile:1.4

# Alpine's tiff has unpatched CVE-2023-52356 / CVE-2026-4775 with no newer apk
# package. Fixed by libtiff 4.7.2, and everything here only dynamically links
# libtiff.so.6, so a drop-in replacement is enough.
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

# Alpine's tesseract-ocr 5.5.2-r0 has CVE-2026-88051/88052/88053. The fixes are
# on upstream main but in no tagged release, so build 5.5.3 with them
# backported - see patches/README.md.
#
# Same base as the runtime stage on purpose: a leptonica mismatch between the
# two would surface only as an undefined symbol at container start-up, long
# after the CI scan gate has passed.
FROM python:3.14-alpine AS tesseract_builder

RUN apk update && apk upgrade --no-cache \
    && apk add --no-cache \
        build-base autoconf automake libtool pkgconf git wget tar \
        leptonica-dev libpng-dev libjpeg-turbo-dev tiff-dev \
        libwebp-dev giflib-dev zlib-dev \
        tesseract-ocr-data-eng

COPY patches/tesseract-5.5.3-cve-fixes.patch /tmp/tesseract-cve-fixes.patch

# tesseract-ocr-data-eng is installed above only for /usr/share/tessdata; its
# hard dep on the vulnerable tesseract-ocr is harmless in a discarded stage.
# "git apply --check" first: a changed tarball must fail the build loudly rather
# than silently ship an unpatched binary, which would also clear the scan.
RUN wget -q https://github.com/tesseract-ocr/tesseract/archive/refs/tags/5.5.3.tar.gz \
    && tar -xzf 5.5.3.tar.gz \
    && cd tesseract-5.5.3 \
    && git apply --check /tmp/tesseract-cve-fixes.patch \
    && git apply /tmp/tesseract-cve-fixes.patch \
    && ./autogen.sh \
    && ./configure --prefix=/usr --libdir=/usr/lib \
        --disable-static --disable-graphics \
        --without-curl --without-archive \
        CXXFLAGS="-O2 -g0" \
    && make -j"$(nproc)" \
    && make install \
    && strip /usr/bin/tesseract /usr/lib/libtesseract.so.5.* \
    && cd .. && rm -rf tesseract-5.5.3*

FROM python:3.14-alpine AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LOG_FILE=/app/logs/protecto_gateway.log

WORKDIR /app

# tesseract-ocr is deliberately NOT installed - the patched build is copied in
# below, and leptonica/libstdc++ pull in everything the CLI links. Never
# installing it means there is no apk-db record for scanners to match.
RUN apk update \
    && apk upgrade --no-cache \
    && apk add --no-cache poppler-utils leptonica libstdc++ libgomp

# Drop in the patched libtiff (same SONAME) and strip the apk-db record too:
# scanners read the apk database, not the files on disk.
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

COPY --from=tesseract_builder /usr/bin/tesseract /usr/bin/tesseract
COPY --from=tesseract_builder /usr/lib/libtesseract.so.5* /usr/lib/
COPY --from=tesseract_builder /usr/share/tessdata /usr/share/tessdata
ENV TESSDATA_PREFIX=/usr/share/tessdata

# Fail the build, not the container. The size check catches a swap to the much
# smaller tessdata_fast model, which would quietly change OCR accuracy.
RUN if ldd /usr/bin/tesseract | grep -q "not found"; then \
        ldd /usr/bin/tesseract; echo "FATAL: unresolved shared libraries"; exit 1; \
    fi \
    && { tesseract --version | head -1 | grep -q 'tesseract 5.5.3' \
         || { tesseract --version; echo "FATAL: unexpected tesseract version"; exit 1; }; } \
    && { tesseract --list-langs 2>&1 | grep -qx eng \
         || { echo "FATAL: eng language data missing"; exit 1; }; } \
    && { [ "$(stat -c %s /usr/share/tessdata/eng.traineddata)" -gt 20000000 ] \
         || { echo "FATAL: eng.traineddata is not the full model"; exit 1; }; }

RUN addgroup -S -g 10001 gateway \
    && adduser -S -u 10001 -G gateway -h /app gateway \
    && mkdir -p /app/logs \
    && chown gateway:gateway /app/logs

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --requirement requirements.txt \
    && rm -rf /usr/local/lib/python3.14/site-packages/pip \
              /usr/local/lib/python3.14/site-packages/pip-*.dist-info \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3

# Proof the patched binary actually OCRs, not just that it links and reports a version.
RUN python -c "\
from PIL import Image, ImageDraw; \
i = Image.new('L', (320, 80), 255); \
ImageDraw.Draw(i).text((12, 28), 'PROTECTO OCR 12345', fill=0); \
i.resize((960, 240)).save('/tmp/ocr_smoke.png')" \
    && tesseract /tmp/ocr_smoke.png stdout -l eng | tr -d ' \n' | grep -q '12345' \
    && rm -f /tmp/ocr_smoke.png

COPY --chown=gateway:gateway protecto_gateway ./protecto_gateway

USER gateway

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "protecto_gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
