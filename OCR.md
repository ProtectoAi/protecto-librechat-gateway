# Local PDF and image OCR

The gateway provides `POST /v1/ocr` inside the existing
`protecto_librechat_gateway` Docker image. No LibreChat source changes, Azure
account, Mistral account, remote OCR API, or additional container is required.

## Enable the gateway

Generate a strong shared key locally, for example with `openssl rand -hex 32`.
Put the same generated value in both applications' `.env` files. Do not use a
provider API key and do not commit the key.

In `protecto_gateway_source/.env`:

```dotenv
OCR_ENABLED=true
OCR_API_KEY=REPLACE_WITH_GENERATED_KEY
OCR_LANGUAGES=eng
OCR_PDF_MODE=auto
```

The endpoint is disabled by default. Enabling OCR with a key shorter than 32
bytes fails startup. The existing Compose files already load these values via
`env_file`; do not copy the example over an existing `.env`.

Build and restart locally (brief downtime, per the project's startup policy):

```sh
./start.ksh
```

For the VM, publish the updated image using [DEPLOYMENT.md](DEPLOYMENT.md), update
the VM's gateway `.env`, then pull/recreate using `compose.remote.yml`:

```sh
docker compose --file compose.remote.yml pull
docker compose --file compose.remote.yml up --detach
```

## Configure LibreChat

Add `OCR_API_KEY` with that same value to LibreChat's `.env`. Merge
[librechat.ocr.example.yaml](librechat.ocr.example.yaml) into `librechat.yaml`.
Preserve existing `fileConfig` / `endpoints` keys; do not duplicate or replace
whole sections. If `endpoints.agents.capabilities` is explicitly configured,
add `context` and `ocr` to that list without removing its existing entries.

The `azure_mistral_ocr` strategy is used solely as an API adapter: LibreChat
posts base64 files directly to the configured `/v1/ocr`. The gateway performs
OCR with local Tesseract, not Azure or Mistral. `mistral_ocr` is deliberately
not used because it first calls `/files` and requests a signed download URL.
`custom_ocr` is not implemented in the inspected LibreChat version.

On macOS Docker Desktop, `host.docker.internal` is provided automatically.
On Ubuntu Docker, merge this into LibreChat's Docker Compose override for the
**api** service (not just `rag_api`):

```yaml
services:
  api:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

The URL assumes both applications run on the same host and the gateway
publishes port 8000. A shared private Docker network can instead use the gateway
service DNS name; change both `baseURL` and `allowedAddresses` to match its
exact host and port. Do not use `localhost` inside the LibreChat container.

Recreate LibreChat after editing its environment/configuration:

```sh
docker compose up --detach --force-recreate api
```

Use **Upload as Text** in chat or **File Context** for an agent. This does not
change the gateway's existing rejection of **Upload to Provider** files.
Image uploads must be sent through the text/OCR path, not the vision attachment
path. Word/Excel/PowerPoint types are intentionally excluded and remain on
LibreChat's document-parsing path. HEIC/HEIF and SVG are not supported here;
convert them to PNG/JPEG first.

References: [LibreChat OCR configuration](https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/ocr)
and [file-type routing](https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/file_config).
The base64 request/response contract was also checked against local LibreChat
`packages/api/src/files/mistral/crud.ts` (`uploadAzureMistralOCR`, `performOCR`,
and `processOCRResult`).

## Processing and limits

| Gateway environment variable | Default | Meaning |
| --- | --- | --- |
| `OCR_ENABLED` | `false` | Enable authenticated OCR |
| `OCR_API_KEY` | empty | Shared LibreChat/gateway secret; minimum 32 bytes |
| `OCR_LANGUAGES` | `eng` | Installed Tesseract language codes, e.g. `eng+fra` |
| `OCR_PDF_MODE` | `auto` | Extract existing text per page; OCR pages with no text |
| `OCR_MAX_FILE_MB` | `20` | Maximum decoded file size, in MiB |
| `OCR_MAX_PAGES` | `50` | Maximum PDF pages or image frames |
| `OCR_TIMEOUT_SECONDS` | `180` | Total worker wall-clock limit |
| `OCR_MAX_CONCURRENT` | `1` | Active uploads/jobs per gateway process; excess returns 429 |
| `OCR_MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded image pixels per frame |
| `OCR_RENDER_MAX_SIDE` | `4000` | Maximum rendered PDF/image side length in pixels |
| `OCR_MAX_TEXT_CHARS` | `2000000` | Maximum extracted characters per document |
| `OCR_WORKER_MEMORY_MB` | `1024` | Linux worker virtual-memory limit, inherited by children |

Only English is installed in the default Docker image. Additional language
packages (e.g. `tesseract-ocr-fra`) must be installed in the Dockerfile before
setting `OCR_LANGUAGES=eng+fra` and rebuilding. Tesseract accuracy depends on
scan quality and layout; this returns text, not faithful table/layout replicas.

`auto` preserves searchable PDF text and OCRs image-only pages. On mixed pages
with both existing text and scanned text, it uses only existing text. Set
`OCR_PDF_MODE=always` to OCR the full rendered page instead. This is slower and
can introduce recognition errors in otherwise searchable text.

Uploads have a 30-second body-read timeout. The worker has a separate total
timeout, a 64 MiB minimum per-output-file limit, and a Linux memory limit.
It is killed together with parser children on timeout/cancellation/disconnect.
Limits fail explicitly; the gateway does not silently truncate page counts or
return partially processed documents. LibreChat/reverse-proxy upload timeouts
and file size limits also apply and may be lower.

## Privacy and diagnostics

OCR sends no document data to external services. Only inline base64 input is
accepted; HTTP URLs, local file paths, and arbitrary downloads are rejected.
Temporary input, rendered pages and extracted text are removed after each
request. They are not stored in Redis, artifacts or a gateway history cache.
An abrupt container/host crash can prevent cleanup; do not persist the
container's temporary directory, and use encrypted disks for sensitive data.

The OCR response contains **original unmasked text** for LibreChat to store and
use as file context. Masking still happens later when the selected chat endpoint
sends that context through Protecto. Choosing another endpoint may bypass that
masking. Keep port 8000 private/firewalled; use TLS if traffic crosses hosts.
The API key is a service credential, not per-user authorization.

Gateway logs contain only `[OCR START]`, `[OCR COMPLETE]`, `[OCR REJECTED]`,
`[OCR TIMEOUT]`, `[OCR DISCONNECTED]` or `[OCR FAILED]` operational metadata: request ID, MIME type,
byte/page counts, elapsed time and status. They do not log uploaded bytes,
file names, extracted text, keys or parser diagnostics.

This route returns a normal final OCR response. It does not send chat progress
or alter masking progress, tool calls, history reconstruction or artifacts.

## Tests

```sh
python -m unittest discover -s tests -v
OCR_RUN_INTEGRATION=true python -m unittest discover -s tests -p 'test_ocr*.py' -v
```

Integration tests require Poppler, Tesseract and the pinned Python dependencies.
They use synthetic image, scanned-PDF and searchable-PDF fixtures; no real
customer documents or external APIs.
