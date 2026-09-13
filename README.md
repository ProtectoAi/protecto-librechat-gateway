# Protecto LibreChat Gateway

Protecto LibreChat Gateway is a stateless, OpenAI-compatible FastAPI service
that sits between LibreChat and supported LLM providers. It masks sensitive
content through Protecto before calling the provider, unmasks the provider
response for the user, and preserves a masked replay record in a LibreChat
artifact so later turns can rebuild conversation history safely.

The gateway currently supports OpenAI and Gemini chat models, tool calls,
streaming, masked embeddings for LibreChat RAG, and optional local OCR for PDF
and image uploads.

## Request flow

```text
LibreChat
  -> Protecto Gateway
       -> rebuild masked history from prior artifacts
       -> mask the current prompt and eligible tool results
       -> OpenAI or Gemini
       -> unmask the provider response and tool arguments
       -> LibreChat response plus Behind the Scene artifact
```

The gateway does not keep conversation history in a database or cache. Prior
masked prompts, masked assistant responses, and the token map are encoded in
the `Behind the Scene` artifact persisted by LibreChat. The next request
replays that artifact, and the gateway reconstructs the masked LLM history from
it. Protecto authentication tokens are held only as short-lived in-process
objects and are not a conversation store.

## Main features

- Masks LibreChat-supplied system/developer instructions and new prompts before
  they reach the LLM provider.
- Uses Protecto `/mask` for small payloads and `/mask/async` for large ones.
- Streams masking progress while asynchronous masking is polled.
- Unmasks streamed and non-streamed assistant responses.
- Optionally masks tool results with `MASK_TOOL_RESULTS=true`.
- Normalizes and masks standard structured text parts and structured JSON tool
  results; unsupported non-text message content is rejected instead of passed
  through.
- Persists masked system/developer instructions in replay artifact metadata and
  restores them when a continuation omits them. Identical live instructions
  reuse the artifact copy; changed instructions are masked as new content.
- Leaves trusted tool metadata unchanged, including function names,
  descriptions, identifiers, and parameter schemas.
- Preserves tool-call-only turns so LibreChat can execute tools correctly.
- Rebuilds prior masked history from LibreChat artifacts.
- Publishes the configured model catalog through `GET /v1/models`.
- Masks text before requesting OpenAI embeddings for LibreChat RAG.
- Provides authenticated, local Tesseract OCR through `POST /v1/ocr`.
- Rejects raw provider file uploads and directs users to **Upload as Text**.

## Repository layout

```text
protecto_gateway_source/
├── protecto_gateway/       FastAPI application and gateway modules
├── tests/                  Unit and integration tests
├── Dockerfile              Production image definition
├── compose.yml             Local build and runtime configuration
├── compose.remote.yml      Private-registry VM deployment
├── start.ksh               Stop, build, and start locally
├── stop.ksh                Stop the local service
├── .env.example            Local configuration template
├── .env.remote.example     Remote configuration template
├── DEPLOYMENT.md           Private-registry deployment details
└── OCR.md                  OCR design, configuration, and limits
```

## Quick start

This walks through getting the gateway running locally and pointed at an
existing LibreChat deployment running the **official LibreChat Docker
image**. It assumes no prior familiarity with this repository.

Prerequisites:

- Docker with Docker Compose v2
- `git` and `curl`
- A LibreChat instance already running from the official
  [LibreChat](https://github.com/danny-avila/LibreChat) Docker image, with
  access to edit its `.env` and `librechat.yaml`
- A reachable Protecto service, plus a Protecto master token and namespace
- OpenAI and/or Gemini API keys, configured in LibreChat (not in this
  gateway) as described in [LibreChat configuration](#librechat-configuration)

### 1. Get the code

```sh
git clone <YOUR_ORG_OR_USER>/protecto_librechat_gateway.git
cd protecto_librechat_gateway
```

(Replace `<YOUR_ORG_OR_USER>` with this repository's actual GitHub path.)

### 2. Configure the gateway

```sh
cp .env.example .env
```

Edit `.env` and at minimum configure:

```dotenv
PROTECTO_URL=https://protecto.example.com
PROTECTO_MASTER_TOKEN=replace-with-protecto-master-token
PROTECTO_NAMESPACE=librechat_namespace_vault

GATEWAY_MODEL_PROVIDERS_MODELS={"OpenAI":["gpt-5.6-terra"],"Gemini":["gemini-3.5-flash"]}
```

See [Gateway configuration](#gateway-configuration) below for the full
variable reference. Do not put Protecto credentials in `librechat.yaml` or
LibreChat headers. The gateway reads them only from its own environment.

### 3. Start the gateway

```sh
./start.ksh
```

`start.ksh` always stops the existing Compose service first, rebuilds the
image, starts it in detached mode, and waits for the health check. This creates
the local image `protecto_librechat_gateway:local`.

Verify it:

```sh
curl --fail http://127.0.0.1:8000/health
docker compose --file compose.yml logs --follow
```

Stop it:

```sh
./stop.ksh
```

### 4. Point your LibreChat (official image) at the gateway

With the gateway running, edit the `.env` and `librechat.yaml` used by your
**official LibreChat Docker image** deployment (not this repository) so
LibreChat routes chat traffic through the gateway instead of calling OpenAI
or Gemini directly:

- Add the custom endpoint block from
  [LibreChat configuration](#librechat-configuration) to `librechat.yaml`,
  pointing `baseURL` at this gateway (`http://host.docker.internal:8000/v1`
  when LibreChat and the gateway run as separate Docker Compose projects on
  the same host).
- On Ubuntu Docker, add `extra_hosts: ["host.docker.internal:host-gateway"]`
  to LibreChat's `api` service so it can resolve that hostname (see
  [LibreChat configuration](#librechat-configuration) for the exact override).
- Set `OPENAI_API_KEY` / `GEMINI_API_KEY` in LibreChat's own `.env` as usual;
  LibreChat forwards them to the gateway via the `X-OpenAI-API-Key` /
  `X-Gemini-API-Key` headers defined in the custom endpoint block.
- Restart the LibreChat `api` container after editing its configuration.

Optional: to also route LibreChat RAG embeddings and file OCR through the
gateway, see [LibreChat RAG embeddings](#librechat-rag-embeddings) and
[LibreChat OCR](#librechat-ocr).

## Gateway configuration

### Protecto and model catalog

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `PROTECTO_URL` | Yes | None | Protecto service base URL |
| `PROTECTO_MASTER_TOKEN` | Yes | None | Gateway-only Protecto master credential |
| `PROTECTO_NAMESPACE` | Yes | None | Protecto vault namespace |
| `GATEWAY_MODEL_PROVIDERS_MODELS` | Yes | None | JSON object mapping provider names to model lists |
| `MASK_TOOL_RESULTS` | No | `false` in code | Mask pending tool-result content before provider calls |
| `NORMALIZE_TOOL_CALL_IDS` | No | `true` | Normalize tool call identifiers for LibreChat compatibility |
| `BUFFER_TEXT_WHEN_TOOLS` | No | `false` | Buffer provider text during tool-capable streaming turns |
| `LOG_SSE` | No | `false` | Enable detailed SSE diagnostics; leave off normally |

The supplied Compose examples set `MASK_TOOL_RESULTS=true`. Environment values
override code defaults.

The model catalog must be valid one-line JSON:

```dotenv
GATEWAY_MODEL_PROVIDERS_MODELS={"OpenAI":["gpt-5.6","gpt-5.6-terra"],"Gemini":["gemini-3.5-flash"]}
```

At runtime the gateway creates model identifiers in `Provider:model` form. If
LibreChat sends `X-Chat-Name: Secured-Chat`, `/v1/models` returns display IDs
such as `Secured-Chat-OpenAI:gpt-5.6-terra`. The gateway removes only that
request's chat-name prefix before routing the model.

### Asynchronous masking

| Variable | Default | Purpose |
| --- | --- | --- |
| `ASYNC_MASK_THRESHOLD_KB` | `10` | Payloads above this size use `/mask/async` |
| `ASYNC_MASK_POLL_SECONDS` | `10` | Base delay between `/async-status` checks |
| `ASYNC_MASK_MAX_RETRIES` | `6` | Base maximum status checks |
| `ASYNC_MASK_SCALE_POLL_SECONDS` | `true` | Scale the delay in proportion to payload size |
| `ASYNC_MASK_SCALE_RETRIES` | `true` | Scale retries in proportion to payload size |

The size is calculated from the UTF-8 JSON mask payload, including field and
JSON formatting bytes. Scaling uses the configured threshold as its baseline.
For example, with a 10 KiB threshold, a 20 KiB payload has a scale factor of
2. Disable either scaling flag when that parameter should remain fixed.

Protecto asynchronous statuses are `PENDING`, `IN-PROGRESS`, `FAILED`, and
`SUCCESS`. The gateway polls `/async-status`, reports estimated progress up to
99%, and reports 100% only after Protecto returns `SUCCESS`.

### OCR

OCR is included in the image but disabled by default. Generate a dedicated
shared key and place the same value in the gateway and LibreChat `.env` files:

```sh
openssl rand -hex 32
```

Gateway configuration:

```dotenv
OCR_ENABLED=true
OCR_API_KEY=replace-with-generated-value
OCR_LANGUAGES=eng
OCR_PDF_MODE=auto
OCR_MAX_FILE_MB=20
OCR_MAX_PAGES=50
OCR_TIMEOUT_SECONDS=180
OCR_MAX_CONCURRENT=1
```

In `auto` mode, searchable PDF pages use their existing text layer and pages
without text are rendered and processed by Tesseract. Only English language
data is installed in the standard image. See [OCR.md](OCR.md) for all resource
limits, privacy behavior, additional languages, and diagnostics.

## LibreChat configuration

The following is a representative custom endpoint. Merge it into the existing
`librechat.yaml` and preserve other endpoints and capabilities:

```yaml
endpoints:
  custom:
    - name: "Secured-Chat"
      baseURL: "http://host.docker.internal:8000/v1"
      apiKey: "OPENAI_API_KEY"
      models:
        default:
          - "OpenAI:gpt-5.6-terra"
        fetch: true
      headers:
        X-Chat-Name: "Secured-Chat"
        X-Conversation-ID: "{{LIBRECHAT_BODY_CONVERSATIONID}}"
        X-Message-ID: "{{LIBRECHAT_BODY_MESSAGEID}}"
        X-User-ID: "{{LIBRECHAT_USER_ID}}"
        X-User-Email: "{{LIBRECHAT_USER_EMAIL}}"
        X-User-Name: "{{LIBRECHAT_USER_NAME}}"
        X-User-Username: "{{LIBRECHAT_USER_USERNAME}}"
        X-OpenAI-API-Key: "${OPENAI_API_KEY}"
        X-Gemini-API-Key: "${GEMINI_API_KEY}"
      multiConvo: true
      titleConvo: true
      titleModel: "current_model"
      modelDisplayLabel: "Secured-Chat"
```

Provider API keys are passed transiently from LibreChat:

- OpenAI requests require `X-OpenAI-API-Key`.
- Gemini requests require `X-Gemini-API-Key`.
- Chat requests require `X-User-Username` for the per-user Protecto token.
- `X-Conversation-ID` and `X-Message-ID` provide request context but do not
  create a gateway-side conversation cache.

Restart LibreChat after changing `.env` or `librechat.yaml`.

### LibreChat OCR

Add this at the root of `librechat.yaml`, not inside the custom endpoint:

```yaml
ocr:
  strategy: "azure_mistral_ocr"
  baseURL: "http://host.docker.internal:8000/v1"
  apiKey: "${OCR_API_KEY}"
  mistralModel: "local-tesseract"
  allowedAddresses:
    - "host.docker.internal:8000"

fileConfig:
  ocr:
    supportedMimeTypes:
      - '^application/pdf$'
      - '^image/(png|jpeg|webp|gif|tiff|bmp)$'
```

`azure_mistral_ocr` is used only as LibreChat's inline/base64 transport
adapter. OCR runs locally in this gateway; Azure and Mistral are not contacted.

On Ubuntu, allow the LibreChat API container to resolve the host gateway:

```yaml
services:
  api:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Use **Upload as Text** or agent **File Context**. **Upload to Provider** sends
raw file bytes toward the LLM and is intentionally rejected by this gateway.

If a reverse proxy fronts LibreChat, its request-size and timeout limits must
permit the upload and OCR duration. For Nginx, a configuration aligned with the
default 20 MiB OCR limit is:

```nginx
client_max_body_size 25M;
proxy_connect_timeout 60s;
proxy_send_timeout 300s;
proxy_read_timeout 300s;
```

## LibreChat RAG embeddings

Configure LibreChat's RAG service to use the gateway as an OpenAI-compatible
embeddings endpoint:

```dotenv
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small
RAG_OPENAI_BASEURL=http://host.docker.internal:8000/v1
RAG_OPENAI_API_KEY=replace-with-openai-provider-api-key
RAG_CHECK_EMBEDDING_CTX_LENGTH=false
```

Gateway configuration:

```dotenv
OPENAI_EMBEDDINGS_URL=https://api.openai.com/v1/embeddings
RAG_PROTECTO_USER_ID=librechat-rag-service
```

The current gateway treats the incoming `RAG_OPENAI_API_KEY` bearer credential
as the OpenAI provider key and forwards it only after masking the embedding
input. It does not use `OPENAI_EMBEDDINGS_API_KEY` or store that provider key.
`RAG_CHECK_EMBEDDING_CTX_LENGTH=false` is required because Protecto masks text,
not integer token arrays.

Keep `RAG_PROTECTO_USER_ID` stable after indexing documents. Changing it can
change the Protecto token context used for document chunks and queries.

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Health and active feature settings |
| `GET` | `/v1/models` | OpenAI-compatible model catalog |
| `GET` | `/v1/models/{model_id}` | Retrieve one model descriptor |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/responses` | OpenAI-compatible Responses API adapter |
| `POST` | `/v1/embeddings` | Protecto-masked OpenAI embeddings |
| `POST` | `/v1/ocr` | Authenticated local PDF/image OCR |

Both chat endpoints support streaming and non-streaming requests. The gateway
is stateless and cannot resolve an OpenAI `previous_response_id` by itself;
LibreChat must replay the conversation content.

## Artifacts and history reconstruction

For a normal text response, the gateway appends a `Behind the Scene` HTML
artifact containing:

- the exact masked user content sent to the LLM;
- the provider's masked assistant response;
- sensitive values identified by Protecto for the readable preview;
- encoded replay data containing the masked user/assistant turn and token map;
- masked tool-result context when present.

LibreChat persists the artifact with the assistant message. On the next turn,
the gateway ignores the older unmasked UI messages and rebuilds that portion of
LLM history from the encoded artifact. Content after the most recent artifact,
including the latest prompt and tool continuation, is masked live.

A tool-call turn intentionally contains tool calls only and no artifact or
progress text. LibreChat executes the tool and sends the result back in a new
request. The eventual text response carries the artifact.

## Logging and privacy

The principal diagnostic tags include:

- `[LIBRECHAT REPLAY]` — message shape only, without content values.
- `[LIBRECHAT MASKED REPLAY PAYLOAD]` — complete payload after masking.
- `[ARTIFACT HISTORY REBUILT]` — artifact and reconstructed-message counts.
- `[PROTECTO MASK ROUTE]` — payload size and selected mask endpoint.
- `[PROTECTO ASYNC MASK STATUS]` — asynchronous polling status.
- `[OCR START]` / `[OCR COMPLETE]` — OCR operational metadata.
- `[TOOL BUFFER]` / `[SSE OUT TOOL]` — tool-call streaming diagnostics.

The masked replay payload may still be sensitive operational data. Restrict log
access, use protected storage, and apply an appropriate retention policy.
Provider API keys, Protecto credentials, OCR keys, and unmasked OCR text are not
written to application logs.

## Build and test

Run the test suite:

```sh
python -m unittest discover -s tests -v
```

Run OCR integration tests when Poppler and Tesseract are installed:

```sh
OCR_RUN_INTEGRATION=true \
  python -m unittest discover -s tests -p 'test_ocr*.py' -v
```

Build the production image:

```sh
docker build --tag protecto_librechat_gateway:local .
```

For multi-platform publishing and VM installation, follow
[DEPLOYMENT.md](DEPLOYMENT.md).

## Remote update

Set the published image in the VM's `.env`:

```dotenv
GATEWAY_IMAGE=YOUR_DOCKER_NAMESPACE/protecto_librechat_gateway:1.0.0
```

Then pull and recreate the service:

```sh
docker compose --file compose.remote.yml pull
docker compose --file compose.remote.yml up --detach
curl --fail http://127.0.0.1:8000/health
```

Compose replaces the running container when the image changes. Running
`docker compose down` first is optional and introduces avoidable downtime.

## Troubleshooting

### Protecto returns 401

Confirm `PROTECTO_URL`, `PROTECTO_MASTER_TOKEN`, and `PROTECTO_NAMESPACE` in the
gateway `.env`. Recreate the gateway container after changing environment
values. Do not send these values from LibreChat headers.

### Missing provider API key

Confirm the applicable LibreChat header is present and its referenced `.env`
variable is populated. Recreate the LibreChat API container after editing its
environment.

### Large PDF never reaches OCR

If there is no LibreChat database file record and no gateway OCR log, inspect
the reverse proxy for HTTP 413. Increase its request-body limit. A 20 MiB file
becomes larger when represented as base64, although the browser-to-LibreChat
request itself is multipart data.

### OCR works for PNG but not PDF

Use **Upload as Text**, verify the recorded MIME type is `application/pdf`, and
look for OCR errors in LibreChat's API logs. File Search uses the RAG parsing
path and does not invoke the configured gateway OCR endpoint.

### Tool call is not executed

Do not add visible assistant content to the same terminal turn as a tool call.
The gateway suppresses visible asynchronous-mask progress when tools are
enabled so LibreChat receives a clean `finish_reason: tool_calls` response.

### History appears masked

Look for `[ARTIFACT HISTORY REBUILT]`, `[ARTIFACT DATA DECODE]`, and unmask
errors. The gateway falls back to the provider's masked text when an unmask
request fails, rather than failing the entire streamed response.
