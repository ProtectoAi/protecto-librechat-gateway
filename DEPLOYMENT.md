# Protecto LibreChat Gateway Private Registry Deployment

The Docker image is named `protecto_librechat_gateway`. Publish it to a private
Docker Hub repository from the laptop, then pull it from the remote VM. No
offline image bundle is created or transferred.

For optional local PDF/image OCR and the LibreChat configuration, see
[OCR.md](OCR.md). OCR is included in the image but disabled until configured.

## 1. Create private Docker Hub access

1. In the enterprise Docker organization, create a private repository named
   `protecto_librechat_gateway`.
2. Give the VM operator's Docker account or team read-only access.
3. Create personal access tokens separately for the publisher and VM operator.
   Do not share an account password or token between machines.

## 2. Publish from the laptop

Run these commands from `protecto_gateway_source`. Replace the namespace and
version with the enterprise organization and release version.

```sh
docker login --username LAPTOP_DOCKER_USERNAME

DOCKER_NAMESPACE=YOUR_ENTERPRISE_ORGANIZATION
IMAGE_VERSION=1.0.0

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag "$DOCKER_NAMESPACE/protecto_librechat_gateway:$IMAGE_VERSION" \
  --tag "$DOCKER_NAMESPACE/protecto_librechat_gateway:latest" \
  --push \
  .

docker buildx imagetools inspect \
  "$DOCKER_NAMESPACE/protecto_librechat_gateway:$IMAGE_VERSION"
```

The inspection output must contain both `linux/amd64` and `linux/arm64`.

## 3. Copy the small deployment configuration

Only the Compose configuration and environment example are needed on the VM:

```sh
ssh ubuntu@VM_HOST 'mkdir -p ~/protecto_librechat_gateway'

scp compose.remote.yml .env.remote.example \
  ubuntu@VM_HOST:protecto_librechat_gateway/
```

## 4. Install and start on the remote VM

Log in with the Docker account that has repository read access:

```sh
ssh ubuntu@VM_HOST
docker login --username VM_DOCKER_USERNAME

cd ~/protecto_librechat_gateway
cp .env.remote.example .env
```

Edit `.env` and set the exact published image:

```text
GATEWAY_IMAGE=YOUR_ENTERPRISE_ORGANIZATION/protecto_librechat_gateway:1.0.0
```

`.env` only holds `GATEWAY_IMAGE` and `GATEWAY_PORT`. Every other gateway
setting — Protecto connection, model catalog, asynchronous masking, OCR,
and RAG embeddings — is inlined directly as editable values in
`compose.remote.yml`'s `environment:` block. Edit that file on the VM to
change any of them, for example the large-context masking defaults:

```yaml
ASYNC_MASK_THRESHOLD_KB: "10"
ASYNC_MASK_POLL_SECONDS: "10"
ASYNC_MASK_MAX_RETRIES: "6"
ASYNC_MASK_SCALE_POLL_SECONDS: "true"
ASYNC_MASK_SCALE_RETRIES: "true"
```

The threshold is the baseline for proportional scaling. With these values, a
20 KiB payload uses a 20-second polling interval and 12 retries. Set either
scaling flag to `false` to keep that parameter at its configured base value.

Pull and start in detached mode:

```sh
docker compose --file compose.remote.yml pull
docker compose --file compose.remote.yml up --detach
docker compose --file compose.remote.yml ps
curl --fail http://127.0.0.1:8000/health
```

## LibreChat RAG with Protecto-masked embeddings

LibreChat's official lite RAG API performs extraction and chunking, then calls
this gateway's OpenAI-compatible `/v1/embeddings` endpoint. The gateway masks
each chunk or query through Protecto before requesting its vector from OpenAI.
The embeddings response contains vectors only, so LibreChat keeps the original
chunk text in its own vector store.

Configure the gateway in `compose.remote.yml`:

```yaml
OPENAI_EMBEDDINGS_URL: "https://api.openai.com/v1/embeddings"
RAG_PROTECTO_USER_ID: "librechat-rag-service"
```

Configure the shared LibreChat/RAG `.env`:

```text
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small
RAG_OPENAI_BASEURL=http://host.docker.internal:8000/v1
RAG_OPENAI_API_KEY=replace-with-a-real-openai-api-key
RAG_CHECK_EMBEDDING_CTX_LENGTH=false
```

The gateway forwards LibreChat's `RAG_OPENAI_API_KEY` bearer credential to
OpenAI as-is after masking the input — it is the real OpenAI API key, not a
separate gateway-issued credential. There is no gateway-side embeddings key
to configure.

`RAG_CHECK_EMBEDDING_CTX_LENGTH=false` is required so the RAG API sends text
rather than integer token arrays. The gateway cannot mask token IDs.

Both document chunks and retrieval queries use the stable
`RAG_PROTECTO_USER_ID`; do not change this identity after indexing documents.
No RAG setting is required in `librechat.yaml`. If the Agents capabilities list
is customized there, retain `file_search`.

## 5. Operations

View logs:

```sh
docker compose --file compose.remote.yml logs --follow
```

Stop while preserving the image and log volume:

```sh
docker compose --file compose.remote.yml down
```

Deploy a new version by updating `GATEWAY_IMAGE` in `.env`, then run:

```sh
docker compose --file compose.remote.yml down
docker compose --file compose.remote.yml pull
docker compose --file compose.remote.yml up --detach
```
