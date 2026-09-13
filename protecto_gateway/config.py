"""Gateway configuration, constants, logging, and prompts."""
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def env_flag(name: str, default: bool=False) -> bool:
    value=os.getenv(name)
    return default if value is None else value.strip().lower() in {"1","true","yes","on"}

def env_positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    value = default if raw_value is None else float(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value

def env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value

def validate_protecto_settings(
        protecto_url: str | None,
        master_token: str | None,
        namespace: str | None,
) -> tuple[str, str, str]:
    """Validate private Protecto connection settings loaded by the gateway."""
    normalized_url = (protecto_url or "").strip().rstrip("/")
    normalized_token = (master_token or "").strip()
    normalized_namespace = (namespace or "").strip()
    if not normalized_url or not normalized_token or not normalized_namespace:
        raise ValueError("Protecto gateway service configuration is incomplete")
    if (
        len(normalized_namespace) > 128
        or any(ord(char) < 32 for char in normalized_namespace)
    ):
        raise ValueError("PROTECTO_NAMESPACE is invalid")

    parsed_url = urlsplit(normalized_url)
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError("PROTECTO_URL must be an HTTP(S) service URL")
    return normalized_url, normalized_token, normalized_namespace

def parse_gateway_models(
        catalog_value: str | None,
) -> list[str]:
    """Build provider:model IDs from a two-level JSON environment value."""
    if not catalog_value:
        raise ValueError(
            "GATEWAY_MODEL_PROVIDERS_MODELS must contain a JSON object"
        )

    try:
        catalog = json.loads(catalog_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "GATEWAY_MODEL_PROVIDERS_MODELS must be valid JSON"
        ) from exc

    if not isinstance(catalog, dict) or not catalog:
        raise ValueError(
            "GATEWAY_MODEL_PROVIDERS_MODELS must be a non-empty JSON object"
        )

    model_ids: list[str] = []

    for provider, models in catalog.items():
        if not isinstance(provider, str):
            raise ValueError("Gateway model provider names must be strings")
        provider = provider.strip()
        if ":" in provider or any(ord(char) < 32 for char in provider):
            raise ValueError(f"Invalid gateway model provider: {provider!r}")
        if not provider:
            raise ValueError(f"Invalid gateway model provider: {provider!r}")

        if not isinstance(models, list) or not models:
            raise ValueError(
                f"Models for {provider} must be a non-empty JSON array"
            )

        for model in models:
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"Every model for {provider} must be a non-empty string"
                )
            model = model.strip()
            model_id = f"{provider}:{model}"
            if model_id not in model_ids:
                model_ids.append(model_id)
    return model_ids

LOG_FILE=os.getenv("LOG_FILE","./logs/protecto_gateway.log")
logger=logging.getLogger("protecto_gateway")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    formatter=logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console=logging.StreamHandler(); console.setFormatter(formatter);
    logger.addHandler(console)
    os.makedirs(os.path.dirname(LOG_FILE) or ".",exist_ok=True)
    file_handler=RotatingFileHandler(LOG_FILE,maxBytes=10*1024*1024,backupCount=5,encoding="utf-8")
    file_handler.setFormatter(formatter); logger.addHandler(file_handler)

OPENAI_API_URL="https://api.openai.com/v1/responses"
OPENAI_EMBEDDINGS_URL = os.getenv(
    "OPENAI_EMBEDDINGS_URL",
    "https://api.openai.com/v1/embeddings",
).strip()
RAG_PROTECTO_USER_ID = os.getenv("RAG_PROTECTO_USER_ID", "").strip()
GEMINI_INTERACTIONS_URL="https://generativelanguage.googleapis.com/v1/interactions"
MASK_TOOL_RESULTS=env_flag("MASK_TOOL_RESULTS",False)
NORMALIZE_TOOL_CALL_IDS=env_flag("NORMALIZE_TOOL_CALL_IDS",True)
LOG_SSE=env_flag("LOG_SSE",False)
BUFFER_TEXT_WHEN_TOOLS = env_flag("BUFFER_TEXT_WHEN_TOOLS", False)
GATEWAY_MODELS = parse_gateway_models(
    os.getenv("GATEWAY_MODEL_PROVIDERS_MODELS"),
)
PROTECTO_URL = os.getenv("PROTECTO_URL")
PROTECTO_MASTER_TOKEN = os.getenv("PROTECTO_MASTER_TOKEN")
PROTECTO_NAMESPACE = os.getenv("PROTECTO_NAMESPACE")
if PROTECTO_URL or PROTECTO_MASTER_TOKEN or PROTECTO_NAMESPACE:
    (
        PROTECTO_URL,
        PROTECTO_MASTER_TOKEN,
        PROTECTO_NAMESPACE,
    ) = validate_protecto_settings(
        PROTECTO_URL,
        PROTECTO_MASTER_TOKEN,
        PROTECTO_NAMESPACE,
    )
CHAT_NAME_HEADER = "x-chat-name"
ASYNC_MASK_THRESHOLD_KB = env_positive_float("ASYNC_MASK_THRESHOLD_KB", 10.0)
ASYNC_MASK_THRESHOLD_BYTES = int(ASYNC_MASK_THRESHOLD_KB * 1024)
ASYNC_MASK_POLL_SECONDS = env_positive_float("ASYNC_MASK_POLL_SECONDS", 10.0)
ASYNC_MASK_MAX_RETRIES = env_positive_int("ASYNC_MASK_MAX_RETRIES", 6)
ASYNC_MASK_SCALE_POLL_SECONDS = env_flag("ASYNC_MASK_SCALE_POLL_SECONDS", True)
ASYNC_MASK_SCALE_RETRIES = env_flag("ASYNC_MASK_SCALE_RETRIES", True)


@dataclass(frozen=True)
class OCRSettings:
    """Limits for local, authenticated OCR; one worker handles one document."""

    enabled: bool = False
    api_key: str = field(default="", repr=False)
    languages: str = "eng"
    pdf_mode: str = "auto"
    max_file_mb: int = 20
    max_pages: int = 50
    timeout_seconds: int = 180
    max_concurrent: int = 1
    max_image_pixels: int = 40_000_000
    render_max_side: int = 4000
    max_text_chars: int = 2_000_000
    worker_memory_mb: int = 1024

    def __post_init__(self) -> None:
        if self.enabled and len(self.api_key.encode("utf-8")) < 32:
            raise ValueError("OCR_API_KEY must contain at least 32 bytes when OCR_ENABLED=true")
        if not re.fullmatch(r"[a-zA-Z0-9_]+(?:\+[a-zA-Z0-9_]+)*", self.languages):
            raise ValueError("OCR_LANGUAGES must contain Tesseract language codes")
        if self.pdf_mode not in {"auto", "always"}:
            raise ValueError("OCR_PDF_MODE must be auto or always")
        for name in (
            "max_file_mb", "max_pages", "timeout_seconds", "max_concurrent",
            "max_image_pixels", "render_max_side", "max_text_chars", "worker_memory_mb",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"OCR {name} must be greater than zero")

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def max_body_bytes(self) -> int:
        return 4 * ((self.max_file_bytes + 2) // 3) + 65536


OCR_SETTINGS = OCRSettings(
    enabled=env_flag("OCR_ENABLED", False),
    api_key=os.getenv("OCR_API_KEY", "").strip(),
    languages=os.getenv("OCR_LANGUAGES", "eng").strip(),
    pdf_mode=os.getenv("OCR_PDF_MODE", "auto").strip(),
    max_file_mb=env_positive_int("OCR_MAX_FILE_MB", 20),
    max_pages=env_positive_int("OCR_MAX_PAGES", 50),
    timeout_seconds=env_positive_int("OCR_TIMEOUT_SECONDS", 180),
    max_concurrent=env_positive_int("OCR_MAX_CONCURRENT", 1),
    max_image_pixels=env_positive_int("OCR_MAX_IMAGE_PIXELS", 40_000_000),
    render_max_side=env_positive_int("OCR_RENDER_MAX_SIDE", 4000),
    max_text_chars=env_positive_int("OCR_MAX_TEXT_CHARS", 2_000_000),
    worker_memory_mb=env_positive_int("OCR_WORKER_MEMORY_MB", 1024),
)

BASE_SYSTEM_PROMPT="""
All Personally Identifiable Information (PII) and sensitive data have been masked.
Ensure that all responses include the specified tags without any omissions,
and maintain the integrity of the tags as they are presented.
Do not create or modify tags.
Since the information is already masked, there is no need for further
truncation or masking, even for PCI and PHI.
You may present the PII and sensitive information as it is.
Always preserve existing placeholders exactly
(e.g., <PER> ... </PER>, <URL> ... </URL>).
Never invent, create, modify, or remove any placeholder tags under any
circumstances.
Provide real, factual, contextually appropriate information whenever no
placeholders are present, adapting tone as requested.
Do not explain or comment on placeholders.
"""
CHAT_STYLE_PROMPT="""
Always ask the user if they are happy with the response.
Always give some message before giving the actual response.
"""
TOOL_MODE_PROMPT="""
When you decide to call a tool, respond with the tool call ONLY.
Do not write any explanatory text before or alongside a tool call.
After tool results are provided, answer the user normally.
"""
def build_system_prompt(has_tools: bool)->str:
    return BASE_SYSTEM_PROMPT + (TOOL_MODE_PROMPT if has_tools else CHAT_STYLE_PROMPT)

ENTITIES=["PASSPORT_NO","PER","URL","PHN","CVV","PASSPORT","EMAIL","CRD","ADDRESS","DOB","ACC_NO","NATIONALITY","PAN","CRD_PIN","IPA","FAX_NO","INSURANCE_NO","POLICY_NO","SSN","ORG","CITY","COUNTRY","HEALTH_BENEFICIARY_NO","PASSWORD","DEATH_DATE","NATIONAL_ID","TAN","DL_NO","LICENCE_NO","VIN","PINCODE","MRN","ROUTING_NO","DISCHARGE_DATE","SWIFT","VEHICLE_REG_NO","DEVICE_ID","ADMIT_DATE"]
CHUNK_SIZE=max(map(len,ENTITIES)); OVERLAP=CHUNK_SIZE
START_TAGS={f"<{e}>":f"</{e}>" for e in ENTITIES}
END_TAG="PROTECTO_END"
token_value=defaultdict(dict)
ARTIFACT_RE=re.compile(r':::artifact\{[^}]*identifier="behind-the-scenes"[^}]*\}\s*(.*?)\s*:::',re.DOTALL)

GEMINI_SCHEMA_ALLOWED_KEYS={"type","format","description","nullable","enum","items","properties","required","anyOf","minItems","maxItems","propertyOrdering"}
GEMINI_ALLOWED_FORMATS={"string":{"enum","date-time"},"integer":{"int32","int64"},"number":{"float","double"}}
