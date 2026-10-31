"""Settings come from environment variables (optionally loaded from the project's .env file)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _parse_value(v: str) -> str:
    v = v.strip()
    if v[:1] in ('"', "'"):  # quoted: take everything up to the closing quote
        q = v[0]
        end = v.find(q, 1)
        return v[1:end] if end != -1 else v[1:]
    for i, ch in enumerate(v):  # unquoted: strip an inline " # comment"
        if ch == "#" and (i == 0 or v[i - 1] in " \t"):
            v = v[:i]
            break
    return v.strip()


def _load_dotenv() -> None:
    """Load ROOT/.env. Values in .env win over empty/missing variables, real non-empty shell variables win over .env."""
    f = ROOT / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        k, v = line.split("=", 1)
        k, v = k.strip(), _parse_value(v)
        if v and not os.environ.get(k):
            os.environ[k] = v


_load_dotenv()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class Settings:
    """Read lazily so tests can change the environment."""

    @property
    def data_dir(self) -> Path:
        p = Path(env("NEXORA_DATA_DIR", str(ROOT / "data")))
        (p / "uploads").mkdir(parents=True, exist_ok=True)
        return p

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "nexora.db"

    @property
    def cors_origins(self) -> list[str]:
        raw = env("NEXORA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
        return [o.strip() for o in raw.split(",") if o.strip()]

    max_message_chars = property(lambda s: env_int("MAX_MESSAGE_CHARS", 8000))
    context_messages = property(lambda s: env_int("CONTEXT_MESSAGES", 20))
    request_timeout = property(lambda s: env_int("REQUEST_TIMEOUT_SECONDS", 60))
    max_upload_mb = property(lambda s: env_int("MAX_UPLOAD_MB", 10))
    max_rows = property(lambda s: env_int("MAX_ROWS", 200000))
    max_train_seconds = property(lambda s: env_int("MAX_TRAIN_SECONDS", 60))
    rate_limit_per_min = property(lambda s: env_int("RATE_LIMIT_PER_MINUTE", 120))
    upload_ttl_hours = property(lambda s: env_int("UPLOAD_TTL_HOURS", 168))
    system_prompt = property(lambda s: env(
        "SYSTEM_PROMPT",
        "You are Nexora AI, a helpful assistant inside an AI/NLP/analytics platform. "
        "You have no internet access. Be accurate and say when you are unsure."))


settings = Settings()
