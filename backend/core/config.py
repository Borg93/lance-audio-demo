"""Typed application settings (pydantic-settings).

Read once via :func:`get_settings`; routers read it off ``state.settings`` (the
injected :class:`~backend.state.AppState`). Only env-varying values live here —
algorithmic constants (RRF k, probe tokens, column-exclude sets) stay as module
constants in their feature packages.

Env vars are ``RAUDIO_*`` (see aliases). ``cors_origins`` accepts either a JSON
list or a bare comma-separated string (``RAUDIO_CORS_ORIGINS=https://a,https://b``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # §4.4 names the encoder URLs MEDIA_EMBED_URL / MEDIA_RERANK_URL; the legacy
    # RAUDIO_* aliases stay accepted (AliasChoices) so existing .env/launch
    # scripts keep working through the transition.
    embed_url: str = Field(
        default="http://127.0.0.1:8001",
        validation_alias=AliasChoices("MEDIA_EMBED_URL", "RAUDIO_EMBED_URL"),
    )
    rerank_url: str = Field(
        default="http://127.0.0.1:8002",
        validation_alias=AliasChoices("MEDIA_RERANK_URL", "RAUDIO_RERANK_URL"),
    )
    host: str = Field(default="127.0.0.1", alias="RAUDIO_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="RAUDIO_PORT")
    db_path: Path = Field(default=Path("transcripts_v2.lance"), alias="RAUDIO_DB")
    # Multi-dataset serving (LANCE_MEDIA_MERGE §4.4): the registry root holds
    # one `<id>.lance` dir per dataset; `db_path`'s stem stays the default
    # dataset so the legacy single-DB routes keep their behavior.
    db_root: Path = Field(default=Path("."), alias="MEDIA_DB_ROOT")
    descriptor_dir: Path = Field(default=Path("config/descriptors"), alias="MEDIA_DESCRIPTOR_DIR")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], alias="RAUDIO_CORS_ORIGINS")

    # Optional S3 object-store backing (RASK_LANDING §4). Set MEDIA_S3_ENDPOINT to
    # serve datasets from MinIO / RustFS / AWS: the registry then lists + opens
    # under MEDIA_S3_DB_ROOT (an s3:// URI) with these storage_options. All unset
    # (the default) = the local-filesystem db_root path, byte-identical to before.
    s3_endpoint: str | None = Field(default=None, alias="MEDIA_S3_ENDPOINT")
    s3_access_key_id: str | None = Field(default=None, alias="MEDIA_S3_ACCESS_KEY_ID")
    s3_secret_access_key: str | None = Field(default=None, alias="MEDIA_S3_SECRET_ACCESS_KEY")
    s3_region: str = Field(default="us-east-1", alias="MEDIA_S3_REGION")
    s3_db_root: str | None = Field(default=None, alias="MEDIA_S3_DB_ROOT")

    @property
    def default_dataset_id(self) -> str:
        return self.db_path.stem

    @property
    def storage_options(self) -> dict[str, str] | None:
        """Lance ``storage_options`` for the object store (None = local filesystem).

        Path-style addressing is forced (``virtual_hosted_style_request=false``)
        because RustFS/MinIO reject virtual-hosted signing — verified live.
        """
        if not self.s3_endpoint:
            return None
        opts = {
            "endpoint": self.s3_endpoint,
            "region": self.s3_region,
            "allow_http": "true" if self.s3_endpoint.startswith("http://") else "false",
            "virtual_hosted_style_request": "false",
        }
        # Pin static credentials only when BOTH are supplied. Otherwise omit them
        # so Lance/object_store falls back to the standard AWS credential chain
        # (AWS_* env vars, ~/.aws, IAM role) — sending "" would be read as an
        # explicit empty key and break that chain (the normal K8s/rask deploy).
        if self.s3_access_key_id and self.s3_secret_access_key:
            opts["access_key_id"] = self.s3_access_key_id
            opts["secret_access_key"] = self.s3_secret_access_key
        return opts

    @property
    def registry_root(self) -> str:
        """The root the registry lists/opens datasets under — the S3 URI when
        configured, else the local ``db_root`` path."""
        if self.s3_endpoint and self.s3_db_root:
            return self.s3_db_root
        return str(self.db_root)
    # Externally-reachable origin for media URLs in MCP clip apps (LAN IP,
    # tunnel, reverse proxy). Unset = derive http://{host}:{port} locally.
    # AnyHttpUrl: this value lands verbatim in the clip app's CSP allow-list
    # and media src, so reject non-URL garbage at boot, not in the iframe.
    media_base_url: AnyHttpUrl | None = Field(default=None, alias="RAUDIO_MEDIA_BASE_URL")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
