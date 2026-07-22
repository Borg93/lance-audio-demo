"""The topics endpoint client — ratch's thin handle to the topics model service.

Pins the offline↔online seam: the local impl builds the correct sealed-env
command (without running it), and the factory selects remote vs local by env.
"""

from __future__ import annotations

from pathlib import Path

from ratch.endpoints.topics import (
    LocalTopicsClient,
    RemoteTopicsClient,
    get_topics_client,
)


class TestLocalTopicsClient:
    def test_command_targets_the_service_sealed_env(self) -> None:
        cmd = LocalTopicsClient().command(Path("db.lance"), llm_url="http://x:8003/v1")
        assert cmd[:3] == ["uv", "run", "--project"]
        assert cmd[3].endswith("services/models/topics")
        assert cmd[4] == "topics-worker"
        assert "--db" in cmd and "db.lance" in cmd
        assert cmd[-2:] == ["--llm-url", "http://x:8003/v1"]

    def test_command_omits_llm_url_when_unset(self) -> None:
        cmd = LocalTopicsClient().command(Path("db.lance"))
        assert "--llm-url" not in cmd


class TestFactory:
    def test_local_when_no_url(self, monkeypatch) -> None:
        monkeypatch.delenv("MEDIA_TOPICS_URL", raising=False)
        assert isinstance(get_topics_client(), LocalTopicsClient)

    def test_remote_when_url_set(self, monkeypatch) -> None:
        monkeypatch.setenv("MEDIA_TOPICS_URL", "http://topics:8000")
        assert isinstance(get_topics_client(), RemoteTopicsClient)
