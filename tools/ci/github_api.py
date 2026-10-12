"""A minimal GitHub REST client for the CI scripts (standard library only, so a hosted job runs it without
installing anything)."""

from __future__ import annotations

import json
import urllib.request
from typing import Callable, Optional


class GitHub:
    def __init__(self, repo: str, token: str, api: str = "https://api.github.com",
                 send: Optional[Callable[[str, str, Optional[dict]], dict]] = None):
        self.repo, self.token, self.api = repo, token, api.rstrip("/")
        self.send = send or self._send

    def _send(self, method: str, path: str, body: Optional[dict]) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.api}{path}", data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "lithos-metal-ci"})
        with urllib.request.urlopen(request, timeout=30) as r:
            raw = r.read()
        return json.loads(raw) if raw else {}

    def get(self, path: str) -> dict:
        return self.send("GET", f"/repos/{self.repo}{path}", None)

    def post(self, path: str, body: dict) -> dict:
        return self.send("POST", f"/repos/{self.repo}{path}", body)
