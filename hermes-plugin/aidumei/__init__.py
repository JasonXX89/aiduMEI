"""aiduMEI memory provider for Hermes Agent (current MemoryProvider contract).

Wraps the aiduMEI REST service (default http://127.0.0.1:8767):

    prefetch          -> POST /search  (relevance guess by query length)
    sync_turn         -> POST /add     (background thread, never blocks the turn)
    on_pre_compress   -> POST /add/raw (salvage turns about to be compressed)
    on_session_end    -> POST /session/end
    on_memory_write   -> POST /add (mirror Hermes MEMORY.md writes)
    get_tool_schemas  -> aidumem_search / aidumem_remember / aidumem_status
    backup_paths      -> aiduMEI data dir for `hermes backup`

Config: $HERMES_HOME/aidumei/config.json (written by the setup panel), else env
AIDUMEM_URL / AIDUMEM_USER_ID / AIDUMEM_BANK_ID / AIDUMEM_DATA_DIR.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from agent.memory_provider import MemoryProvider, RecallStatus

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8767"
_CONNECT_TIMEOUT = 2.0       # is_available(): must not slow startup
_QUERY_TIMEOUT = 6.0         # prefetch blocks the turn start: keep short
_WRITE_TIMEOUT = 20.0        # writes run on background threads
_MIN_QUERY_LEN = 6           # below this, no recall (saves a round trip)
_MAX_CONTEXT_CHARS = 4000

# Host-side boundary neutralization — same vocabulary as the server-side
# ducky.security.injection_guard. Change one, change the other.
_BOUNDARY_MARKERS = (
    "<<<RECORD_START", "<<<RECORD_END", "[END OF DATA CONTEXT]", "[DATA:",
    "<memory>", "</memory>", "[以下为召回的记忆数据",
)


def _neutralize_markers(text: str) -> str:
    """Break up boundary markers with a zero-width joiner so recalled text can
    never forge a prompt boundary. Reads identically to a human."""
    for marker in _BOUNDARY_MARKERS:
        if marker in text:
            text = text.replace(marker, marker[:2] + "\u200d" + marker[2:])
    return text


def _config_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = Path(get_hermes_home())
    except Exception:
        home = Path.home() / ".hermes"
    return home / "aidumei" / "config.json"


def _load_config() -> dict:
    p = _config_path()
    if p.exists():
        with suppress(Exception):
            return json.loads(p.read_text(encoding="utf-8"))
    return {
        "url": os.environ.get("AIDUMEM_URL", DEFAULT_URL),
        "user_id": os.environ.get("AIDUMEM_USER_ID", "default"),
        "bank_id": os.environ.get("AIDUMEM_BANK_ID", "default"),
        "data_dir": os.environ.get("AIDUMEM_DATA_DIR", ""),
    }


class AiduMeiMemoryProvider(MemoryProvider):
    """aiduMEI long-term memory: hybrid vector + FTS5 recall over a local service."""

    def __init__(self) -> None:
        cfg = _load_config()
        self._url = str(cfg.get("url") or DEFAULT_URL).rstrip("/")
        self._user_id = str(cfg.get("user_id") or "default")
        self._bank_id = str(cfg.get("bank_id") or "default")
        self._data_dir = str(cfg.get("data_dir") or "")
        self._token = os.environ.get("AIDUMEM_API_TOKEN", "")
        self._session_id = ""
        self._writes: List[threading.Thread] = []
        self._writes_lock = threading.Lock()
        self._prefetched: Dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        self._last_recall_count = 0
        self._last_recall_returned = False

    # -- identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "aidumei"

    def is_available(self) -> bool:
        """No network: a configured URL is enough. The service may come up later."""
        return bool(self._url)

    def unavailable_reason(self) -> str:
        return ""

    # -- config surface ------------------------------------------------------

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "aidumei" / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if path.exists():
            with suppress(Exception):
                existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update(values)
        with suppress(Exception):
            from utils import atomic_json_write
            atomic_json_write(path, existing, mode=0o600)
            return
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "url", "description": "aiduMEI service URL", "default": DEFAULT_URL},
            {"key": "user_id", "description": "Memory namespace (user_id)", "default": "default"},
            {"key": "bank_id", "description": "Semantic workspace (bank_id)", "default": "default"},
            {"key": "data_dir", "description": "aiduMEI data directory (for backups)", "default": ""},
        ]

    # -- HTTP ----------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _post(self, path: str, payload: dict, timeout: float) -> Optional[dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self._url}{path}", data=data, headers=self._headers(), method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urlerror.URLError, urlerror.HTTPError, OSError, ValueError) as exc:
            logger.debug("aiduMEI %s failed: %s", path, exc)
            return None

    def _get(self, path: str, timeout: float) -> Optional[dict]:
        req = urlrequest.Request(f"{self._url}{path}", headers=self._headers(), method="GET")
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urlerror.URLError, urlerror.HTTPError, OSError, ValueError) as exc:
            logger.debug("aiduMEI GET %s failed: %s", path, exc)
            return None

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        for key in ("platform", "user_id", "agent_identity"):
            val = kwargs.get(key)
            if val:
                logger.debug("aiduMEI init kwarg %s=%s", key, val)

    def shutdown(self) -> None:
        with self._writes_lock:
            threads = list(self._writes)
        for t in threads:
            t.join(timeout=5.0)

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                          reset: bool = False, rewound: bool = False, **kwargs) -> None:
        self._session_id = new_session_id or ""

    # -- recall --------------------------------------------------------------

    def system_prompt_block(self) -> str:
        return ""

    def _search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        payload = {
            "query": query,
            "user_id": self._user_id,
            "bank_id": self._bank_id,
            "limit": limit,
        }
        res = self._post("/search", payload, _QUERY_TIMEOUT)
        if not res or res.get("status") != "ok":
            return []
        return res.get("results") or []

    def _format(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = []
        for r in results:
            text = (r.get("memory") or r.get("text") or r.get("content") or "").strip()
            if not text:
                continue
            score = r.get("score")
            prefix = f"[{score:.2f}] " if isinstance(score, (int, float)) else ""
            lines.append(f"- {prefix}{_neutralize_markers(text)}")
        if not lines:
            return ""
        body = "\n".join(lines)
        if len(body) > _MAX_CONTEXT_CHARS:
            body = body[:_MAX_CONTEXT_CHARS] + "\n…"
        return "<memory>\n" + body + "\n</memory>"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        key = session_id or self._session_id or "_"
        with self._prefetch_lock:
            cached = self._prefetched.pop(key, "")
        if cached:
            self._last_recall_returned = True
            return cached

        q = (query or "").strip()
        if len(q) < _MIN_QUERY_LEN:
            self._last_recall_returned = False
            self._last_recall_count = 0
            return ""

        results = self._search(q)
        self._last_recall_count = len(results)
        self._last_recall_returned = bool(results)
        return self._format(results)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the next turn's recall off the critical path."""
        q = (query or "").strip()
        if len(q) < _MIN_QUERY_LEN:
            return
        key = session_id or self._session_id or "_"

        def _run() -> None:
            block = self._format(self._search(q))
            if block:
                with self._prefetch_lock:
                    self._prefetched[key] = block

        threading.Thread(target=_run, name="aidumei-prefetch", daemon=True).start()

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._last_recall_returned:
            return None
        return RecallStatus(provider_label="aiduMEI", count=self._last_recall_count, glyph="⚕")

    # -- write ---------------------------------------------------------------

    def _add_async(self, content: str, metadata: Optional[dict] = None) -> None:
        if not content or not content.strip():
            return
        payload: Dict[str, Any] = {
            "messages": content.strip(),
            "user_id": self._user_id,
            "bank_id": self._bank_id,
            "async_mode": True,
        }
        if metadata:
            payload["metadata"] = metadata

        def _run() -> None:
            self._post("/add", payload, _WRITE_TIMEOUT)

        t = threading.Thread(target=_run, name="aidumei-add", daemon=True)
        with self._writes_lock:
            self._writes = [x for x in self._writes if x.is_alive()] + [t]
        t.start()

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        parts = [p for p in (user_content, assistant_content) if p and p.strip()]
        if not parts:
            return
        self._add_async(
            "\n".join(parts),
            {"source": "hermes.sync_turn", "session_id": session_id or self._session_id},
        )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Salvage turns about to be compressed into durable memory."""
        chunks = []
        for m in messages or []:
            role = m.get("role") if isinstance(m, dict) else None
            content = m.get("content") if isinstance(m, dict) else None
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                chunks.append(f"{role}: {content.strip()}")
        if not chunks:
            return ""
        self._add_async(
            "\n".join(chunks),
            {"source": "hermes.pre_compress", "session_id": self._session_id},
        )
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        url = f"{self._url}/session/end?session_id={self._session_id}&user_id={self._user_id}&bank_id={self._bank_id}"
        req = urlrequest.Request(url, data=b"{}", headers=self._headers(), method="POST")
        try:
            with urlrequest.urlopen(req, timeout=_WRITE_TIMEOUT):
                pass
        except (urlerror.URLError, urlerror.HTTPError, OSError) as exc:
            logger.debug("aiduMEI session/end failed: %s", exc)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        if action == "remove" or not (content or "").strip():
            return
        md = dict(metadata or {})
        md.update({"source": "hermes.memory_write", "target": target, "action": action})
        self._add_async(content, md)

    def backup_paths(self) -> List[str]:
        if self._data_dir and Path(self._data_dir).is_dir():
            return [str(Path(self._data_dir).resolve())]
        return []

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "aidumem_search",
                "description": "Search long-term memory in aiduMEI (hybrid vector + full-text recall).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for."},
                        "limit": {"type": "integer", "description": "Max results (default 5).", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "aidumem_remember",
                "description": "Store a durable fact or decision in aiduMEI long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to remember."},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "aidumem_status",
                "description": "Report aiduMEI service health, engine mode, and memory counts.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "aidumem_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "query required"}, ensure_ascii=False)
            try:
                limit = int(args.get("limit") or 5)
            except (TypeError, ValueError):
                limit = 5
            results = self._search(query, limit=max(1, min(limit, 20)))
            return json.dumps(
                {"count": len(results), "results": results}, ensure_ascii=False,
            )

        if tool_name == "aidumem_remember":
            content = str(args.get("content") or "").strip()
            if not content:
                return json.dumps({"error": "content required"}, ensure_ascii=False)
            res = self._post(
                "/add",
                {"messages": content, "user_id": self._user_id, "bank_id": self._bank_id},
                _WRITE_TIMEOUT,
            )
            return json.dumps(
                {"ok": bool(res), "response": res}, ensure_ascii=False,
            )

        if tool_name == "aidumem_status":
            health = self._get("/health", _CONNECT_TIMEOUT + 3.0)
            if not health:
                return json.dumps(
                    {"ok": False, "error": f"aiduMEI unreachable at {self._url}"},
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "ok": health.get("health_status") == "ok",
                    "health_status": health.get("health_status"),
                    "version": health.get("version"),
                    "engine_mode": health.get("engine_mode"),
                    "degraded": health.get("degraded"),
                    "url": self._url,
                    "user_id": self._user_id,
                    "bank_id": self._bank_id,
                },
                ensure_ascii=False,
            )

        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")


def register(ctx) -> None:
    """Register aiduMEI as a memory provider plugin."""
    ctx.register_memory_provider(AiduMeiMemoryProvider())
