#!/usr/bin/env python3
"""aiduMEI 域查询工具 — 跨域列出/检索记忆。

用法：
    python memq.py list                          # 列出所有域 + 记忆条数
    python memq.py search "关键词"                # 在所有域里搜（默认全扫）
    python memq.py search "关键词" --user hermes  # 只在 hermes 域搜
    python memq.py search "关键词" --user hermes --bank default
    python memq.py dump --user openclaw --limit 20   # 列出某域最近记忆

域 = (user_id, bank_id) 组合。当前环境：
    hermes   / default   Emma + Lisa
    openclaw / default   OpenClaw 六个 agent
    default  / default   遗留默认域
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("AIDUMEM_API_BASE", "http://127.0.0.1:8767").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
FACTS_DB = os.path.join(HERE, "data", "facts.db")


def _post(path: str, payload: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def registered_domains() -> list[tuple[str, str]]:
    """从 memory_banks 注册表读全部 (user_id, bank_id)。"""
    if not os.path.exists(FACTS_DB):
        return []
    try:
        conn = sqlite3.connect(f"file:{FACTS_DB}?mode=ro", uri=True)
        rows = conn.execute("SELECT user_id, bank_id FROM memory_banks ORDER BY user_id, bank_id").fetchall()
        conn.close()
        return [(str(r[0]), str(r[1])) for r in rows]
    except Exception:
        return []


def cmd_list(_args) -> int:
    domains = registered_domains()
    if not domains:
        print("未发现任何已注册的域（memory_banks 为空）")
        return 1
    print(f"{'user_id':<12} {'bank_id':<10} 状态")
    print("-" * 46)
    for uid, bid in domains:
        # 用一条空查询探活（服务端按域过滤）
        r = _post("/search", {"query": " ", "user_id": uid, "bank_id": bid, "limit": 1})
        state = "ok" if r.get("status") == "ok" else f"err: {r.get('error') or r.get('detail')}"
        print(f"{uid:<12} {bid:<10} {state}")
    return 0


def cmd_search(args) -> int:
    if args.user:
        targets = [(args.user, args.bank or "default")]
    else:
        targets = registered_domains() or [("default", "default")]

    total = 0
    for uid, bid in targets:
        r = _post("/search", {
            "query": args.query, "user_id": uid, "bank_id": bid,
            "limit": args.limit,
        })
        hits = r.get("results") or []
        rerank = (r.get("_rerank") or {}).get("applied")
        print(f"\n=== user_id={uid}  bank_id={bid}  ({len(hits)} 条, rerank={rerank}) ===")
        if r.get("error"):
            print(f"  ! {r['error']}: {str(r.get('detail'))[:150]}")
            continue
        if not hits:
            print("  (无命中)")
            continue
        for h in hits:
            score = h.get("score")
            s = f"{score:.4f}" if isinstance(score, (int, float)) else "  -   "
            text = (h.get("memory") or h.get("content") or "").replace("\n", " ")
            print(f"  [{s}] {text[:95]}")
            total += 1
    print(f"\n合计 {total} 条")
    return 0


def cmd_dump(args) -> int:
    """列出某域的记忆（用宽泛查询近似全量）。"""
    uid = args.user or "default"
    bid = args.bank or "default"
    r = _post("/search", {
        "query": args.query or "的记忆", "user_id": uid, "bank_id": bid,
        "limit": args.limit,
    })
    hits = r.get("results") or []
    print(f"=== {uid}/{bid} — {len(hits)} 条 ===")
    for i, h in enumerate(hits, 1):
        text = (h.get("memory") or h.get("content") or "").replace("\n", " ")
        print(f"{i:3}. {text[:110]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="aiduMEI 跨域查询")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有域").set_defaults(func=cmd_list)

    s = sub.add_parser("search", help="检索（默认扫所有域）")
    s.add_argument("query")
    s.add_argument("--user", default="")
    s.add_argument("--bank", default="")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("dump", help="列出某域记忆")
    d.add_argument("--user", default="")
    d.add_argument("--bank", default="")
    d.add_argument("--query", default="")
    d.add_argument("--limit", type=int, default=20)
    d.set_defaults(func=cmd_dump)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
