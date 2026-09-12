"""aiduMEI 双域隔离验证：hermes（Emma/Lisa） vs openclaw（OpenClaw agent）"""
import json
import urllib.request

BASE = "http://127.0.0.1:8767"


def call(path, payload):
    r = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(r, timeout=120) as x:
        return json.loads(x.read().decode())


MARK = "双域隔离探针-zz91"
print("=== 1. 写入 hermes 域 ===")
a = call("/add", {"messages": f"{MARK} 属于 Emma/Lisa。", "user_id": "hermes", "bank_id": "default"})
print("  add:", a.get("status"), a.get("action"))

print("=== 2. 写入 openclaw 域 ===")
b = call("/add", {"messages": f"{MARK} 属于 OpenClaw。", "user_id": "openclaw", "bank_id": "default"})
print("  add:", b.get("status"), b.get("action"))

print()
print("=== 3. 各域检索 ===")
for uid in ("hermes", "openclaw", "default"):
    r = call("/search", {"query": MARK, "user_id": uid, "bank_id": "default", "limit": 5})
    hits = r.get("results") or []
    print(f"  user_id={uid:9} verdict={r.get('recall_verdict'):9} hits={len(hits)}")
    for h in hits[:2]:
        print(f"      -> user_id={h.get('user_id')} | {(h.get('memory') or '')[:45]}")

print()
print("=== 4. 清理 ===")
for uid in ("hermes", "openclaw"):
    c = call("/delete_all", {"user_id": uid, "bank_id": "default", "confirm": True})
    print(f"  delete_all({uid}):", c.get("status"))
