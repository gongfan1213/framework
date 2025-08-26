import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict

try:
    from detoxify import Detoxify  # 毒性检测模型
except Exception:
    Detoxify = None


@dataclass
class Record:
    instruction: str
    input: str
    output: str
    meta: Dict


PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{11}\b"),  # 手机/身份证示例（按国家自定义）
    re.compile(r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}"),  # 邮箱
]


def hash_signature(text: str, n=5) -> str:
    tokens = text.split()
    window = " ".join(tokens[:n] + tokens[-n:])
    return hashlib.md5(window.encode()).hexdigest()


def is_toxic(text: str, threshold=0.7) -> bool:
    if Detoxify is None:
        return False
    scores = Detoxify('multilingual').predict(text)
    return max(scores.values()) > threshold


def has_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def semantic_dedup(records: List[Record]) -> List[Record]:
    seen = set()
    unique = []
    for r in records:
        sig = hash_signature((r.instruction or '') + ' ' + (r.output or ''))
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(r)
    return unique


def clean(records: List[Record]) -> List[Record]:
    cleaned = []
    for r in semantic_dedup(records):
        text = f"{r.instruction}\n{r.input}\n{r.output}"
        if has_pii(text):
            continue
        if is_toxic(text):
            continue
        cleaned.append(r)
    return cleaned


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        raw = [json.loads(line) for line in f]
    records = [Record(x.get('instruction',''), x.get('input',''), x.get('output',''), x) for x in raw]
    out = clean(records)
    print(json.dumps([r.__dict__ for r in out], ensure_ascii=False))
