"""API 사용량을 로컬 CSV에 기록하고 간단히 집계해서 보여주는 도구.

agent.py/evaluate.py가 OpenAI를 호출할 때마다 log_usage()로 한 줄씩 쌓이고,
`python usage.py`로 모델별 호출 횟수·토큰·(가격을 채워두면) 예상 비용을 볼 수 있다.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).parent / "data" / "api_usage_log.csv"
FIELDNAMES = ["timestamp", "node", "model", "prompt_tokens", "completion_tokens", "total_tokens"]

# 1,000 토큰당 USD 가격 (사용자 확인, 2026-09-17 기준. 캐시된 입력 토큰은 별도로
# 집계하지 않으므로 일반 입력 단가로만 계산한다 — 실제보다 비용이 다소 높게 잡힐 수 있음).
PRICING_PER_1K: dict[str, dict[str, float | None]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
}


def log_usage(node: str, model: str, usage) -> None:
    """OpenAI 응답의 usage 객체를 CSV 한 줄로 남긴다."""
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "node": node,
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        )


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["prompt_tokens"] = int(row["prompt_tokens"])
        row["completion_tokens"] = int(row["completion_tokens"])
        row["total_tokens"] = int(row["total_tokens"])
    return rows


def summarize_by_model() -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for row in load_log():
        bucket = summary.setdefault(
            row["model"],
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += row["prompt_tokens"]
        bucket["completion_tokens"] += row["completion_tokens"]
        bucket["total_tokens"] += row["total_tokens"]
    return summary


def summarize_by_node() -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for row in load_log():
        bucket = summary.setdefault(row["node"], {"calls": 0, "total_tokens": 0})
        bucket["calls"] += 1
        bucket["total_tokens"] += row["total_tokens"]
    return summary


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = PRICING_PER_1K.get(model)
    if not price or price["prompt"] is None or price["completion"] is None:
        return None
    return (prompt_tokens / 1000) * price["prompt"] + (completion_tokens / 1000) * price["completion"]


def print_report() -> None:
    by_model = summarize_by_model()
    if not by_model:
        print("기록된 API 호출이 없습니다. agent.py나 evaluate.py를 먼저 실행하세요.")
        return

    print("=== 모델별 사용량 ===")
    grand_total_tokens = 0
    grand_total_cost = 0.0
    cost_unknown = False

    for model, s in by_model.items():
        cost = estimate_cost(model, s["prompt_tokens"], s["completion_tokens"])
        grand_total_tokens += s["total_tokens"]
        if cost is None:
            cost_unknown = True
            cost_str = "가격 미설정"
        else:
            grand_total_cost += cost
            cost_str = f"${cost:.4f}"
        print(
            f"- {model}: 호출 {s['calls']}회, 입력 {s['prompt_tokens']:,} / 출력 "
            f"{s['completion_tokens']:,} / 합계 {s['total_tokens']:,} 토큰, 예상 비용 {cost_str}"
        )

    print("\n=== 노드별 호출 횟수 ===")
    for node, s in summarize_by_node().items():
        print(f"- {node}: 호출 {s['calls']}회, {s['total_tokens']:,} 토큰")

    print(f"\n전체 토큰: {grand_total_tokens:,}")
    if cost_unknown:
        print(
            "일부 모델의 가격이 설정되지 않아 총 비용은 표시하지 않습니다. "
            "usage.py의 PRICING_PER_1K를 실제 가격으로 채우면 비용까지 계산됩니다."
        )
    else:
        print(f"예상 총 비용: ${grand_total_cost:.4f}")


if __name__ == "__main__":
    print_report()
