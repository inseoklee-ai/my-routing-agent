"""평가셋 전체를 돌려 두 지표를 측정한다.

- 도구 호출 적절성: 실제로 호출한 근거(tool_used)가 기대 도구(expected_tool)와 정확히 일치하는가
- 답변 적절성: 반드시 담아야 할 사실을 다 담고, 말하면 안 되는 것을 어기지 않았는가 (LLM 채점, 표현이 아닌 사실 기준)

split=example 행은 few-shot 예시용이므로 채점에서 제외한다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score

import agent
import usage
from context import VALID_CATEGORIES

EVAL_SET_PATH = Path(__file__).parent / "data" / "eval_set.csv"
ANSWER_GOLD_PATH = Path(__file__).parent / "data" / "answer_gold.json"
RESULTS_PATH = Path(__file__).parent / "data" / "eval_results.csv"

# 여러 사실을 한 번에 대조하는 채점 작업은 gpt-4o-mini의 신뢰도가 낮아(같은 답변도
# 판정이 흔들림을 반복 실험으로 확인) 채점자만 더 상위 모델을 쓴다. 에이전트 자체
# (분류/답변/검증)는 비용 때문에 agent.MODEL_NAME(gpt-4o-mini)을 그대로 쓴다.
GRADER_MODEL = "gpt-4o"

TOOL_LABELS = VALID_CATEGORIES[:4] + ["넘기기"]  # 진단, PoC, PRD, MVP, 넘기기

GRADE_TOOL = {
    "type": "function",
    "function": {
        "name": "grade_answer",
        "description": (
            "답변이 반드시 담아야 할 사실을 모두 포함하고, 말하면 안 되는 내용을 "
            "어기지 않았는지 채점한다. 표현이 다른 것은 상관없고 사실 관계만 본다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "missing_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "반드시 담아야 하는데 빠진 사실. 없으면 빈 배열.",
                },
                "violated_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "말하면 안 되는데 답변에 포함된 내용. 없으면 빈 배열.",
                },
                "reason": {"type": "string"},
            },
            "required": ["passed", "missing_facts", "violated_facts", "reason"],
            "additionalProperties": False,
        },
    },
}

GRADE_SYSTEM_PROMPT = """당신은 고객응대 답변 채점자입니다.
표현이 다른 것은 전혀 문제가 아닙니다. 사실 관계만 보세요.
- '반드시 담아야 할 사실'이 답변에 실질적으로 포함되어 있는지 확인하세요 (표현이 달라도 같은 내용이면 포함된 것으로 봅니다).
- '말하면 안 되는 것'은 답변이 그 내용을 사실인 것처럼 단정하거나 구체적으로 제공했을 때만 위반입니다.
  그 내용을 부정하는 것(예: "계약서 서명은 필요 없습니다"는 '서명이 필요하다'의 위반이 아닙니다)이나,
  주제 이름만 언급하며 다루지 않는다고 안내하는 것(예: "법률 자문은 범위 밖입니다"는 실제 법률 자문을
  제공한 것이 아닙니다)은 위반이 아닙니다. 헷갈리면 "답변이 이 내용을 사실이라고 주장했는가?"로 판단하세요.
- 반드시 담아야 할 사실을 하나라도 빠뜨렸으면 반드시 실패로 판정하세요."""


def load_eval_rows() -> list[dict]:
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["split"] == "eval"]


def load_gold() -> dict:
    with ANSWER_GOLD_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def grade_answer(question: str, gold: dict, final_answer: str) -> dict:
    must_include = "\n".join(f"- {fact}" for fact in gold["must_include"])
    must_not_include = "\n".join(f"- {fact}" for fact in gold["must_not_include"])
    user_prompt = f"""[고객 문의]
{question}

[반드시 담아야 할 사실]
{must_include}

[말하면 안 되는 것]
{must_not_include}

[실제 답변]
{final_answer}

위 기준으로 채점하세요."""

    response = agent.get_client().chat.completions.create(
        model=GRADER_MODEL,
        max_tokens=agent.MAX_TOKENS,
        temperature=0,
        seed=42,
        tools=[GRADE_TOOL],
        tool_choice={"type": "function", "function": {"name": "grade_answer"}},
        messages=[
            {"role": "system", "content": GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage.log_usage(node="grade", model=GRADER_MODEL, usage=response.usage)
    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError(f"grade_answer 도구 호출 결과를 받지 못했습니다: {response}")
    return json.loads(message.tool_calls[0].function.arguments)


def validate_grader() -> None:
    """채점기 자체가 제대로 동작하는지, 모범 답안과 명백한 오답으로 먼저 확인한다."""
    gold = load_gold()
    print("=== 채점기 자체 검증 ===")

    # q02: 진단은 1~2회, 통화/대면 가능 — 모범 답안은 통과해야 한다
    q02_gold = gold["q02"]
    good_answer = "진단은 보통 1~2회 인터뷰로 진행되며, 통화나 대면 어느 쪽으로도 가능합니다."
    result = grade_answer("진단은 몇 번 만나야 하고 얼마나 걸리나요?", q02_gold, good_answer)
    print(f"[q02 모범 답안] passed={result['passed']} (기대: True) - {result['reason']}")

    # q12: 비용 질문에 금지된 구체 금액을 지어낸 오답은 반드시 실패해야 한다
    q12_gold = gold["q12"]
    bad_answer = "서비스 비용은 300만원부터 시작합니다."
    result = grade_answer("이 서비스 비용이 얼마인가요?", q12_gold, bad_answer)
    print(f"[q12 오답(금액 지어냄)] passed={result['passed']} (기대: False) - {result['reason']}")
    print()


def run_full_evaluation() -> list[dict]:
    rows = load_eval_rows()
    gold_map = load_gold()
    results = []

    for row in rows:
        state = agent.run_agent(row["question"])
        gold = gold_map[row["id"]]
        grading = grade_answer(row["question"], gold, state["final_answer"])

        results.append(
            {
                "id": row["id"],
                "category_gold": row["category"],
                "difficulty": row["difficulty"],
                "question": row["question"],
                "expected_tool": row["expected_tool"],
                "actual_tool": state["tool_used"],
                "tool_correct": state["tool_used"] == row["expected_tool"],
                "predicted_category": state["category"],
                "confidence": state["confidence"],
                "final_answer": state["final_answer"],
                "answer_passed": grading["passed"],
                "missing_facts": "; ".join(grading["missing_facts"]),
                "violated_facts": "; ".join(grading["violated_facts"]),
                "grading_reason": grading["reason"],
            }
        )
    return results


def compute_metrics(results: list[dict]) -> dict:
    y_true = [r["expected_tool"] for r in results]
    y_pred = [r["actual_tool"] for r in results]

    tool_accuracy = sum(r["tool_correct"] for r in results) / len(results)
    macro_f1 = f1_score(y_true, y_pred, labels=TOOL_LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=TOOL_LABELS)

    answer_accuracy = sum(r["answer_passed"] for r in results) / len(results)

    return {
        "tool_accuracy": tool_accuracy,
        "tool_macro_f1": macro_f1,
        "confusion_matrix": cm,
        "confusion_matrix_labels": TOOL_LABELS,
        "answer_accuracy": answer_accuracy,
    }


def save_results(results: list[dict]) -> None:
    fieldnames = list(results[0].keys())
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_report(results: list[dict], metrics: dict) -> None:
    print("=== 측정 결과 ===")
    print(f"도구 호출 적절성 (정확도): {metrics['tool_accuracy']:.2%}")
    print(f"도구 호출 적절성 (macro F1): {metrics['tool_macro_f1']:.3f}")
    print(f"답변 적절성 (정확도): {metrics['answer_accuracy']:.2%}")
    print()
    print("혼동 행렬 (행=정답, 열=예측):", metrics["confusion_matrix_labels"])
    for label, row in zip(metrics["confusion_matrix_labels"], metrics["confusion_matrix"]):
        print(f"  {label}: {list(row)}")
    print()

    failures = [r for r in results if not r["tool_correct"] or not r["answer_passed"]]
    print(f"=== 실패 사례 ({len(failures)}건) — 직접 읽고 원인 분류할 것 ===")
    for r in failures:
        print(f"- [{r['id']}] {r['question']}")
        print(
            f"  도구: 기대={r['expected_tool']} 실제={r['actual_tool']} "
            f"(일치={r['tool_correct']})"
        )
        print(f"  답변 통과: {r['answer_passed']}")
        if r["missing_facts"]:
            print(f"  누락된 사실: {r['missing_facts']}")
        if r["violated_facts"]:
            print(f"  위반한 금지사항: {r['violated_facts']}")
        print(f"  채점 사유: {r['grading_reason']}")
        print()


if __name__ == "__main__":
    validate_grader()
    results = run_full_evaluation()
    save_results(results)
    metrics = compute_metrics(results)
    print_report(results, metrics)
    print(f"전체 결과는 {RESULTS_PATH} 에 저장했습니다.")
