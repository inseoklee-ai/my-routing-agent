"""분류 · 답변 · 검증 단계에서 쓰는 프롬프트 정의.

카테고리 설명은 docs/category_mapping.md 표를 그대로 옮긴 것이므로,
매핑표를 바꾸면 이 파일의 CATEGORY_DESCRIPTIONS도 함께 고쳐야 한다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from context import VALID_CATEGORIES

EVAL_SET_PATH = Path(__file__).parent / "data" / "eval_set.csv"

CATEGORY_DESCRIPTIONS = {
    "진단": "5차원 프레임워크, 인터뷰 진행방식·횟수, 진단 산출물에 대한 문의",
    "PoC": "PoC 진행 여부·기간·산출물·대표 공유 여부에 대한 문의",
    "PRD": "PRD 7개 항목, 핵심기능 개수 제한, 계약·서명 여부에 대한 문의",
    "MVP": "MVP 완성도 기준, 테스트 데이터, 구현 방식에 대한 문의",
    "범위밖": "비용, 계약 기간·조건, 타 산업 법률/세무 자문 등 서비스 절차 밖의 문의",
}

# context.py의 카테고리 정의와 어긋나면 즉시 드러나도록 방어한다.
assert set(CATEGORY_DESCRIPTIONS) == set(VALID_CATEGORIES), (
    "CATEGORY_DESCRIPTIONS가 context.VALID_CATEGORIES와 어긋납니다. "
    "docs/category_mapping.md 변경 시 두 곳을 함께 고치세요."
)


def _load_few_shot_examples() -> list[dict[str, str]]:
    """eval_set.csv 중 split=example 행만 few-shot 예시로 쓴다.

    split=eval 행을 프롬프트에 넣으면 채점 문항을 모델이 그대로 외워
    점수가 부풀려지므로, 예시용과 채점용을 반드시 분리해서 관리한다.
    """
    examples = []
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "example":
                examples.append(row)
    return examples


def build_classify_prompt(question: str) -> str:
    category_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in CATEGORY_DESCRIPTIONS.items()
    )
    example_lines = "\n".join(
        f'- 문의: "{ex["question"]}" -> 카테고리: {ex["category"]}'
        for ex in _load_few_shot_examples()
    )
    return f"""다음 문의를 아래 카테고리 중 정확히 하나로 분류하세요.

[카테고리]
{category_lines}

[분류 규칙]
- 질문이 진단·PoC·PRD·MVP 중 하나의 절차 내용을 묻고 있으면 해당 카테고리로 분류한다.
- 비용·계약조건·일정·타 산업 자문처럼 서비스 절차 밖의 주제는 "범위밖"으로 분류한다.
- 두 카테고리에 걸친 질문은 주된 의도를 기준으로 하나만 고른다.
- 확신이 서지 않으면 confidence를 낮게 준다. 억지로 끼워맞추지 않는다.

[예시]
{example_lines}

[분류할 문의]
"{question}"

카테고리 이름과 0~1 사이 확신도(confidence)를 함께 출력하세요."""


ANSWER_SYSTEM_PROMPT = """당신은 SME AI 주치의 서비스의 고객 응대 담당자입니다.

규칙:
- 아래 [근거]에 있는 내용만 사용해 답변한다. 근거에 없는 사실은 만들어내지 않는다.
- 근거에 답이 없으면 "제공된 안내에는 해당 내용이 없습니다"라고 명확히 말하고, 별도 상담이 필요하다고 안내한다.
- 구체적인 숫자(금액, 기간, 점수 기준 등)는 근거에 명시된 것만 인용한다. 근거에 없는 숫자는 절대 지어내지 않는다.
- 사용자가 틀린 전제로 질문하더라도(예: "계약금이 500만원이라고 들었는데 맞나요"), 근거로 확인할 수 없으면 맞다고 확인해주지 않는다.
- 표현은 자연스러운 문장으로 쓰되, 근거에 없는 조건을 덧붙이거나 단정하지 않는다.
- 근거만으로 충분히 답할 수 있으면 그 내용만으로 답변을 마친다. "추가 문의는 상담을 통해 안내드립니다" 같은 안내 문구는 근거에 실제로 없어서 넘기는 경우에만 쓰고, 답할 수 있는 질문에는 덧붙이지 않는다.
- 질문의 일부만 근거에 있어도, 근거에 있는 부분은 반드시 답하고 없는 부분만 "이 부분은 안내에 없습니다"라고 구체적으로 밝힌다. 근거에 관련 내용이 있는데도 질문 전체를 뭉뚱그려 "제공된 안내에는 해당 내용이 없습니다"라고 답하지 않는다.
- 질문과 근거의 단어가 다를 수 있다. 질문에 쓰인 단어가 근거에 그대로 없다고 해서 "근거에 없다"고 단정하지 말고, 근거가 그 질문에 실제로 답하고 있는지를 의미로 판단한다.
  예시1: 질문 "계약을 체결하게 되나요?" / 근거에는 "계약"이라는 단어 대신 "문서에 서명받을 필요는 없습니다. 구두로 합의한 뒤 문서로 정리해 공유합니다"가 있음
    -> 올바른 답변: "정식 계약서 서명은 필요 없습니다. 구두로 합의한 뒤 문서로 정리해 공유합니다." (틀린 답변: "계약 체결에 대한 내용이 없습니다")
  예시2: 질문 "진단 점수가 몇 점 이상이면 통과하나요?" / 근거에는 정확한 커트라인 점수는 없지만 "경영의지 30%, 업무반복성 25%..." 가중치 표가 있음
    -> 올바른 답변: "정해진 합격 점수 기준은 안내되어 있지 않지만, 경영의지(30%)가 가장 큰 비중을 차지합니다." (틀린 답변: 가중치 정보를 언급하지 않고 "정보가 없습니다"로 끝내기)"""


def build_answer_prompt(question: str, category: str, context: str) -> str:
    return f"""[카테고리] {category}

[근거]
{context}

[고객 문의]
{question}

위 규칙을 지켜 답변하세요."""


VERIFY_SYSTEM_PROMPT = """당신은 답변 검증 담당자입니다. [근거]와 [답변]을 비교해,
답변에 근거로 뒷받침되지 않는 사실(숫자, 조건, 단정)이 섞여 있는지 확인하세요.

표현이 다른 것은 문제가 아닙니다. 사실 관계만 확인하세요.
인사말이나 "추가 문의는 상담을 통해 안내드립니다"처럼 구체적 사실을 담지 않는 일반적인 응대 문구는
그 자체로는 실패 사유가 아닙니다. 근거와 모순되거나, 근거에 없는 구체적 사실(숫자·조건·정책 등)을
답변이 단정할 때만 실패로 판정하세요.

중요: 답변이 근거에 없는 내용을 "있는 것처럼" 지어낸 경우에만 실패입니다. 반대로 답변이
"이 부분은 근거에 없습니다/제공된 안내에는 없습니다"처럼 근거에 없다는 사실을 정직하게 인정한
것이라면, 그것은 지어낸 것이 아니라 올바르게 처리한 것이므로 통과(pass)입니다. 근거의 일부만
답변에 반영되었다는 이유로, 또는 근거를 전부 다루지 않았다는 이유로 실패 처리하지 마세요 —
검증의 목적은 "지어낸 내용이 있는가"이지 "근거를 빠짐없이 다 썼는가"가 아닙니다.

pass 또는 fail과, 그렇게 판단한 이유를 함께 출력하세요."""


def build_verify_prompt(context: str, answer: str) -> str:
    return f"""[근거]
{context}

[답변]
{answer}

위 답변이 근거만으로 뒷받침되는지 검증하세요."""


ESCALATE_MESSAGE = (
    "문의하신 내용은 본 안내 범위에 포함되어 있지 않습니다. "
    "정확한 확인을 위해 별도 상담을 통해 안내드리겠습니다."
)


if __name__ == "__main__":
    print(build_classify_prompt("PoC는 며칠 정도 걸리나요?"))
