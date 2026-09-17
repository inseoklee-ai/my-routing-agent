"""분류 -> (넘기기 판정) -> 근거조회 -> 답변 -> 검증 -> (넘기기 판정) 파이프라인. LangGraph로 구현.

OpenAI API를 사용한다. MODEL_NAME만 바꾸면 더 성능 좋은 모델로 쉽게 교체할 수 있다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

import context
import prompts
import usage

# 저장소 루트의 .env에서 OPENAI_API_KEY를 읽는다 (.env는 .gitignore에 등록되어
# git에는 올라가지 않는다 — 누구든 이 저장소를 받으면 .env.example을 .env로 복사해
# 자기 키를 넣으면 된다).
load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = "gpt-4o-mini"
# 검증(verify) 단계는 "정직하게 모른다고 말한 것"과 "지어낸 것"을 구분해야 하는데,
# gpt-4o-mini는 반복 실험에서 이 판단을 간헐적으로 뒤집었다. 분류는 mini로도 100%
# 정확했으므로 그대로 두고, 검증과 답변 생성만 더 신뢰할 수 있는 모델로 올린다.
# (답변 생성: 질문과 다른 표현으로 쓰인 근거를 mini가 못 찾아내는 패러프레이즈 실패가 있었음)
VERIFY_MODEL = "gpt-4o"
ANSWER_MODEL = "gpt-4o-mini"
CONFIDENCE_THRESHOLD = 0.6
MAX_TOKENS = 1024

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """OpenAI 클라이언트는 실제로 호출할 때 처음 생성한다.

    생성자가 API 키를 즉시 요구하므로, 모듈 임포트나 그래프 구성처럼
    키 없이도 가능한 작업이 키 부재로 실패하지 않도록 지연 생성한다.
    """
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_inquiry",
        "description": "고객 문의를 정해진 카테고리 중 하나로 분류하고 확신도를 매긴다.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": context.VALID_CATEGORIES},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "이 분류가 맞다고 확신하는 정도 (0~1)",
                },
            },
            "required": ["category", "confidence"],
            "additionalProperties": False,
        },
    },
}

VERIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "report_verification",
        "description": "답변이 근거 문서만으로 뒷받침되는지 검증 결과를 보고한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["passed", "reason"],
            "additionalProperties": False,
        },
    },
}


class AgentState(TypedDict, total=False):
    question: str
    category: str
    confidence: float
    tool_used: str
    retrieved_context: str
    draft_answer: str
    verification_passed: bool
    verification_reason: str
    final_answer: str


def _call_tool(
    system_prompt: str, user_prompt: str, tool: dict, node: str, model: str = MODEL_NAME
) -> dict:
    """도구를 강제 호출시켜 구조화된 결과만 받는다."""
    function_name = tool["function"]["name"]
    response = get_client().chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=0,
        seed=42,
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": function_name}},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage.log_usage(node=node, model=model, usage=response.usage)
    message = response.choices[0].message
    if not message.tool_calls:
        raise RuntimeError(f"{function_name} 도구 호출 결과를 받지 못했습니다: {response}")
    return json.loads(message.tool_calls[0].function.arguments)


def classify_node(state: AgentState) -> dict:
    result = _call_tool(
        system_prompt="당신은 고객 문의를 정확히 분류하는 담당자입니다.",
        user_prompt=prompts.build_classify_prompt(state["question"]),
        tool=CLASSIFY_TOOL,
        node="classify",
    )
    return {"category": result["category"], "confidence": result["confidence"]}


def route_after_classify(state: AgentState) -> str:
    if state["category"] == "범위밖" or state["confidence"] < CONFIDENCE_THRESHOLD:
        return "escalate"
    return "retrieve"


def retrieve_node(state: AgentState) -> dict:
    retrieved = context.get_context(state["category"])
    return {"retrieved_context": retrieved, "tool_used": state["category"]}


def answer_node(state: AgentState) -> dict:
    response = get_client().chat.completions.create(
        model=ANSWER_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        seed=42,
        messages=[
            {"role": "system", "content": prompts.ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompts.build_answer_prompt(
                    question=state["question"],
                    category=state["category"],
                    context=state["retrieved_context"],
                ),
            },
        ],
    )
    usage.log_usage(node="answer", model=ANSWER_MODEL, usage=response.usage)
    draft = response.choices[0].message.content
    return {"draft_answer": draft}


def verify_node(state: AgentState) -> dict:
    result = _call_tool(
        system_prompt=prompts.VERIFY_SYSTEM_PROMPT,
        user_prompt=prompts.build_verify_prompt(
            context=state["retrieved_context"], answer=state["draft_answer"]
        ),
        tool=VERIFY_TOOL,
        node="verify",
        model=VERIFY_MODEL,
    )
    return {
        "verification_passed": result["passed"],
        "verification_reason": result["reason"],
    }


def route_after_verify(state: AgentState) -> str:
    return "end" if state["verification_passed"] else "escalate"


def escalate_node(state: AgentState) -> dict:
    scope_note = context.get_context("범위밖")
    return {
        "final_answer": f"{prompts.ESCALATE_MESSAGE}\n\n{scope_note}",
        "tool_used": "넘기기",
    }


def finalize_node(state: AgentState) -> dict:
    return {"final_answer": state["draft_answer"]}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("answer", answer_node)
    builder.add_node("verify", verify_node)
    builder.add_node("escalate", escalate_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify", route_after_classify, {"retrieve": "retrieve", "escalate": "escalate"}
    )
    builder.add_edge("retrieve", "answer")
    builder.add_edge("answer", "verify")
    builder.add_conditional_edges(
        "verify", route_after_verify, {"end": "finalize", "escalate": "escalate"}
    )
    builder.add_edge("finalize", END)
    builder.add_edge("escalate", END)

    return builder.compile()


graph = build_graph()


def run_agent(question: str) -> AgentState:
    return graph.invoke({"question": question})


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    else:
        result = run_agent("PoC는 며칠 정도 걸리나요?")
        print(result)
