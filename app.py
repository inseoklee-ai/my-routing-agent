"""사용자가 직접 대화해볼 수 있는 데모 화면.

streamlit run app.py 로 실행한다.
"""

import streamlit as st

import agent

st.set_page_config(page_title="SME AI 주치의 문의 응대", page_icon="🤖")

st.title("SME AI 주치의 - 고객 문의 응대 데모")
st.caption(
    "진단 · PoC · PRD · MVP 절차에 대해 물어보세요. "
    "안내 범위 밖의 문의(비용·계약·법률 자문 등)는 넘기기로 안내합니다."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("문의를 입력하세요", placeholder="예: PoC는 며칠 정도 걸리나요?")
    submitted = st.form_submit_button("문의하기")

if submitted and question.strip():
    with st.spinner("답변을 준비하는 중입니다..."):
        result = agent.run_agent(question)
    st.session_state.history.insert(0, {"question": question, "result": result})

if not st.session_state.history:
    st.info("위 입력창에 문의를 적고 [문의하기]를 눌러보세요.")

for item in st.session_state.history:
    q = item["question"]
    result = item["result"]

    st.markdown(f"#### Q. {q}")
    st.markdown(result["final_answer"])

    with st.expander("어떻게 답했는지 보기 (호출한 도구 · 근거 · 검증 결과)"):
        st.write(f"**분류된 카테고리**: {result['category']} (확신도 {result['confidence']:.2f})")
        st.write(f"**호출한 도구(조회한 근거 섹션)**: `{result['tool_used']}`")

        if result.get("retrieved_context"):
            st.write("**근거로 쓴 문서 부분**")
            st.code(result["retrieved_context"], language="markdown")

        if "verification_passed" in result:
            passed = result["verification_passed"]
            st.write(f"**검증 결과**: {'✅ 통과' if passed else '⚠️ 실패 → 넘김으로 전환'}")
            st.caption(f"검증 사유: {result.get('verification_reason', '-')}")
        else:
            st.write("**검증 결과**: 분류 단계에서 범위 밖 또는 확신도 부족으로 판단되어, 근거 조회·검증 없이 바로 넘겨졌습니다.")

    st.divider()
