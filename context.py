"""근거 문서를 카테고리별로 쪼개고, 카테고리에 맞는 조각만 꺼내는 모듈.

docs/category_mapping.md 에서 정한 대응 관계를 그대로 코드로 옮긴 것이다.
"""

from __future__ import annotations

import re
from pathlib import Path

MANUAL_PATH = Path(__file__).parent / "docs" / "sme_ai_doctor_service_manual.md"

# docs/category_mapping.md 의 매핑표와 반드시 일치시킬 것
CATEGORY_TO_SECTION = {
    "진단": "진단",
    "PoC": "PoC",
    "PRD": "PRD",
    "MVP": "MVP",
    "범위밖": "안내 범위",
}

VALID_CATEGORIES = list(CATEGORY_TO_SECTION.keys())

_SECTION_HEADER_RE = re.compile(r"^##\s+(.+)$")


def _split_into_sections(markdown_text: str) -> dict[str, str]:
    """`## 제목` 헤더를 기준으로 마크다운을 섹션별 본문으로 나눈다."""
    sections: dict[str, str] = {}
    current_title: str | None = None
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        header_match = _SECTION_HEADER_RE.match(line)
        if header_match:
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = header_match.group(1).strip()
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()

    return sections


def load_sections(manual_path: Path = MANUAL_PATH) -> dict[str, str]:
    """매뉴얼 파일을 읽어 {섹션 제목: 본문} 딕셔너리로 반환한다."""
    text = manual_path.read_text(encoding="utf-8")
    return _split_into_sections(text)


def get_context(category: str, sections: dict[str, str] | None = None) -> str:
    """카테고리에 매핑된 문서 섹션 본문만 반환한다.

    '범위밖'은 실질 근거가 아니라 '다루지 않는다'는 사실 자체가 근거이므로,
    안내 범위 섹션(고지 문구)을 그대로 반환한다.
    """
    if category not in CATEGORY_TO_SECTION:
        raise ValueError(
            f"알 수 없는 카테고리: {category!r} (허용된 값: {VALID_CATEGORIES})"
        )

    sections = sections if sections is not None else load_sections()
    section_title = CATEGORY_TO_SECTION[category]
    return sections.get(section_title, "")


if __name__ == "__main__":
    loaded = load_sections()
    print(f"발견된 섹션: {list(loaded.keys())}\n")
    for category in VALID_CATEGORIES:
        preview = get_context(category, loaded)[:60].replace("\n", " ")
        print(f"[{category}] -> {preview}...")
