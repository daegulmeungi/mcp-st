import os
import json
from json import JSONDecodeError

import httpx
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# Gemini SDK
from google import genai
from google.genai import types

# ---------- 환경 변수에서 Gemini API 키 읽기 ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다. "
        "Google AI Studio에서 키를 발급받고 환경 변수에 넣어주세요."
    )

# Gemini 클라이언트 생성
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- MCP 서버 인스턴스 ----------
mcp = FastMCP("PDF Summarizer & Quiz (Gemini)")


# ===================== 공용 모델 / 헬퍼 =====================

class PdfSummary(BaseModel):
    """PDF 요약 결과 구조"""

    summary: str = Field(
        description="PDF 전체 내용을 한국어로 요약한 글"
    )


class QuizItem(BaseModel):
    """한 문제에 대한 구조"""

    question: str = Field(description="한국어 객관식 문제 질문")
    choices: list[str] = Field(
        description="보기 리스트. 보통 4개 보기 사용"
    )
    correct_index: int = Field(
        description="정답 보기의 인덱스 (0부터 시작)"
    )
    explanation: str | None = Field(
        default=None,
        description="정답과 관련한 간단한 해설 (선택 사항)"
    )


class PdfSummaryWithQuiz(BaseModel):
    """요약 + 퀴즈 묶음 구조"""

    summary: str = Field(description="PDF 전체 요약")
    quiz: list[QuizItem] = Field(
        description="객관식 퀴즈 목록"
    )


def _download_pdf(pdf_url: str) -> bytes:
    """URL에서 PDF 바이너리를 받아오는 공용 함수"""
    pdf_url = pdf_url.strip()  # 앞뒤 공백/줄바꿈 제거

    try:
        resp = httpx.get(pdf_url, timeout=60.0)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        raise RuntimeError(f"PDF 다운로드 실패 (url={pdf_url!r}): {e}") from e


# ===================== 1) 기존: 요약만 하는 툴 =====================

@mcp.tool()
def summarize_pdf_from_url(
    pdf_url: str,
    prompt: str = "이 PDF를 한국어로 간단히 요약해줘. 5문장 이내로."
) -> PdfSummary:
    """
    URL로 접근 가능한 PDF를 가져와 Gemini로 요약합니다.
    - pdf_url : http(s)로 바로 열 수 있는 PDF 주소
    - prompt  : 요약 방식에 대한 추가 지시
    """
    pdf_bytes = _download_pdf(pdf_url)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                prompt,
            ],
        )
        summary_text = response.text

    except Exception as e:
        raise RuntimeError(f"Gemini 요약 호출 실패: {e}") from e

    return PdfSummary(summary=summary_text)


# ===================== 2) 확장: 요약 + 퀴즈 생성 툴 =====================

@mcp.tool()
def summarize_and_quiz_pdf_from_url(
    pdf_url: str,
    num_questions: int = 5,
) -> PdfSummaryWithQuiz:
    """
    URL로 접근 가능한 PDF를 읽고,
    - 한국어 요약
    - 객관식 퀴즈 num_questions개
    를 JSON 구조로 반환합니다.

    반환 구조:
    {
      "summary": "...",
      "quiz": [
        {
          "question": "...",
          "choices": ["...", "...", "...", "..."],
          "correct_index": 0,
          "explanation": "..."
        },
        ...
      ]
    }
    """
    pdf_bytes = _download_pdf(pdf_url)

    # Gemini에게 JSON 형식으로 요약 + 퀴즈 생성 요청
    system_prompt = f"""
당신은 대학 강의를 위한 학습 자료를 만드는 보조 교사입니다.

주어진 PDF 내용을 바탕으로 아래 JSON 형식에 맞게 출력하세요.

{{
  "summary": "PDF 전체 내용을 한국어로 5~7문장으로 요약한 텍스트",
  "quiz": [
    {{
      "question": "객관식 문제의 질문 (한국어)",
      "choices": ["선택지1", "선택지2", "선택지3", "선택지4"],
      "correct_index": 0,
      "explanation": "정답과 관련한 간단한 해설 (한국어)"
    }}
  ]
}}

요구사항:
- quiz 배열에는 정확히 {num_questions}개의 문제만 포함하세요.
- 각 choices는 4개의 보기를 가지게 하세요.
- correct_index는 0부터 3 사이의 정수입니다.
- summary와 quiz의 내용은 모두 PDF 내용에 기반해야 합니다.
- 반드시 위 JSON 형식 그대로를 출력하고,
  다른 설명, 자연어 문장, 마크다운, 코드블록, 주석은 절대 추가하지 마세요.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                system_prompt,
            ],
        )

        raw_text = response.text.strip()

        # 혹시 ```json ... ``` 이런 식으로 감싸졌다면 제거 시도
        if raw_text.startswith("```"):
            # ```json 또는 ```로 시작하는 경우를 간단히 처리
            raw_text = raw_text.strip("`")
            # 'json' 같은 언어 표시 제거
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

    except JSONDecodeError as e:
        # 디버깅용으로 앞부분만 보여주기
        snippet = raw_text[:300]
        raise RuntimeError(
            f"Gemini가 올바른 JSON을 반환하지 않았습니다: {e}\n"
            f"응답 앞부분: {snippet!r}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Gemini 요약+퀴즈 호출 실패: {e}") from e

    # Pydantic으로 구조 검증
    try:
        result = PdfSummaryWithQuiz.parse_obj(data)
    except Exception as e:
        raise RuntimeError(
            f"JSON을 PdfSummaryWithQuiz 구조로 변환하는 데 실패했습니다: {e}\n"
            f"data={data!r}"
        ) from e

    return result


# ===================== 메인 실행 (선택) =====================

if __name__ == "__main__":
    # python server.py 로 직접 실행하고 싶을 때 사용
    mcp.run()
