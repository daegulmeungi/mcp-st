import os
import json
from json import JSONDecodeError
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# ---------- 1. Gemini 클라이언트 설정 ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다. "
        "Google AI Studio에서 키를 발급받고 환경 변수에 넣어주세요."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------- 2. Pydantic 모델 (요약 + 퀴즈 구조) ----------

class QuizItem(BaseModel):
    """한 문제에 대한 구조"""
    question: str = Field(description="한국어 객관식 문제 질문")
    choices: List[str] = Field(description="보기 리스트. 보통 4개 보기 사용")
    correct_index: int = Field(description="정답 보기의 인덱스 (0부터 시작)")
    explanation: Optional[str] = Field(
        default=None,
        description="정답과 관련한 간단한 해설 (선택 사항)"
    )


class PdfSummaryWithQuiz(BaseModel):
    """요약 + 퀴즈 묶음 구조"""
    summary: str = Field(description="PDF 전체 요약")
    quiz: List[QuizItem] = Field(description="객관식 퀴즈 목록")


# ---------- 3. PDF 바이트를 받아서 요약+퀴즈 생성하는 함수 ----------

def summarize_and_quiz_pdf_bytes(pdf_bytes: bytes, num_questions: int) -> PdfSummaryWithQuiz:
    """
    PDF 바이너리와 문제 개수를 받아
    - 요약 + 객관식 퀴즈를 생성하여 PdfSummaryWithQuiz로 반환
    """
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
            model="gemini-2.5-flash",  # 필요시 다른 모델로 변경 가능
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
                system_prompt,
            ],
        )
        raw_text = response.text.strip()

        # 혹시 ```json ... ``` 형태로 감싸져 있으면 제거 시도
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

    except JSONDecodeError as e:
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


# ---------- 4. FastAPI 앱 정의 ----------

app = FastAPI(title="PDF 요약 + 퀴즈 생성 웹 (Gemini)")


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    """PDF 업로드 폼 페이지"""
    return """
    <html>
      <head>
        <meta charset="utf-8" />
        <title>PDF 요약 + 퀴즈 생성기</title>
      </head>
      <body>
        <h1>PDF 요약 + 퀴즈 생성기 (Gemini)</h1>
        <form action="/summarize-quiz" method="post" enctype="multipart/form-data">
          <p>
            PDF 파일 선택:
            <input type="file" name="pdf" accept="application/pdf" required />
          </p>
          <p>
            생성할 문제 개수:
            <input type="number" name="num_questions" value="5" min="1" max="20" />
          </p>
          <button type="submit">요약 & 퀴즈 생성</button>
        </form>
      </body>
    </html>
    """


@app.post("/summarize-quiz", response_class=HTMLResponse)
async def summarize_quiz(
    pdf: UploadFile = File(...),
    num_questions: int = Form(5),
):
    """업로드된 PDF를 요약 + 퀴즈 생성하여 HTML로 보여주는 엔드포인트"""

    try:
        pdf_bytes = await pdf.read()
        result = summarize_and_quiz_pdf_bytes(pdf_bytes, num_questions)
    except Exception as e:
        # 에러 발생 시 간단한 HTML로 에러 메시지 출력
        return f"""
        <html>
          <head><meta charset="utf-8" /><title>에러</title></head>
          <body>
            <h1>에러 발생</h1>
            <pre>{str(e)}</pre>
            <p><a href="/">돌아가기</a></p>
          </body>
        </html>
        """

    # 결과를 예쁘게 HTML로 렌더링
    html_parts = [
        "<html>",
        "<head><meta charset='utf-8' /><title>요약 & 퀴즈 결과</title></head>",
        "<body>",
        "<h1>요약 결과</h1>",
        f"<pre style='white-space: pre-wrap;'>{result.summary}</pre>",
        "<hr>",
        "<h1>퀴즈</h1>",
    ]

    for i, q in enumerate(result.quiz, start=1):
        html_parts.append(f"<h2>문제 {i}</h2>")
        html_parts.append(f"<p>{q.question}</p>")
        html_parts.append("<ol type='A'>")
        for idx, choice in enumerate(q.choices):
            # 여기서는 정답을 바로 표시해두었는데,
            # 학생용 페이지라면 표시 안 하고 teacher만 보게 할 수도 있음
            mark = " ✅ (정답)" if idx == q.correct_index else ""
            html_parts.append(f"<li>{choice}{mark}</li>")
        html_parts.append("</ol>")
        if q.explanation:
            html_parts.append(
                f"<p><b>해설:</b> {q.explanation}</p>"
            )
        html_parts.append("<hr>")

    html_parts.append("<p><a href='/'>다른 PDF로 다시 시도하기</a></p>")
    html_parts.append("</body></html>")

    return "".join(html_parts)
