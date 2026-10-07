"""RAG pipeline: โหลดเอกสาร → ทำความสะอาด → แบ่ง chunk → embedding → ค้นหาด้วย FAISS → สร้าง prompt → เรียก LLM (Groq)"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize as thai_normalize
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"

# multilingual-e5-small: โมเดลขนาดเล็ก (~470 MB) รองรับภาษาไทย เหมาะกับหน่วยความจำของ Streamlit Community Cloud
EMBED_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 500     # จำนวนตัวอักษรสูงสุดต่อ chunk
CHUNK_OVERLAP = 1    # จำนวนบรรทัดที่ซ้อนทับระหว่าง chunk ที่อยู่ติดกัน
NOT_FOUND = "ไม่พบข้อมูล"


@dataclass
class Chunk:
    text: str      # เนื้อหาของ chunk
    source: str    # ชื่อไฟล์ต้นทาง
    title: str     # ชื่อเอกสาร (หัวข้อ # )
    section: str   # ชื่อหัวข้อย่อย (หัวข้อ ## )

    @property
    def header(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title


# ---------------------------------------------------------------------------
# 1. Document Loading & Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """ทำความสะอาดข้อความ: รวมรูปแบบ Unicode, ลบอักขระล่องหน, จัดลำดับสระ/วรรณยุกต์ไทย, ลบช่องว่างซ้ำ"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[​-‍﻿]", "", text)
    lines = []
    for line in text.splitlines():
        line = thai_normalize(line)               # แก้สระ/วรรณยุกต์ซ้ำหรือผิดลำดับ
        line = re.sub(r"\*\*|__", "", line)       # ลบเครื่องหมายตัวหนาของ Markdown
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    return "\n".join(lines)


def load_documents(data_dir: Path = DATA_DIR) -> list[tuple[str, str]]:
    """อ่านไฟล์ .txt และ .md ทั้งหมดในโฟลเดอร์ data/ คืนค่าเป็น (ชื่อไฟล์, ข้อความที่ทำความสะอาดแล้ว)"""
    docs = []
    for path in sorted(data_dir.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, clean_text(path.read_text(encoding="utf-8"))))
    return docs


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def split_sections(text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """แยกเอกสารตามหัวข้อ Markdown: '# ' คือชื่อเอกสาร, '## ' คือหัวข้อย่อย"""
    title, section, lines = "", "", []
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            if lines:
                sections.append((section, lines))
            section, lines = line[3:].strip(), []
        else:
            lines.append(line)
    if lines:
        sections.append((section, lines))
    return title, sections


def split_long_line(line: str, size: int) -> list[str]:
    """ตัดบรรทัดที่ยาวเกิน size ตามขอบเขตคำด้วย PyThaiNLP (ภาษาไทยไม่เว้นวรรคระหว่างคำ จึงตัดตามตัวอักษรตรง ๆ ไม่ได้)"""
    if len(line) <= size:
        return [line]
    pieces, current = [], ""
    for word in word_tokenize(line, engine="newmm", keep_whitespace=True):
        if current and len(current) + len(word) > size:
            pieces.append(current.strip())
            current = ""
        current += word
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_documents(docs: list[tuple[str, str]], size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """แบ่งเอกสารเป็น chunk ตามหัวข้อย่อย หัวข้อที่ยาวเกิน size จะถูกแบ่งต่อตามบรรทัด โดยซ้อนทับกัน overlap บรรทัด"""
    chunks = []
    for source, text in docs:
        title, sections = split_sections(text)
        for section, lines in sections:
            units = [piece for line in lines for piece in split_long_line(line, size)]
            current: list[str] = []
            for unit in units:
                if current and len("\n".join(current + [unit])) > size:
                    chunks.append(Chunk("\n".join(current), source, title, section))
                    current = current[-overlap:] if overlap else []
                current.append(unit)
            if current:
                chunks.append(Chunk("\n".join(current), source, title, section))
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding & Vector Search (FAISS)
# ---------------------------------------------------------------------------

class VectorStore:
    """เก็บ embedding ของทุก chunk ใน FAISS index และค้นหาด้วย cosine similarity"""

    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device="cpu")
        # e5 ต้องใส่คำนำหน้า "passage: " ให้เอกสาร และ "query: " ให้คำถาม
        # ใส่ชื่อเอกสารและหัวข้อไว้ด้วยเพื่อให้ chunk มีบริบทครบ
        passages = [f"passage: {c.header}\n{c.text}" for c in chunks]
        embeddings = self.model.encode(passages, normalize_embeddings=True,
                                       batch_size=32, convert_to_numpy=True)
        # เวกเตอร์ถูก normalize แล้ว Inner Product จึงเท่ากับ cosine similarity
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype="float32"))

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        vector = self.model.encode([f"query: {query}"], normalize_embeddings=True,
                                   convert_to_numpy=True)
        scores, ids = self.index.search(np.asarray(vector, dtype="float32"),
                                        min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


def build_vector_store(data_dir: Path = DATA_DIR) -> VectorStore:
    return VectorStore(chunk_documents(load_documents(data_dir)))


# ---------------------------------------------------------------------------
# 4. Prompt Engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""คุณคือ "น้องหอ" ผู้ช่วยตอบคำถามเกี่ยวกับหอพักนักศึกษา มจพ. ปราจีนบุรี

กฎที่ต้องปฏิบัติอย่างเคร่งครัด:
1. ตอบโดยใช้ข้อมูลจาก "เอกสารอ้างอิง" ที่ให้มาในข้อความล่าสุดเท่านั้น ห้ามใช้ความรู้ภายนอก ห้ามเดา และห้ามแต่งตัวเลข เวลา หรือเบอร์โทรศัพท์ขึ้นเอง
2. ทุกข้อเท็จจริงในคำตอบต้องอ้างอิงหมายเลขเอกสารในวงเล็บเหลี่ยมท้ายประโยค เช่น [1] หรือ [2][3] (ใช้วงเล็บ [ ] เท่านั้น ห้ามใช้【 】)
3. ถ้าเอกสารอ้างอิงไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบขึ้นต้นด้วยคำว่า "{NOT_FOUND}" แล้วบอกสั้น ๆ ว่าเอกสารของหอพักไม่ได้ระบุเรื่องนี้ และแนะนำให้สอบถามสำนักงานหอพัก ห้ามใส่หมายเลขอ้างอิงในกรณีนี้
4. ถ้าเอกสารตอบได้เพียงบางส่วน ให้ตอบเฉพาะส่วนที่มีข้อมูลพร้อมอ้างอิง และบอกว่าส่วนที่เหลือ{NOT_FOUND}
5. ตอบเป็นภาษาเดียวกับคำถาม (ภาษาไทยหรือภาษาอังกฤษ) เฉพาะเมื่อคำถามเป็นภาษาอังกฤษและไม่มีข้อมูล ให้ตอบว่า "{NOT_FOUND} (No information found in the dormitory documents.)" ถ้าคำถามเป็นภาษาไทยให้ตอบเป็นภาษาไทยล้วน
6. ตอบให้กระชับ ตรงประเด็น ใช้รายการแบบ bullet เมื่อมีหลายข้อ และคงตัวเลขให้ตรงตามเอกสาร"""

CONDENSE_PROMPT = """จากบทสนทนาด้านล่าง ให้เขียนคำถามล่าสุดของผู้ใช้ใหม่เป็นคำถามที่สมบูรณ์ในตัวเอง เข้าใจได้โดยไม่ต้องอ่านบทสนทนาก่อนหน้า
ใช้ภาษาเดียวกับคำถามล่าสุด และตอบกลับเฉพาะคำถามที่เขียนใหม่เท่านั้น ไม่ต้องตอบคำถาม

บทสนทนา:
{history}

คำถามล่าสุด: {question}
คำถามที่เขียนใหม่:"""


def format_context(results: list[tuple[Chunk, float]]) -> str:
    """จัดรูปแบบ chunk ที่ค้นได้ให้มีหมายเลขและแหล่งที่มา เพื่อให้ LLM อ้างอิงได้"""
    blocks = [f"[{i}] (ที่มา: {chunk.source} | หัวข้อ: {chunk.header})\n{chunk.text}"
              for i, (chunk, _) in enumerate(results, start=1)]
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[tuple[Chunk, float]],
                   history: list[dict]) -> list[dict]:
    """ประกอบข้อความที่ส่งให้ LLM: system prompt + ประวัติแชตล่าสุด + เอกสารอ้างอิงและคำถาม"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({
        "role": "user",
        "content": f"เอกสารอ้างอิง:\n{format_context(results)}\n\nคำถาม: {question}",
    })
    return messages


# ---------------------------------------------------------------------------
# 5. Large Language Model (Groq API)
# ---------------------------------------------------------------------------

def condense_question(client, model: str, history: list[dict], question: str) -> str:
    """เขียนคำถามต่อเนื่อง (เช่น "แล้วห้องแอร์ล่ะ") ให้เป็นคำถามสมบูรณ์ เพื่อให้ค้นเอกสารได้ถูกต้อง"""
    if not history:
        return question
    transcript = "\n".join(f"{'ผู้ใช้' if m['role'] == 'user' else 'ผู้ช่วย'}: {m['content']}"
                           for m in history)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": CONDENSE_PROMPT.format(history=transcript, question=question)}],
        temperature=0,
        max_tokens=1024,
    )
    rewritten = (response.choices[0].message.content or "").strip()
    return rewritten or question


def stream_answer(client, model: str, question: str, results: list[tuple[Chunk, float]],
                  history: list[dict]):
    """เรียก LLM แบบ streaming แล้วส่งข้อความกลับทีละส่วน"""
    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(question, results, history),
        temperature=0.1,
        max_tokens=2048,  # โมเดล reasoning (เช่น gpt-oss) นับ token ที่ใช้คิดรวมในส่วนนี้ด้วย
        stream=True,
    )
    for part in stream:
        delta = part.choices[0].delta.content if part.choices else None
        if delta:
            # บางโมเดล (เช่น gpt-oss) อ้างอิงเป็น 【1】 แปลงเป็น [1] ทีละตัวอักษร เพราะวงเล็บกับตัวเลขอาจมาคนละ chunk
            yield delta.replace("【", "[").replace("】", "]")
