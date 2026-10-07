# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## โปรเจกต์

งานเก็บคะแนน NLP ครั้งที่ 2 (โจทย์อยู่ใน `NLP-SubTest2.ipynb` ซึ่งเป็นไฟล์ที่ต้องส่งด้วย): แชตบอต RAG "น้องหอ" ตอบคำถามหอพักนักศึกษา มจพ. ปราจีนบุรี ด้วย Streamlit แล้ว deploy บน Streamlit Community Cloud จาก GitHub repo ส่วนตัว ข้อมูลใน `data/` เป็นข้อมูลจำลองทั้งหมด ต้องระบุว่าเป็นข้อมูลจำลองไว้เสมอ (ใน app, README และเอกสาร)

## คำสั่ง

เครื่องแล็บนี้ห้ามติดตั้งโปรแกรม จึงใช้ Python 3.11 แบบ embeddable ที่ `.python/` (ไม่ได้อยู่ใน PATH และถูก gitignore ไว้) ต้องเรียกผ่านพาธเต็ม:

```bash
.\.python\python.exe -m pip install -r requirements.txt
```

```bash
.\.python\python.exe -m streamlit run app.py
```

Python แบบ embeddable ใช้ `python311._pth` จึง**ไม่เพิ่มโฟลเดอร์ของสคริปต์เข้า `sys.path`** สคริปต์ทดสอบที่ `import rag` ต้อง `sys.path.insert(0, <project dir>)` เอง (Streamlit จัดการให้อยู่แล้วเมื่อรัน `app.py`)

API key อ่านจาก `st.secrets["GROQ_API_KEY"]` (ไฟล์ `.streamlit/secrets.toml` ในเครื่อง ถูก gitignore ไว้) และ fallback ไปที่ environment variable `GROQ_API_KEY` **ห้าม commit key เด็ดขาด** (ถูกหักคะแนน)

## สถาปัตยกรรม

- `rag.py`: pipeline ทั้งหมด แบ่งเป็นส่วนตามเกณฑ์ให้คะแนน (Loading/Cleaning → Chunking → Embedding & FAISS → Prompt → LLM)
  - Chunking ใช้หัวข้อ Markdown ในไฟล์ข้อมูล: บรรทัด `# ` คือชื่อเอกสาร และ `## ` คือหัวข้อย่อย ไฟล์ใหม่ใน `data/` ต้องใช้รูปแบบนี้ chunk จะถูกแบ่งตามหัวข้อย่อย หัวข้อที่ยาวเกิน `CHUNK_SIZE` จะแบ่งตามบรรทัด และบรรทัดที่ยาวเกินจะตัดตามขอบเขตคำด้วย pythainlp `newmm`
  - Embedding ใช้ `intfloat/multilingual-e5-small` ซึ่ง**ต้อง**มีคำนำหน้า `passage: ` / `query: ` เวกเตอร์ถูก normalize แล้วใช้ `IndexFlatIP` (เท่ากับ cosine)
  - ข้อความ "ไม่พบข้อมูล" (`NOT_FOUND`) ต้องคงไว้ตามโจทย์ การปฏิเสธการตอบมาจาก system prompt ไม่ได้ใช้ score threshold (คะแนนของ e5 กระจุกตัวอยู่ที่ประมาณ 0.8–0.9 แม้เป็นคำถามที่ไม่เกี่ยวข้อง)
- `app.py`: UI สร้าง `VectorStore` ครั้งเดียวด้วย `@st.cache_resource` ทุกเทิร์นจะเขียนคำถามต่อเนื่องให้สมบูรณ์ (`condense_question`) แล้วค้นหา จากนั้น stream คำตอบ แต่ละ assistant message ใน `st.session_state.messages` เก็บ `sources` ไว้เพื่อแสดงอ้างอิงซ้ำตอน rerun ส่วนเลข `[n]` ในคำตอบใช้ระบุว่า LLM อ้างอิงเอกสารใดจริง
- Groq ถอดโมเดลเก่าออกเป็นระยะ (เช่น `llama-3.3-70b-versatile` ใช้ไม่ได้แล้ว เมื่อเรียกจะได้ 404 `model_not_found`) แอปจึงดึงรายชื่อจาก `models.list()` แล้วเรียงตาม `PREFERRED_MODELS` ใน `app.py` ค่าเริ่มต้นคือ `openai/gpt-oss-120b` ซึ่งบางครั้งอ้างอิงเป็น `【n】` และ `stream_answer` จะแปลงวงเล็บให้เป็น `[n]` ตั้งแต่ตอน stream
- `.streamlit/config.toml` ตั้ง `fileWatcherType = "none"` เพื่อปิด traceback เรื่อง torchvision ที่เกิดจาก watcher สแกน `transformers` (แก้โค้ดแล้วต้องรีสตาร์ตแอปเอง ซึ่งรวมถึงแอปที่ deploy ด้วย: หลัง push Streamlit Cloud จะดึงโค้ดใหม่ ("Updated app!") แต่ยังรันโค้ดเก่า ต้องกด Manage app → ⋮ → Reboot app ทุกครั้ง)

## Deploy

- Repo: https://github.com/ThanawinPTINE/NLP-SubTest2 (public) · แอป: https://kmutnb-prachinburi-dorm-chatbot.streamlit.app (Python 3.11, key อยู่ใน Secrets ของ Streamlit)
- Git และ GitHub CLI แบบพกพาอยู่ใน `.tools/` ต้องตั้ง `$env:GH_CONFIG_DIR = ".tools\gh-config"` ก่อนใช้ `git push` ด้วย `.\.tools\git\cmd\git.exe` (credential helper ของ repo นี้ชี้ไปที่ gh)
- `submission/` (PDF ภาพหน้าจอและ `report.html` ที่ใช้สร้าง PDF) ถูก gitignore ไว้ PDF สร้างด้วย Edge headless `--print-to-pdf` เพื่อให้ภาษาไทยแสดงถูกต้อง

## ข้อกำหนดของงานที่ต้องรักษาไว้

- `data/` ต้องมีอย่างน้อย 10 ไฟล์ หรือรวม ≥ 15,000 ตัวอักษร (ปัจจุบันมี 12 ไฟล์ รวมประมาณ 24k) ข้อเท็จจริงทุกไฟล์ต้องสอดคล้องกัน (ไฟล์ภาษาอังกฤษ `12_...en.txt` ซ้ำตัวเลขจากไฟล์ภาษาไทย)
- `test_questions.csv` (UTF-8 with BOM เพื่อให้ Excel เปิดภาษาไทยได้) ต้องมี ≥ 10 ข้อ และมีข้อที่ไม่มีคำตอบในเอกสาร ≥ 2 ข้อ ห้ามเพิ่มข้อมูลใน `data/` ที่ทำให้คำถามประเภท "ไม่มีคำตอบ" (สระว่ายน้ำ ร้านส้มตำ รถรับส่งไปตลาด) มีคำตอบขึ้นมา
- ต้องแสดงเอกสารอ้างอิงทุกคำตอบ รวมถึงคำตอบ "ไม่พบข้อมูล" ด้วย
