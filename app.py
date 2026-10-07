"""น้องหอ: แชตบอต RAG ตอบคำถามหอพักนักศึกษา มจพ. ปราจีนบุรี (Streamlit + FAISS + Groq)"""

import os
import re

import streamlit as st
from groq import Groq

import rag

st.set_page_config(page_title="น้องหอ | หอพัก มจพ. ปราจีนบุรี", page_icon="🏠", layout="centered")

# โมเดลที่แนะนำ เรียงตามลำดับความชอบ (Groq ถอดโมเดลเก่าออกเป็นระยะ จึงตรวจกับรายชื่อจริงจาก API ทุกชั่วโมง)
PREFERRED_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
NON_CHAT_KEYWORDS = ("whisper", "guard", "orpheus", "tts")
HISTORY_TURNS = 6  # จำนวนข้อความล่าสุดที่ส่งให้ LLM เป็นบริบทของบทสนทนา
SAMPLE_QUESTIONS = [
    "ประตูหอพักปิดกี่โมง",
    "ค่าไฟหน่วยละเท่าไร",
    "แอร์เสียต้องแจ้งซ่อมยังไง",
    "ทำคีย์การ์ดหายต้องทำอย่างไร",
    "How much is the laundry?",
    "หอพักมีสระว่ายน้ำไหม",
]


@st.cache_resource(show_spinner="กำลังโหลดโมเดล embedding และสร้าง FAISS index (ครั้งแรกใช้เวลาประมาณ 1 นาที)...")
def load_vector_store() -> rag.VectorStore:
    # cache_resource ทำให้โหลดโมเดลและสร้าง index เพียงครั้งเดียว ใช้ร่วมกันทุกผู้ใช้
    return rag.build_vector_store()


def get_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:  # ไม่มีไฟล์ secrets.toml หรือไม่มี key นี้
        return os.environ.get("GROQ_API_KEY")


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(api_key: str) -> list[str]:
    """รายชื่อโมเดลแชตที่ key นี้ใช้ได้ โดยให้โมเดลที่แนะนำขึ้นก่อน"""
    try:
        ids = {m.id for m in Groq(api_key=api_key).models.list().data}
    except Exception:
        return PREFERRED_MODELS
    preferred = [m for m in PREFERRED_MODELS if m in ids]
    others = sorted(m for m in ids - set(preferred)
                    if not any(word in m for word in NON_CHAT_KEYWORDS))
    return preferred + others or PREFERRED_MODELS


def default_model() -> str | None:
    try:
        return st.secrets.get("GROQ_MODEL")
    except Exception:
        return None


def render_sources(message: dict) -> None:
    """แสดงเอกสารอ้างอิงใต้คำตอบทุกครั้ง พร้อมทำเครื่องหมายเอกสารที่ LLM อ้างอิงจริง"""
    sources = message.get("sources", [])
    if not sources:
        return
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", message["content"])}
    cited_files = sorted({sources[i - 1]["source"] for i in cited if 0 < i <= len(sources)})
    if cited_files:
        st.caption("📄 อ้างอิงจาก: " + ", ".join(cited_files))
    else:
        st.caption("📄 ไม่มีเอกสารที่ตอบคำถามนี้ได้ (ด้านล่างคือเอกสารที่ใกล้เคียงที่สุด)")

    with st.expander(f"📚 เอกสารที่ค้นพบ {len(sources)} รายการ"):
        if message.get("query") and message["query"] != message.get("question"):
            st.caption(f"🔎 คำค้นที่ใช้: {message['query']}")
        for i, src in enumerate(sources, start=1):
            mark = "✅" if i in cited else "▫️"
            st.markdown(f"{mark} **[{i}] {src['header']}**  \n"
                        f"`{src['source']}` · ความใกล้เคียง {src['score']:.3f}")
            st.text(src["text"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

store = load_vector_store()
api_key = get_api_key()

with st.sidebar:
    st.header("🏠 หอพัก มจพ. ปราจีนบุรี")
    st.write("ถามเรื่องค่าเช่า ค่าน้ำไฟ กฎระเบียบ เวลาปิดหอ การแจ้งซ่อม Wi-Fi ที่จอดรถ "
             "และเรื่องอื่น ๆ ของหอพักได้ทั้งภาษาไทยและภาษาอังกฤษ")

    st.subheader("ตัวอย่างคำถาม")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.subheader("ตั้งค่า")
    models = available_models(api_key) if api_key else PREFERRED_MODELS
    model = st.selectbox("โมเดล LLM (Groq)", models,
                         index=models.index(default_model()) if default_model() in models else 0)
    top_k = st.slider("จำนวนเอกสารที่ค้นหา (top-k)", min_value=2, max_value=8, value=4)
    if st.button("🗑️ ล้างประวัติแชต", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    docs = {c.source for c in store.chunks}
    st.caption(f"คลังความรู้: {len(docs)} เอกสาร · {len(store.chunks)} chunks  \n"
               f"Embedding: `{rag.EMBED_MODEL}`")
    st.caption("⚠️ ข้อมูลในระบบเป็นข้อมูลจำลองเพื่อการศึกษาในรายวิชา NLP "
               "ไม่ใช่ข้อมูลทางการของมหาวิทยาลัย")

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

st.title("💬 น้องหอ")
st.caption("ผู้ช่วยตอบคำถามหอพักนักศึกษา มจพ. ปราจีนบุรี · ตอบจากเอกสารของหอพักเท่านั้น พร้อมแสดงแหล่งอ้างอิงทุกครั้ง")

if not api_key:
    st.error("ไม่พบ GROQ_API_KEY: ถ้ารันในเครื่องให้ใส่ใน `.streamlit/secrets.toml` "
             "ถ้า deploy บน Streamlit Community Cloud ให้ใส่ในเมนู Settings → Secrets")
    st.stop()
client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🏠"):
        st.markdown("สวัสดีครับ ผม **น้องหอ** 👋 สงสัยเรื่องอะไรเกี่ยวกับหอพัก ถามได้เลยครับ "
                    "หรือกดตัวอย่างคำถามที่แถบด้านซ้าย")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🏠" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message)

question = st.chat_input("พิมพ์คำถามเกี่ยวกับหอพัก...")
question = question or st.session_state.pop("pending_question", None)

if question:
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[-HISTORY_TURNS:]]
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🏠"):
        try:
            with st.spinner("กำลังค้นหาเอกสาร..."):
                query = rag.condense_question(client, model, history, question)
                results = store.search(query, k=top_k)
            answer = st.write_stream(rag.stream_answer(client, model, query, results, history))
        except Exception as e:
            st.error(f"เรียกใช้ LLM ไม่สำเร็จ: {e}\n\nลองเลือกโมเดลอื่นที่แถบด้านซ้าย หรือตรวจสอบ GROQ_API_KEY")
            st.stop()

        reply = {
            "role": "assistant",
            "content": answer,
            "question": question,
            "query": query,
            "sources": [{"source": c.source, "header": c.header, "text": c.text, "score": s}
                        for c, s in results],
        }
        render_sources(reply)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(reply)
