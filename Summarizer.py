import streamlit as st
from transformers import pipeline
import pdfplumber
from docx import Document
import io
import pyttsx3
import random
import re
import os
import time
from gtts import gTTS

# --- Helper functions ---
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception:
        try:
            file.seek(0)
            text = file.read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    return text

def extract_text_from_docx(file):
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        try:
            file.seek(0)
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

def extract_text_from_txt(file):
    try:
        file.seek(0)
        return file.read().decode("utf-8")
    except Exception:
        return ""

def clean_text(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def chunk_text(text, max_words=900):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# --- Quiz helpers ---
STOPWORDS = {
    "the", "and", "is", "in", "it", "of", "to", "a", "that", "this", "for", "with",
    "as", "on", "are", "was", "be", "by", "an", "or", "from", "at", "which", "we",
    "were", "their", "has", "have", "but", "not", "they", "you", "i", "he", "she",
    "his", "her", "its", "will", "can", "all", "about"
}


def extract_candidate_keywords(text, top_k=10):
    """Simple frequency-based keyword extraction."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freqs = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1
    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words][:top_k]


def make_quiz_from_summary(summary_text, num_questions=5):
    """Create fill-in-the-blank multiple choice quiz questions."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
    if not sentences:
        return []

    keywords = extract_candidate_keywords(summary_text, top_k=max(15, num_questions * 3))
    if not keywords:
        return []

    random.shuffle(keywords)
    used_keywords = set()
    questions = []

    for k in keywords:
        if len(questions) >= num_questions:
            break
        # pick a sentence containing the keyword
        chosen_sentence = next((s for s in sentences if re.search(rf"\b{k}\b", s, re.I)), None)
        if not chosen_sentence:
            chosen_sentence = random.choice(sentences)

        blank_sentence = re.sub(rf"(?i)\b{k}\b", "_____", chosen_sentence, count=1)
        distractors = [w for w in keywords if w != k and w not in used_keywords]
        random.shuffle(distractors)
        choices = [k] + distractors[:3]

        # capitalize choices for readability
        choices = [c.capitalize() for c in choices]
        random.shuffle(choices)

        questions.append({
            "question": blank_sentence,
            "choices": choices,
            "answer": k.capitalize()
        })
        used_keywords.add(k)

    random.shuffle(questions)
    return questions


def run_quiz(summary_text, user_num_questions=5):
    """Streamlit quiz interface with shuffled questions and score tracking."""
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {
            "questions": make_quiz_from_summary(summary_text, user_num_questions),
            "answers": {},
            "submitted": False
        }

    quiz_state = st.session_state.quiz_state
    questions = quiz_state["questions"]

    if not questions:
        st.info("⚠️ Could not generate quiz questions from this summary.")
        return

    st.markdown("## Knowledge Quiz")
    st.caption("Answer all questions and click *Submit Quiz* below to see your score.")

    # Render questions
    for i, q in enumerate(questions, start=1):
        st.markdown(f"**Q{i}.** {q['question']}")
        choice = st.radio(
            f"Select your answer for Q{i}",
            q["choices"],
            key=f"quiz_q_{i}",
            index=0 if f"quiz_q_{i}" not in st.session_state else q["choices"].index(st.session_state.get(f'quiz_q_{i}', q["choices"][0])),
        )
        quiz_state["answers"][i] = choice

    # Submit and scoring
    if st.button("Submit Quiz"):
        correct = 0
        total = len(questions)

        for i, q in enumerate(questions, start=1):
            user_ans = quiz_state["answers"].get(i)
            if user_ans == q["answer"]:
                correct += 1

        score = (correct / total) * 100
        st.session_state.quiz_state["submitted"] = True
        st.success(f"You scored {correct}/{total} ({score:.1f}%) ✅")

        if score == 100:
            st.balloons()
        elif score >= 70:
            st.info("Great job!🔥 You understood most of the summary.")
        elif score >= 40:
            st.warning("Not bad.☺️ But you can do better! Review the summary again.")
        else:
            st.error("Keep studying!😉 Try summarizing again and retake the quiz.")

    # Reset option
    if st.session_state.quiz_state.get("submitted", False):
        if st.button("Retake Quiz"):
            st.session_state.quiz_state = {
                "questions": make_quiz_from_summary(summary_text, user_num_questions),
                "answers": {},
                "submitted": False
            }
            st.rerun()


# --- TTS helper ---
def generate_audio(summary_text, offline_mode=False, lang="en"):
    audio_bytes = io.BytesIO()
    try:
        if offline_mode:
            engine = pyttsx3.init()
            engine.save_to_file(summary_text, "temp_audio.mp3")
            engine.runAndWait()
            with open("temp_audio.mp3", "rb") as f:
                audio_bytes.write(f.read())
            audio_bytes.seek(0)
            if os.path.exists("temp_audio.mp3"):
                os.remove("temp_audio.mp3")
        else:
            tts = gTTS(text=summary_text, lang=lang)
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"TTS generation failed: {e}")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="Smart Summarizer", page_icon="🧠", layout="wide")
st.title("StudySpark. Summarize • Quiz • Listen • Library")
st.markdown("Upload/paste text, summarize, create a quiz, listen, and store materials in your local library.")

# Sidebar settings
st.sidebar.header("Configure Summary ⚙️")
model_choice = st.sidebar.selectbox(
    "Choose Model:",
    ("DistilBART (fast)", "BART Large (high quality)", "T5 Small (compact)"),
    index=0
)
model_map = {
    "DistilBART (fast)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (high quality)": "facebook/bart-large-cnn",
    "T5 Small (compact)": "t5-small"
}
selected_model = model_map[model_choice]

summary_style = st.sidebar.radio("Summary style:", ("Concise", "Balanced", "Detailed"), index=1)
if summary_style == "Concise":
    min_len, max_len = 20, 80
elif summary_style == "Detailed":
    min_len, max_len = 80, 350
else:
    min_len, max_len = 40, 180

enable_voice = st.sidebar.checkbox("Enable voice (play & download)", value=True)
offline_mode = st.sidebar.checkbox("Use Offline Voice (pyttsx3)", value=False)
tts_lang = st.sidebar.selectbox(
    "Select Language:", 
    ["en", "es", "fr", "de", "hi", "ja", "zh"], 
    index=0
)
enable_quiz = st.sidebar.checkbox("Enable Quiz Generation", value=True)
num_quiz_qs = st.sidebar.slider("Number of quiz questions", 1, 20, 3)
max_chunk_words = st.sidebar.slider("Chunk size (words)", 400, 1200, 900, step=100)

theme_choice = st.sidebar.radio("Select Theme:", ["Light Mode", "Dark Mode"], index=1)
if theme_choice == "Light Mode":
    st.markdown("""
    <style>
    body, .stApp {background-color: #ffffff; color: #000000;}
    .stTextInput>div>input, .stTextArea>div>textarea {background-color:#f0f0f0; color:#000;}
    .stButton>button {background-color:#e0e0e0; color:#000;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body, .stApp {background-color: #0e1117; color: #ffffff;}
    .stTextInput>div>input, .stTextArea>div>textarea {background-color:#1c1f26; color:#fff;}
    .stButton>button {background-color:#2a2d36; color:#fff;}
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Gabryel. For Students & Professionals")

# Load summarizer model
@st.cache_resource
def load_summarizer_model(model_name):
    return pipeline("summarization", model=model_name)

# Session state init
if "model_obj" not in st.session_state:
    st.session_state["model_obj"] = None
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "summary_text" not in st.session_state:
    st.session_state["summary_text"] = ""
if "last_quiz" not in st.session_state:
    st.session_state["last_quiz"] = {"id": None, "questions": []}
if "quiz_answers" not in st.session_state:
    st.session_state["quiz_answers"] = {}

# --- Tabs ---
tab_summarizer, tab_library = st.tabs(["📄 Summarizer", "📚 Library"])

# Library tab
LIB_FOLDER = "library"
os.makedirs(LIB_FOLDER, exist_ok=True)
with tab_library:
    st.header("Library")
    st.markdown("Upload and store course materials locally.")
    uploaded_library_file = st.file_uploader("Upload file (PDF/DOCX/TXT)", type=["pdf","docx","txt"], key="lib_uploader")
    if uploaded_library_file:
        safe_name = re.sub(r"[^\w\-_\. ]","_", uploaded_library_file.name)
        save_path = os.path.join(LIB_FOLDER, safe_name)
        with open(save_path,"wb") as f: f.write(uploaded_library_file.getbuffer())
        st.success(f"Saved: {safe_name}")

    st.markdown("### Library Files")
    files = sorted(os.listdir(LIB_FOLDER))
    search_q = st.text_input("Search files:", key="lib_search")
    if search_q: files = [f for f in files if search_q.lower() in f.lower()]
    if files:
        for fn in files:
            col_a, col_b, col_c, col_d = st.columns([3,1,1,1])
            col_a.write(fn)
            if col_b.button("Preview", key=f"preview_{fn}"):
                try:
                    p = os.path.join(LIB_FOLDER, fn)
                    if fn.lower().endswith(".pdf"):
                        preview_text = extract_text_from_pdf(open(p,"rb"))[:1500]
                    elif fn.lower().endswith(".docx"):
                        preview_text = extract_text_from_docx(open(p,"rb"))[:1500]
                    elif fn.lower().endswith(".txt"):
                        preview_text = open(p,"r",encoding="utf-8",errors="ignore").read(1500)
                    else: preview_text = "Preview not supported."
                except Exception as e:
                    preview_text = f"Preview failed: {e}"
                st.info(preview_text)
            if col_c.button("Load into Summarizer", key=f"load_{fn}"):
                path = os.path.join(LIB_FOLDER, fn)
                try:
                    if fn.lower().endswith(".pdf"):
                        loaded = extract_text_from_pdf(open(path,"rb"))
                    elif fn.lower().endswith(".docx"):
                        loaded = extract_text_from_docx(open(path,"rb"))
                    elif fn.lower().endswith(".txt"):
                        loaded = open(path,"r",encoding="utf-8",errors="ignore").read()
                    else: loaded = ""
                    st.session_state["input_text"] = clean_text(loaded)
                    st.success(f"Loaded '{fn}' into Summarizer.")
                except Exception as e: st.error(f"Failed: {e}")
            if col_d.button("Delete", key=f"del_{fn}"):
                try:
                    os.remove(os.path.join(LIB_FOLDER, fn))
                    st.warning(f"Deleted {fn}")
                    st.experimental_rerun()
                except Exception as e: st.error(f"Delete failed: {e}")
    else:
        st.info("Library empty.")

# Summarizer tab
with tab_summarizer:
    st.header("Summarizer")
    col_main, col_info = st.columns([3,1])
    with col_main:
        input_area = st.text_area("Upload/paste text or load from Library", value=st.session_state.get("input_text",""), height=300)
        st.session_state["input_text"] = input_area

        summary_size = st.slider("Approx summary size (words)", 50, 500, 150, step=10)
        if st.button("Generate Summary"):
            if not st.session_state.get("input_text"):
                st.warning("No input text found.")
            else:
                with st.spinner("Summarizing..."):
                    if st.session_state["model_obj"] is None:
                        st.session_state["model_obj"] = load_summarizer_model(selected_model)
                    model_obj = st.session_state["model_obj"]
                    user_max = summary_size
                    user_min = max(20,int(summary_size*0.25))
                    chunks = chunk_text(st.session_state["input_text"], max_words=max_chunk_words)
                    parts = []
                    prog = st.progress(0)
                    for i,ch in enumerate(chunks):
                        try:
                            out = model_obj(ch, max_length=user_max, min_length=user_min, do_sample=False)
                            if isinstance(out,list) and isinstance(out[0],dict) and "summary_text" in out[0]:
                                parts.append(out[0]["summary_text"])
                            else: parts.append(str(out))
                        except Exception:
                            parts.append("")
                        prog.progress((i+1)/len(chunks))
                    summary_text = clean_text(" ".join(parts))
                    st.session_state["summary_text"] = summary_text

        if st.session_state.get("summary_text"):
            st.success("Summary generated ✅")
            st.markdown("### Summary")
            st.write(st.session_state["summary_text"])
            st.caption(f"Original words: {len(st.session_state['input_text'].split())} | Summary words: {len(st.session_state['summary_text'].split())}")

            # Audio
            if enable_voice:
                with st.spinner("Generating audio..."):
                    audio = generate_audio(st.session_state["summary_text"], offline_mode=offline_mode, lang=tts_lang)
                    if audio:
                        st.audio(audio, format="audio/mp3")
                        st.download_button("Download audio (mp3)", data=audio.getvalue(), file_name="summary.mp3", mime="audio/mp3")
                    else:
                        st.warning("Audio generation failed.")

            st.download_button("Download summary (.txt)", data=st.session_state["summary_text"], file_name="summary.txt", mime="text/plain")

            # Quiz
            if enable_quiz:
                current_summary = st.session_state["summary_text"]
                quiz_id = st.session_state["last_quiz"].get("id")
                if quiz_id is None or st.session_state["last_quiz"].get("summary") != current_summary:
                    questions = make_quiz_from_summary(current_summary, num_questions=num_quiz_qs)
                    st.session_state["last_quiz"] = {"id": int(time.time()*1000), "summary": current_summary, "questions": questions}
                    st.session_state["quiz_answers"] = {}

                qs = st.session_state["last_quiz"]["questions"]
                if qs:
                    st.markdown("### Test Yourself QUIZ (TYQ)")
                    show_answers = st.checkbox("Show answers", value=False)
                    for idx, q in enumerate(qs, start=1):
                        q_key = f"quiz_{st.session_state['last_quiz']['id']}_{idx}"
                        if q_key not in st.session_state["quiz_answers"]:
                            st.session_state["quiz_answers"][q_key] = None
                        choice = st.radio(f"Q{idx}: {q['question']}", q["choices"], key=q_key)
                        st.session_state["quiz_answers"][q_key] = choice
                        if show_answers:
                            st.markdown(f"**Answer:** {q['answer']}")
                            if choice == q['answer']: st.success("Correct ✅")
                            else: st.error("Incorrect ❌")

    with col_info:
        st.subheader("Current Config")
        st.write("Model:", model_choice)
        st.write("Style:", summary_style)
        st.write("Voice:", "On" if enable_voice else "Off")
        st.write("Quiz:", "On" if enable_quiz else "Off")
        st.write("Input words:", len(st.session_state.get("input_text","").split()))

st.markdown("---")
st.caption("Built by Gabriel. StudySpark ⚡️ •Summarizer • Quiz • Listen • Local Library")
