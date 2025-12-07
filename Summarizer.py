import streamlit as st
from transformers import pipeline
import pdfplumber
from docx import Document
import io, os, re, time, random, json
from gtts import gTTS
import pyttsx3

# Text chunking config
max_chunk_words = 300   
# Optional / advanced libs (graceful)
try:
    from streamlit_quill import st_quill
    HAS_QUILL = True
except Exception:
    HAS_QUILL = False

# KeyBERT for advanced keywords (optional)
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    HAS_KEYBERT = True
except Exception:
    HAS_KEYBERT = False

# PDF writer fallback (optional)
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# ----------------- Helpers -----------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def chunk_text(text: str, max_words: int = 900):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

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

# Simple heuristics for focus modes
_DATE_REGEX = re.compile(r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)
_ACTION_KEYWORDS = ["should", "must", "need to", "complete", "do", "implement", "fix", "action", "deadline", "due", "assign"]

def extract_key_dates(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    date_sents = [s for s in sentences if _DATE_REGEX.search(s)]
    seen, out = set(), []
    for s in date_sents:
        k = s[:160]
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out

def extract_action_items(text):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    action_sents = []
    for s in sentences:
        low = s.lower()
        if any(kw in low for kw in _ACTION_KEYWORDS) or re.match(r'^[A-Z][a-z]+\s', s) and low.split()[0].endswith(':'):
            action_sents.append(s)
    return action_sents

# ----------------- Advanced keyword extraction (KeyBERT) -----------------
@st.cache_resource
def _load_keybert_models():
    if not HAS_KEYBERT:
        return None
    # use a light sentence-transformers model to keep memory reasonable
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    kw = KeyBERT(model=embed_model)
    return kw

def extract_keywords_keybert(text, top_n=10):
    kw_model = _load_keybert_models()
    if not kw_model:
        return []
    # returns list of (phrase, score)
    kws = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=top_n)
    return [k for k,score in kws]

# Fallback simple extractor (frequency + YAKE if available)
try:
    import yake
    HAS_YAKE = True
except Exception:
    HAS_YAKE = False

def extract_keywords_fallback(text, top_n=10):
    if HAS_YAKE:
        kw_ex = yake.KeywordExtractor(top=top_n, n=2)
        kws = kw_ex.extract_keywords(text)
        return [k for k,score in kws]
    # frequency fallback
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    STOPWORDS = set(["the","and","that","this","with","from","which","have","has","were","will","would","there","their","these","those","about","what","when","where","how","why","for","not","but"])
    freqs = {}
    for w in words:
        if w in STOPWORDS: continue
        freqs[w] = freqs.get(w,0) + 1
    sorted_words = sorted(freqs.items(), key=lambda x:x[1], reverse=True)
    return [w for w,_ in sorted_words][:top_n]

def extract_keywords(text, top_n=10):
    # prefer KeyBERT if available
    if HAS_KEYBERT:
        try:
            kws = extract_keywords_keybert(text, top_n=top_n)
            if kws: return kws
        except Exception:
            pass
    # fallback to YAKE or frequency
    return extract_keywords_fallback(text, top_n=top_n)

# ----------------- Quiz helpers -----------------
def make_quiz_from_summary(summary_text, num_questions=5):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
    if not sentences: return []
    # extract phrases/keywords
    keywords = extract_keywords(summary_text, top_n=max(15, num_questions*3))
    if not keywords:
        return []
    random.shuffle(keywords)
    used = set()
    questions = []
    for k in keywords:
        if len(questions) >= num_questions: break
        chosen_sentence = next((s for s in sentences if re.search(rf"\b{re.escape(k)}\b", s, re.I)), None)
        if not chosen_sentence:
            chosen_sentence = random.choice(sentences)
        blank_sentence = re.sub(rf"(?i)\b{re.escape(k)}\b", "______", chosen_sentence, count=1)
        distractors = [w for w in keywords if w.lower() != k.lower() and w not in used]
        random.shuffle(distractors)
        choices = [k] + distractors[:3]
        choices = [c.capitalize() for c in choices]
        random.shuffle(choices)
        questions.append({"question": blank_sentence, "choices": choices, "answer": k.capitalize()})
        used.add(k)
    random.shuffle(questions)
    return questions

# ----------------- TTS helper -----------------
def generate_audio(summary_text, offline_mode=False, lang="en"):
    audio_bytes = io.BytesIO()
    try:
        if offline_mode:
            engine = pyttsx3.init()
            tmp = "temp_audio.mp3"
            engine.save_to_file(summary_text, tmp)
            engine.runAndWait()
            with open(tmp, "rb") as f:
                audio_bytes.write(f.read())
            audio_bytes.seek(0)
            try:
                os.remove(tmp)
            except Exception:
                pass
        else:
            tts = gTTS(text=summary_text, lang=lang)
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Smart Summarizer", page_icon="📄", layout="wide")
st.title("AI Summarizer. Summary• Quiz • Listen • Library")

# Sidebar controls
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Model (quality vs speed):", ["DistilBART (fast)", "BART Large (high quality)", "T5 Small (compact)"], index=0)
model_map = {"DistilBART (fast)":"sshleifer/distilbart-cnn-12-6", "BART Large (high quality)":"facebook/bart-large-cnn", "T5 Small (compact)":"t5-small"}
selected_model = model_map[model_choice]

summary_style = st.sidebar.radio("Summary style:", ["Concise","Balanced","Detailed"], index=1)
if summary_style == "Concise": min_len,max_len = 20,80
elif summary_style == "Detailed": min_len,max_len = 80,350
else: min_len,max_len = 40,180

focus_option = st.sidebar.selectbox("Summary focus (context-aware):", ["General Summary","Main Ideas","Key Dates / Timeline","Action Items"])
progressive_mode = st.sidebar.checkbox("Progressive streaming summary", value=True)

# Rich editor minimal toggle (include hyperlinks)
enable_quill = st.sidebar.checkbox("Use minimal rich-text editor (Quill) if available", value=True)
enable_quill = enable_quill and HAS_QUILL

# Voice / TTS
enable_voice = st.sidebar.checkbox("Enable voice (play & download)", value=True)
offline_voice = st.sidebar.checkbox("Offline TTS (pyttsx3) fallback", value=False)
tts_lang_map = {"English":"en","Spanish":"es","French":"fr","German":"de","Hindi":"hi","Chinese":"zh-CN"}
tts_lang_choice = st.sidebar.selectbox("TTS Language:", list(tts_lang_map.keys()), index=0)
tts_lang = tts_lang_map[tts_lang_choice]

# Quiz UX
enable_quiz = st.sidebar.checkbox("Enable Quiz", value=True)
num_quiz_qs = st.sidebar.slider("Number of quiz questions", 1, 20, 5)
immediate_feedback = st.sidebar.checkbox("Immediate per-question feedback", value=True)
quiz_timer_enabled = st.sidebar.checkbox("Enable quiz timer (per quiz)", value=False)
quiz_time_seconds = st.sidebar.slider("Quiz time (seconds)", 30, 900, 180, step=30)

# Library & chunks
#max_chunk_words = st.sidebar.slider("Chunk size (words)", 400, 1200, 900, step=100)

# Theme toggle
theme_choice = st.sidebar.radio("Theme:", ["Dark","Light"], index=0)
if theme_choice == "Light":
    st.markdown("<style>body, .stApp{background-color:#fff;color:#000}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body, .stApp{background-color:#0e1117;color:#fff}</style>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("Built by Gabryel. Minimal editor, streaming summaries, KeyBERT support")

# Cached HF model loader
@st.cache_resource
def load_summarizer_model(model_name):
    return pipeline("summarization", model=model_name)

# Session defaults
if "model_obj" not in st.session_state: st.session_state["model_obj"] = None
if "input_text" not in st.session_state: st.session_state["input_text"] = ""
if "summary_text" not in st.session_state: st.session_state["summary_text"] = ""
if "last_quiz" not in st.session_state: st.session_state["last_quiz"] = {"id":None,"summary":"","questions":[]}
if "quiz_answers" not in st.session_state: st.session_state["quiz_answers"] = {}
if "quiz_started_at" not in st.session_state: st.session_state["quiz_started_at"] = None

# Library folder
LIB_FOLDER = "library"
os.makedirs(LIB_FOLDER, exist_ok=True)

# ---------- Page layout: Tabs ----------
tab_sum, tab_lib, tab_quiz, tab_study= st.tabs(["📄 Summarizer","📚 Library","📊 Quiz", "📖 Study Mode"])

# ----- Library tab -----
with tab_lib:
    st.header("Library. Local files")
    st.markdown("Upload course materials (PDF, DOCX, TXT). Add a category tag after uploading.")
    uploaded = st.file_uploader("Upload file", type=["pdf","docx","txt"], key="lib_upl")
    cat = st.selectbox("Category (tag) for upload:", ["Textbook","Lecture Notes","Article","Other"])
    if uploaded:
        safe = re.sub(r"[^\w\-_\. ]","_", uploaded.name)
        save_path = os.path.join(LIB_FOLDER, safe)
        with open(save_path, "wb") as f: f.write(uploaded.getbuffer())
        # attach metadata JSON
        meta = {"name": safe, "category": cat, "time": time.time()}
        meta_path = save_path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as m:
            json.dump(meta, m)
        st.success(f"Saved {safe} in library as {cat}")

    st.markdown("### Files")
    files = [f for f in sorted(os.listdir(LIB_FOLDER)) if not f.endswith(".meta.json")]
    search_q = st.text_input("Search files (filename):", key="lib_search")
    if search_q:
        files = [f for f in files if search_q.lower() in f.lower()]

    if not files:
        st.info("Library empty. Upload a file to start.")
    else:
        for fn in files:
            c1,c2,c3,c4 = st.columns([3,1,1,1])
            # show category if exists
            meta_path = os.path.join(LIB_FOLDER, fn + ".meta.json")
            cat_label = ""
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as mm:
                        cat_label = json.load(mm).get("category","")
                except Exception:
                    cat_label = ""
            c1.write(f"{fn} {f' — {cat_label}' if cat_label else ''}")
            if c2.button("Preview", key=f"preview_{fn}"):
                try:
                    p = os.path.join(LIB_FOLDER, fn)
                    if fn.lower().endswith(".pdf"):
                        txt = extract_text_from_pdf(open(p,"rb"))[:2500] or "No readable text."
                    elif fn.lower().endswith(".docx"):
                        txt = extract_text_from_docx(open(p,"rb"))[:2500] or "No readable text."
                    else:
                        with open(p,"r",encoding="utf-8",errors="ignore") as f:
                            txt = f.read(2500)
                except Exception as e:
                    txt = f"Preview failed: {e}"
                st.info(txt)
            if c3.button("Load to Summarizer", key=f"load_{fn}"):
                try:
                    p = os.path.join(LIB_FOLDER, fn)
                    if fn.lower().endswith(".pdf"):
                        loaded = extract_text_from_pdf(open(p,"rb"))
                    elif fn.lower().endswith(".docx"):
                        loaded = extract_text_from_docx(open(p,"rb"))
                    else:
                        with open(p,"r",encoding="utf-8",errors="ignore") as f:
                            loaded = f.read()
                    st.session_state["input_text"] = clean_text(loaded)
                    st.success(f"Loaded '{fn}' into Summarizer.")
                except Exception as e:
                    st.error(f"Load failed: {e}")
            if c4.button("Delete", key=f"del_{fn}"):
                try:
                    os.remove(os.path.join(LIB_FOLDER, fn))
                    if os.path.exists(os.path.join(LIB_FOLDER, fn + ".meta.json")):
                        os.remove(os.path.join(LIB_FOLDER, fn + ".meta.json"))
                    st.warning(f"Deleted {fn}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# ----- Summarizer tab -----
with tab_sum:
    st.header("Summarizer")
    # Input editor (minimal rich editor if available)
    if enable_quill and HAS_QUILL:
        st.markdown("**Paste/write into the editor**")
        # minimal toolbar config — only limited buttons relevant to summarizer
        toolbar = [
            ["bold","italic","underline"], ["blockquote"], [{"list":"ordered"},{"list":"bullet"}], ["link"], [{"header":[1,2,3]}]
        ]
        quill_content = st_quill(key="quill", toolbar=toolbar)
        if quill_content:
            # quill returns HTML-like string — strip tags for summarizer
            plain = re.sub(r"<[^>]+>", " ", quill_content)
            st.session_state["input_text"] = clean_text(plain)
    else:
        ta = st.text_area("Paste or type text here (or load from Library)", value=st.session_state.get("input_text",""), height=320)
        st.session_state["input_text"] = ta

    # Controls and quick info

    # Summarize action
    summary_size = st.slider("Approx summary size (words)", 50, 500, 150, step=10)
    gen_btn = st.button("Generate Summary")
    if gen_btn:
        if not st.session_state.get("input_text","").strip():
            st.warning("No input text found.")
        else:
            text_full = st.session_state["input_text"]
            summary_result = ""
            # Heuristic modes first
            if focus_option == "Key Dates / Timeline" or focus_option == "Main Ideas" and False:
                # keep Key Dates mode
                pass

            if focus_option == "Key Dates / Timeline":
                dates = extract_key_dates(text_full)
                if dates:
                    summary_result = "\n".join([f"- {d}" for d in dates])
                else:
                    st.info("No explicit dates found — falling back to model.")
            elif focus_option == "Action Items":
                actions = extract_action_items(text_full)
                if actions:
                    summary_result = "\n".join([f"- {a}" for a in actions])
                else:
                    st.info("No action-like sentences found — falling back to model.")

            # Use model for main/general or when heuristic fails
            if not summary_result:
                with st.spinner("Summarizing..."):
                    try:
                        if st.session_state["model_obj"] is None:
                            st.session_state["model_obj"] = load_summarizer_model(selected_model)
                        model_obj = st.session_state["model_obj"]
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
                        st.stop()

                    chunks = chunk_text(text_full, max_words=max_chunk_words)
                    partial = []
                    stream_placeholder = st.empty()
                    prog = st.progress(0)
                    for i, ch in enumerate(chunks):
                        try:
                            out = model_obj(ch, max_length=summary_size, min_length=max(20, int(summary_size*0.25)), do_sample=False)
                            if isinstance(out, list) and isinstance(out[0], dict) and "summary_text" in out[0]:
                                text_part = out[0]["summary_text"]
                            elif isinstance(out, list) and isinstance(out[0], str):
                                text_part = out[0]
                            elif isinstance(out, dict) and "summary_text" in out:
                                text_part = out["summary_text"]
                            else:
                                text_part = str(out)
                        except Exception:
                            text_part = ""
                        partial.append(text_part)
                        if progressive_mode:
                            stream_placeholder.markdown("### Summary (streaming)\n\n" + clean_text(" ".join(partial)))
                        prog.progress((i+1)/len(chunks))
                    summary_result = clean_text(" ".join([p for p in partial if p]))

            st.session_state["summary_text"] = summary_result

    # Display summary and extras
    if st.session_state.get("summary_text",""):
        st.success("Summary ready")
        st.markdown("### Summary")
        st.write(st.session_state["summary_text"])
        st.caption(f"Original words: {len(st.session_state.get('input_text','').split())}  |  Summary words: {len(st.session_state.get('summary_text','').split())}")

        # Copy summary button
        if st.button("Copy summary to clipboard"):
            st.experimental_set_query_params()
            st.write("Use your browser or OS copy shortcut: cmd/cntrl+C on the summary area (Streamlit cannot copy to system clipboard programmatically in all browsers).")
            # Note: Streamlit web can't always write to OS clipboard; keep instruction.

        # Download TXT
        st.download_button("Download summary (.txt)", data=st.session_state["summary_text"], file_name="summary.txt", mime="text/plain")

        # Download PDF (if fpdf available)
        if HAS_FPDF:
            def make_pdf_bytes(text):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                for line in text.split("\n"):
                    pdf.multi_cell(0, 7, line)
                bio = io.BytesIO()
                pdf.output(dest="S").encode("latin-1")
                # FPDF output returns bytes via output(dest="S")
                bio.write(pdf.output(dest="S").encode("latin-1"))
                bio.seek(0)
                return bio
            pdf_b = make_pdf_bytes(st.session_state["summary_text"])
            st.download_button("Download summary (PDF)", data=pdf_b.getvalue(), file_name="summary.pdf", mime="application/pdf")
        else:
            st.info("PDF export requires 'fpdf' package. Install to enable PDF download.")

        # Audio
        if enable_voice:
            with st.spinner("Generating audio..."):
                audio_bytes = generate_audio(st.session_state["summary_text"], offline_mode=offline_voice, lang=tts_lang)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button("Download audio (mp3)", data=audio_bytes.getvalue(), file_name="summary.mp3", mime="audio/mp3")
                else:
                    st.warning("Audio generation failed (try toggling offline/online TTS).")

        # Prepare quiz if enabled
        if enable_quiz:
            if st.session_state["last_quiz"]["summary"] != st.session_state["summary_text"]:
                questions = make_quiz_from_summary(st.session_state["summary_text"], num_questions=num_quiz_qs)
                st.session_state["last_quiz"] = {"id": int(time.time()*1000), "summary": st.session_state["summary_text"], "questions": questions}
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_started_at"] = None

            st.info("Quiz created — open the Quiz tab or take it now.")
            if st.button("Take Quiz now"):
                st.experimental_set_query_params(tab="quiz")
                st.rerun()

# ----- Quiz tab -----
with tab_quiz:
    st.header("Interactive Quiz")
    quiz_data = st.session_state["last_quiz"].get("questions", [])
    if not quiz_data:
        st.info("No quiz available. Generate a summary first.")
    else:
        qid = st.session_state["last_quiz"]["id"]
        total_q = len(quiz_data)
        st.markdown(f"**Quiz ({total_q} questions)**")
        # initialize storage
        if "quiz_answers" not in st.session_state:
            st.session_state["quiz_answers"] = {}
        # start timer if enabled
        if quiz_timer_enabled and st.session_state.get("quiz_started_at") is None:
            st.session_state["quiz_started_at"] = time.time()

        # compute remaining time
        if quiz_timer_enabled:
            elapsed = int(time.time() - st.session_state.get("quiz_started_at", time.time()))
            remaining = max(0, quiz_time_seconds - elapsed)
            st.metric("Time left (s)", remaining)
            if remaining == 0:
                st.warning("Time is up. Auto-submitting...")
                submit_now = True
            else:
                submit_now = False
        else:
            submit_now = False

        # progress bar
        answered_count = len([v for v in st.session_state.get("quiz_answers", {}).values() if v is not None])
        p = st.progress(answered_count / total_q if total_q > 0 else 0.0)

        # show questions
        for idx, q in enumerate(quiz_data, start=1):
            q_key = f"quiz_{qid}_{idx}"
            st.markdown(f"**Q{idx}.** {q['question']}")
            choices = q["choices"] + ["Skip"]
            # default
            prev = st.session_state["quiz_answers"].get(q_key, None)
            if prev is None:
                # Streamlit radio needs default selection; we present 'Choose...' as first element
                radio_choices = ["Choose..."] + choices
                sel = st.radio(f"Select (Q{idx})", radio_choices, key=q_key)
                if sel == "Choose...":
                    st.session_state["quiz_answers"][q_key] = None
                elif sel == "Skip":
                    st.session_state["quiz_answers"][q_key] = None
                else:
                    st.session_state["quiz_answers"][q_key] = sel
            else:
                # ensure prev still in choices
                if prev not in choices:
                    radio_choices = ["Choose..."] + choices
                    sel = st.radio(f"Select (Q{idx})", radio_choices, index=0, key=q_key)
                else:
                    radio_choices = ["Choose..."] + choices
                    sel_idx = radio_choices.index(prev) if prev in radio_choices else 0
                    sel = st.radio(f"Select (Q{idx})", radio_choices, index=sel_idx, key=q_key)
                if sel == "Choose..." or sel == "Skip":
                    st.session_state["quiz_answers"][q_key] = None
                else:
                    st.session_state["quiz_answers"][q_key] = sel

            # Immediate feedback
            if immediate_feedback:
                ans = st.session_state["quiz_answers"].get(q_key)
                if ans is None:
                    st.info("No answer selected.")
                else:
                    if ans == q["answer"]:
                        st.success("Correct ✅")
                    else:
                        st.error(f"Incorrect ❌ — Correct: {q['answer']}")

            answered_count = len([v for v in st.session_state.get("quiz_answers", {}).values() if v is not None])
            p.progress(answered_count / total_q if total_q > 0 else 0.0)
            st.markdown("---")

        # submit
        if submit_now or st.button("Submit Quiz"):
            correct = 0
            for idx, q in enumerate(quiz_data, start=1):
                q_key = f"quiz_{qid}_{idx}"
                ans = st.session_state["quiz_answers"].get(q_key)
                if ans == q["answer"]:
                    correct += 1
            score = (correct / total_q) * 100 if total_q > 0 else 0
            st.success(f"Final score: {correct}/{total_q} ({score:.1f}%)")
            if score == 100:
                st.balloons()
            elif score >= 70:
                st.info("Great job!")
            elif score >= 40:
                st.warning("Partial — review & retry.")
            else:
                st.error("Keep studying!")

            # Save / reset / export
            c1, c2, c3 = st.columns([1,1,1])
            if c1.button("Save results locally"):
                out = {"time": time.time(), "correct": correct, "total": total_q, "score": score, "summary_hash": hash(st.session_state.get("summary_text",""))}
                save_path = "quiz_results.json"
                try:
                    existing = []
                    if os.path.exists(save_path):
                        with open(save_path, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                    existing.append(out)
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(existing, f, indent=2)
                    st.success("Saved locally.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

            if c2.button("Reset quiz (new attempt)"):
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_started_at"] = None
                st.erun()

            if c3.button("Export results (.json)"):
                data = json.dumps({"time": time.time(), "correct": correct, "total": total_q, "score": score}, indent=2)
                st.download_button("Download results", data=data, file_name="quiz_result.json", mime="application/json")
# Footer
st.markdown("---")
st.caption("Built by Gabriel. Smart Summarizer • Context-aware • KeyBERT support (optional) • Minimal editor • Streaming summaries")
