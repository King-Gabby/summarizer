import streamlit as st
from transformers import pipeline
import pdfplumber
from docx import Document
import io
from gtts import gTTS
import random
import re

#helper & Extraction Utils
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception:
        #fallback: try reading as bytes and decode
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
        #fallback read bytes
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
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text, max_words=900):
    """Split text into chunks of up to max_words words."""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

#lightweight Keyword Quiz
# -------------------------
#very small stopword set so we don't depend on external libs
STOPWORDS = {
    "the","and","is","in","it","of","to","a","that","this","for","with","as","on","are",
    "was","be","by","an","or","from","at","which","we","were","their","has","have","but",
    "not","they","you","I","he","she","his","her","its","will","can","all","about"
}

def extract_candidate_keywords(text, top_k=10):
    """Simple frequency-based keyword extraction from summary text."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freqs = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freqs[w] = freqs.get(w, 0) + 1
    #sort by frequency
    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    keywords = [w for w, _ in sorted_words][:top_k]
    return keywords

def make_quiz_from_summary(summary_text, num_questions=3):
    """
    Creates simple fill-in-the-blank multiple choice questions using keywords found in the summary.
    Each question: sentence with keyword masked, plus 3 distractors.
    Returns: list of dicts {question, choices, answer}
    """
    questions = []
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', summary_text) if s.strip()]
    if not sentences:
        return questions

    keywords = extract_candidate_keywords(summary_text, top_k=max(10, num_questions*3))
    if not keywords:
        return questions

    used_keywords = set()
    random.shuffle(keywords)

    for k in keywords:
        if len(questions) >= num_questions:
            break
        # find a sentence that contains the keyword
        chosen_sentence = None
        for s in sentences:
            if re.search(r"\b" + re.escape(k) + r"\b", s, flags=re.I):
                chosen_sentence = s
                break
        if not chosen_sentence:
            # fallback: pick any sentence
            chosen_sentence = random.choice(sentences)

        #make blank by replacing first occurrence
        pattern = re.compile(r"(?i)\b" + re.escape(k) + r"\b")
        blank_sentence = pattern.sub("_____", chosen_sentence, count=1)

        # create choices: correct = k, others = random words/distractors
        distractors = [w for w in keywords if w != k and w not in used_keywords]
        random.shuffle(distractors)
        choices = [k] + distractors[:3]
        #if not enough distractors, add random words from summary
        if len(choices) < 4:
            extra = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", summary_text.lower()) if w not in choices and w not in STOPWORDS]
            random.shuffle(extra)
            for e in extra:
                if len(choices) >= 4:
                    break
                choices.append(e)
        random.shuffle(choices)
        questions.append({
            "question": blank_sentence,
            "choices": choices,
            "answer": k
        })
        used_keywords.add(k)
    return questions


#TTS (gTTS) helper
def generate_audio(summary_text, lang="en"):
    try:
        tts = gTTS(text=summary_text, lang=lang)
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        return None

#streamlit UI & Logic
st.set_page_config(page_title="Smart Summarizer", page_icon="🧠", layout="wide")
st.title("Smart Summarizer. Summarize • Quiz • Listen")
st.markdown("Upload or paste text, choose a style, generate a summary, optionally create a short quiz, and listen to the summary.")

#sidebar controls (as requested)
st.sidebar.header("Settings")

#model selection. Choose lightweight by default
model_choice = st.sidebar.selectbox(
    "Model (quality vs speed):",
    ("DistilBART (fast)", "BART Large (high quality)", "T5 Small (compact)"),
    index=0
)
model_map = {
    "DistilBART (fast)": "sshleifer/distilbart-cnn-12-6",
    "BART Large (high quality)": "facebook/bart-large-cnn",
    "T5 Small (compact)": "t5-small"
}
selected_model = model_map[model_choice]

#summary style
summary_style = st.sidebar.radio("Summary style:", ("Concise", "Balanced", "Detailed"), index=1)

#map styles to token-length guidance (approximate)
if summary_style == "Concise":
    min_len, max_len = 20, 80
elif summary_style == "Detailed":
    min_len, max_len = 80, 350
else:  # Balanced
    min_len, max_len = 40, 180

#additional sidebar toggles
enable_voice = st.sidebar.checkbox("Enable voice (play & download)", value=True)
enable_quiz = st.sidebar.checkbox("Enable quiz generation", value=True)
num_quiz_qs = st.sidebar.slider("Number of quiz questions", 1, 6, 3)
max_chunk_words = st.sidebar.slider("Chunk size (words)", 400, 1200, 900, step=100)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Gabryel — built for students & professionals")

#load model (cached)

@st.cache_resource
def load_summarizer_model(model_name):
    try:
        return pipeline("summarization", model=model_name)
    except Exception as e:
        # bubbled up - streamlit will show exception on UI
        raise e

#attempt to load model (deferred until user action to avoid long startup)
model_warning = st.empty()
model_obj = None

# Main input area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload (PDF / DOCX / TXT) or paste text below", type=["pdf", "docx", "txt"])
    pasted = st.text_area("Or paste text here:", height=260)

    #use uploaded file if present, otherwise pasted text
    input_text = ""
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "pdf":
            input_text = extract_text_from_pdf(uploaded_file)
        elif file_ext == "docx":
            input_text = extract_text_from_docx(uploaded_file)
        elif file_ext == "txt":
            input_text = extract_text_from_txt(uploaded_file)
        input_text = clean_text(input_text)
        if not input_text:
            st.warning("Uploaded file could not be read or is empty.")
    else:
        input_text = clean_text(pasted)

    if not input_text:
        st.info("Paste text or upload a file to enable summarization.")

with col2:
    st.subheader("Quick Info")
    st.write("**Model:**", model_choice)
    st.write("**Style:**", summary_style)
    st.write("**Voice:**", "On" if enable_voice else "Off")
    st.write("**Quiz:**", "On" if enable_quiz else "Off")
    st.write("Input size (approx words):", len(input_text.split()))

#summarize action
if input_text:
    if st.button("Generate Summary "):
        #load model when user triggers summarize
        with st.spinner("Loading model and summarizing... this may take a moment on first run"):
            try:
                model_obj = load_summarizer_model(selected_model)
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()

            #chunk long input
            chunks = chunk_text(input_text, max_words=max_chunk_words)
            all_parts = []
            progress = st.progress(0)
            for i, chunk in enumerate(chunks):
                try:
                    out = model_obj(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                    # some pipelines return list of dicts or a string; handle both
                    if isinstance(out, list) and isinstance(out[0], dict) and "summary_text" in out[0]:
                        text_part = out[0]["summary_text"]
                    elif isinstance(out, list) and isinstance(out[0], str):
                        text_part = out[0]
                    elif isinstance(out, dict) and "summary_text" in out:
                        text_part = out["summary_text"]
                    else:
                        text_part = str(out)
                except Exception as e:
                    #if a chunk fails, try with smaller chunk size fallback
                    try:
                        #retry single smaller chunk
                        small_chunks = chunk_text(chunk, max_words=max(200, max_chunk_words//2))
                        part_accum = []
                        for sc in small_chunks:
                            r = model_obj(sc, max_length=max_len, min_length=min_len, do_sample=False)
                            if isinstance(r, list) and isinstance(r[0], dict) and "summary_text" in r[0]:
                                part_accum.append(r[0]["summary_text"])
                            else:
                                part_accum.append(str(r))
                        text_part = " ".join(part_accum)
                    except Exception:
                        text_part = ""  # skip if still failing
                all_parts.append(text_part)
                progress.progress((i + 1) / len(chunks))

            summary_text = clean_text(" ".join([p for p in all_parts if p]))

        #display summary
        st.success("Summary generated ✅")
        st.markdown("### Summary")
        st.write(summary_text)
        st.caption(f"Original words: {len(input_text.split())}  |  Summary words: {len(summary_text.split())}")

        #voice (gTTS)
        if enable_voice:
            with st.spinner("Generating audio..."):
                audio_file = generate_audio(summary_text)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
                    st.download_button("Download audio (mp3)", data=audio_file, file_name="summary.mp3", mime="audio/mp3")
                else:
                    st.warning("Audio generation failed. Try again or disable voice.")

        #download summary as text
        st.download_button("Download summary (.txt)", data=summary_text, file_name="summary.txt", mime="text/plain")

        #quiz generation
        if enable_quiz:
            with st.spinner("Creating quiz..."):
                questions = make_quiz_from_summary(summary_text, num_questions=num_quiz_qs)
                if not questions:
                    st.info("Could not generate quiz questions from this summary.")
                else:
                    st.markdown("### Quiz — Test Yourself")
                    show_answers = st.checkbox("Show answers", value=False)
                    for idx, q in enumerate(questions, start=1):
                        st.markdown(f"**Q{idx}.** {q['question']}")
                        choices = q["choices"]
                        # display as radio but not interactive storing (just for display)
                        user_choice = st.radio(f"Choices (Q{idx})", choices, key=f"q_{idx}")
                        if show_answers:
                            st.markdown(f"**Answer:** {q['answer']}")
                        # optional: immediate correctness feedback
                        if show_answers:
                            if user_choice == q['answer']:
                                st.success("Correct ✅")
                            else:
                                st.error("Incorrect ❌")

#footer
st.markdown("---")
st.caption("Built by Gabryel — Summarizer • Quiz • Voice | Model: " + model_choice)
