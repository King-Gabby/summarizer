import streamlit as st
from transformers import pipeline
import pdfplumber
from docx import Document
import wikipedia
import io
from gtts import gTTS  

# Configuring the app
st.set_page_config(page_title="Smart Summarizer", page_icon="🧠", layout="wide")

st.title("Smart Summarizer & Learn Mode")
st.write("Summarize documents or learn new topics instantly. Ideal for students, professionals, and lifelong learners!")
st.divider()

#summarization model 
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

#text extraction function
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

#text to speech
def generate_audio(summary_text):
    tts = gTTS(text=summary_text, lang='en')
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

#tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Summarizer", "📚 Learn Mode", " 📕 Dictionary", "🏫 Library"])

#tab 1 summarizer
with tab1:
    st.subheader("📄 Upload or Paste Text")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    text_input = st.text_area("Or paste your text here:", height=200, placeholder="Paste your article, notes, or essay...")

    text = ""
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            text = extract_text_from_docx(uploaded_file)
        elif file_type == "txt":
            text = uploaded_file.read().decode("utf-8")
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    elif text_input.strip():
        text = text_input.strip()

    st.sidebar.subheader("⚙️ Summary Settings")
    length_option = st.sidebar.slider("Select summary length:", 50, 500, 150, step=50)
    st.sidebar.caption("💡 Tip: Longer summaries capture more details, while shorter ones give concise overviews.")

    if text:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing... please wait ⏳"):
                try:
                    summary = summarizer(
                        text[:2000],
                        max_length=length_option,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
                    st.success("Summary Generated!")
                    st.write("### 📝 Your Summary:")
                    st.write(summary)
                    st.code(summary, language="text")

                    # 🎧 Add Voice Narration
                    audio_data = generate_audio(summary)
                    st.audio(audio_data, format='audio/mp3')
                    st.download_button("Download Summary as .txt", data=summary, file_name="summary.txt", mime="text/plain")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
    else:
        st.info("Upload a file or paste text to get started.")


#tab2. Learn mode
with tab2:
    st.subheader("📚 Explore and Learn")
    topic = st.text_input("Enter a topic to learn about: ", placeholder= "e.g Artificial intelligence, Space, Blockchain...")

    length_option2 = st.slider("Summary length:", 50, 400, 150, step=50, key="learn_length")
    if topic:
        if st.button("🔎 Search & Summarize", key="learn_button"):
            with st.spinner(f"Fetching and summarizing '{topic}'... ⏳"):
                try:
                    wiki_content = wikipedia.page(topic).content
                    summary = summarizer(
                        wiki_content[:2000],
                        max_length=length_option2,
                        min_length=30,
                        do_sample=False
                    )[0]['summary_text']
                    st.success(f"Here's what I learned about **{topic}**:")
                    st.write(summary)

                    # 🎧 Add Voice Narration
                    audio_data = generate_audio(summary)
                    st.audio(audio_data, format='audio/mp3')

                    st.download_button(
                        "Download Learning Summary",
                        data=summary,
                        file_name=f"{topic}_summary.txt",
                        mime="text/plain"
                    )
                except wikipedia.exceptions.DisambiguationError as e:
                    st.error(f"⚠️ Too many results for '{topic}'. Try being more specific.")
                except wikipedia.exceptions.PageError:
                    st.error("❌ Topic not found. Please try another search.")
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
    else:
        st.info("💡 Enter a topic above to explore and learn something new!")

with tab3:
    st.subheader("🪄 Learn a new word today")
    dic_box = st.text_area("Enter a word: ", height=100, placeholder= "")

with tab4:
    col1, col2 =  st.columns([1,2])
    with col1:
        st.subheader("Books")
    with col2:
        st.subheader("Courses")

#my footer
st.divider()
st.write("Learn Faster, Work Smarter. Powered by CelesTium AI 🤖")
st.caption("Built by Gabriel ❤️✨| ")
#Powered by HuggingFace, Streamlit, and Wikipedia | 🎧 Voice by gTTS")
