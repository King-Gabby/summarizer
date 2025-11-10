# Smart Summarizer

A Streamlit app to **summarize text**, generate **quizzes**, and provide **text-to-speech (TTS)** functionality.  
You can upload PDFs, DOCX, or TXT files, or paste text manually. Features include:

- ✅ Multi-language TTS (online) and offline voice (pyttsx3)  
- ✅ Save and load files in a local library  
- ✅ Lightweight quiz generation from summaries  
- ✅ Light/Dark theme support  

## Installation

```bash
git clone <your_repo_url>
cd smart-summarizer
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run summarizer.py
# summarizer
This is a hybrid machine learning web-app that is essentially applicable for students, office-workers, professionals and everyday users. 
#core features
summarizer
learn mode
dictionary
library


##Folder structure 
smart-summarizer/
├── summarizer.py
├── requirements.txt
├── library/          # Local storage for uploaded files
├── README.md
└── .gitignore


