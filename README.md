# NLP Preprocessing Visualizer

This mini project turns a simple NLP preprocessing pipeline into an educational Streamlit page. Beginners can paste any English text and see how it changes across the classic cleaning steps.

## Project layout

```
.
├── app.py             # Core preprocessing helpers (cleaning, tokenizing, etc.)
├── streamlit_app.py   # Streamlit interface that visualizes every step
├── README.md          # This guide
└── requirements.txt   # Python dependencies
```

## Quick start

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate      # macOS / Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open the URL** shown in the terminal (usually http://localhost:8501) and try your own text.

## What you can learn
- How raw text is cleaned (lowercasing, removing URLs/numbers/punctuation, trimming spaces).
- How tokenization, stopword removal, stemming, and lemmatization change the text.
- How many tokens you lose/gain at each step.
- A side-by-side comparison table showing each word with its stemmed and lemmatized versions.

## Notes
- The app downloads required NLTK resources automatically the first time you run it.
- Everything is written in clear, beginner-friendly language so students can focus on the concepts rather than code details.
