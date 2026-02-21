# ====================================================
# Neural Machine Translation App (Medical Domain)
# ====================================================
'''
NLP Applications Assignment 2 Part A
Group No.: 66

Team Members and Contributions:
Sr.No	BITS ID	       Name 	                Contribution
1	   2024aa05973	Sakshi Niranjan Kulkarni	    100%
2	   2024aa05445	Dheeraj Kholia	                100%
3	   2024aa05629	Aditi Tyagi	                    100%
4	   2024ab05283	Addvija Shekhar Medhekar	    100%
5	   2024aa05249	Balamurugan G	                100% 
'''

# Import Libraries
from flask import Flask, render_template, request                           # Flask modules for web application
from transformers import MarianMTModel, MarianTokenizer                     # MarianMT model and tokenizer for translation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction      # BLEU score utilities from NLTK

import nltk                                                                 # For tokenization
nltk.download('punkt')                                                      # Downloading tokenizer resources 

# Initialize Flask application
app = Flask(__name__)

# ----------------------------------------------------
# Load MarianMT Model
# ----------------------------------------------------
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"   # English to Hindi model

# Cache models to avoid reloading every request
loaded_models = {}

def load_model():
    """
    Loads the MarianMT English-Hindi model.
    Models are cached after first load for faster execution.
    """
    if "Hindi" not in loaded_models:
        # Load tokenizer and model from HuggingFace
        tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        model = MarianMTModel.from_pretrained(MODEL_NAME)

        # Store in cache
        loaded_models["Hindi"] = (tokenizer, model)

    return loaded_models["Hindi"]

# ----------------------------------------------------
# Translation Function (Generates Multiple Candidates)
# ----------------------------------------------------
def translate_text(text, tokenizer, model):
    """
    Translates input English medical text into Hindi.

    Beam search is used to generate multiple candidate translations.
    This allows evaluation of multiple outputs as required in the assignment.
    """

    # Convert text into token format required by Transformer
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # Generate translations using beam search
    translated_tokens = model.generate(
        **inputs,
        num_beams=5,              # Beam width
        num_return_sequences=3,   # Generate 3 candidate translations
        early_stopping=True
    )

    # Convert tokens back to readable text
    translations = [
        tokenizer.decode(t, skip_special_tokens=True)
        for t in translated_tokens
    ]

    return translations

# ----------------------------------------------------
# BLEU Score Computation
# ----------------------------------------------------
def compute_bleu(reference, candidate):
    """
    Computes:
    - Overall BLEU score
    - Individual n-gram precision scores
    - Brevity Penalty (BP)

    BLEU evaluates translation quality by comparing candidate
    translation with reference translation.
    """

    # Basic tokenization using whitespace split
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()

    # Apply smoothing to avoid zero scores for short sentences
    smoothing = SmoothingFunction().method4

    # Overall BLEU score
    bleu = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        smoothing_function=smoothing
    )

    # Individual modified n-gram precision scores
    ngrams = {
        "1-gram": round(sentence_bleu(reference_tokens, candidate_tokens, weights=(1, 0, 0, 0)), 4),
        "2-gram": round(sentence_bleu(reference_tokens, candidate_tokens, weights=(0.5, 0.5, 0, 0)), 4),
        "3-gram": round(sentence_bleu(reference_tokens, candidate_tokens, weights=(0.33, 0.33, 0.33, 0)), 4),
        "4-gram": round(sentence_bleu(reference_tokens, candidate_tokens, weights=(0.25, 0.25, 0.25, 0.25)), 4),
    }

    # --------------------------------------
    # Brevity Penalty (Manual Calculation)
    # --------------------------------------
    ref_len = len(reference_tokens[0])
    cand_len = len(candidate_tokens)

    if cand_len > ref_len:
        bp = 1
    else:
        bp = pow(2.718, (1 - ref_len / cand_len)) if cand_len != 0 else 0

    return round(bleu, 4), round(bp, 4), ngrams

# ----------------------------------------------------
# Main Route – User Interaction
# ----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    """
    Handles:
    - User input
    - Model selection
    - Translation generation
    - BLEU evaluation
    """

    results = []
    source_text = ""
    reference_text = ""

    if request.method == "POST":

        # Get source text from UI
        source_text = request.form.get("source_text")

        # Reference translation (manual entry or file upload)
        reference_text = request.form.get("reference_text")
        file = request.files.get("reference_file")

        # If file is uploaded, override manual reference
        if file and file.filename != "":
            reference_text = file.read().decode("utf-8")

        # Load English→Hindi model
        tokenizer, model = load_model()

        # Generate multiple candidate translations
        translations = translate_text(source_text, tokenizer, model)

        # Evaluate each candidate using BLEU
        if reference_text:
            for t in translations:
                bleu, bp, ngrams = compute_bleu(reference_text, t)

                results.append({
                    "translation": t,
                    "bleu": bleu,
                    "bp": bp,
                    "ngrams": ngrams
                })
        else:
            for t in translations:
                results.append({
                    "translation": t,
                    "bleu": "N/A",
                    "bp": "N/A",
                    "ngrams": {}
                })

    return render_template(
        "index.html",
        results=results,
        source_text=source_text,
        reference_text=reference_text,
        selected_language="Hindi"
    )

# -------------------------------------------------
# Run Flask App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
