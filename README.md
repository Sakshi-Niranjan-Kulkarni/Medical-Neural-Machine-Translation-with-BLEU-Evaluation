# Neural Machine Translation (NMT) for Medical Text with BLEU Evaluation

This project implements a **Neural Machine Translation (NMT)** system for translating **medical/healthcare-related text** and evaluating translation quality using the **BLEU score**.
It provides an interactive interface for translating text, comparing it against reference translations, and analysing n-gram precision scores.

---

## Project Overview

The application:

* Translates medical/healthcare text using **Transformer-based NMT models**
* Computes **BLEU score**
* Displays:
  * Model translation output
  * BLEU score
  * Brevity penalty
  * Modified n-gram precision (1-gram, 2-gram, etc.)
* Supports evaluation of **multiple candidate translations**

---

## User Interface Features

### Input

* Text box for **source medical text**
* Option to:

  * Upload a **reference translation file**
  * Manually input reference translation

### Output Display

* NMT-generated translation (English to Hindi language
* BLEU score
* Brevity penalty
* N-gram precision table:
  * 1-gram precision
  * 2-gram precision
  * 3-gram precision
  * 4-gram precision


## Translation & Evaluation Implementation

### 1. NMT Translation

The system uses Transformer-based models such as:
* `Helsinki-NLP MarianMT models`
These models are pre-trained on large bilingual corpora and can be fine-tuned for medical domain adaptation.

---

### 2. BLEU Score Computation

The BLEU score evaluates translation quality by comparing the candidate translation with one or more reference translations.

####  Modified N-gram Precision

For each n-gram level:
		Pn = ∑candidate n-grams / ∑clipped n-gram matches

* Prevents inflated scores from repeated words
* Uses clipped counts based on reference frequency

---

#### Brevity Penalty (BP)

BP = 1                  if c > r

BP = exp(1 - r/c)       if c ≤ r
	​

Where:
* `c` = candidate length
* `r` = reference length

This penalizes overly short translations.

---

#### Final BLEU Score

BLEU=BP × exp(n=1∑N​wn​logPn​)

* Typically computed up to 4-grams
* Uniform weights (`w_n = 0.25`) for 1–4 grams

---

### 3. Multiple Candidate Evaluation

The system allows:
* Comparing multiple model outputs
* Evaluating beam search variations
* Comparing baseline vs fine-tuned models

Each candidate's translation receives:
* Individual BLEU score
* Individual n-gram precision breakdown

---
