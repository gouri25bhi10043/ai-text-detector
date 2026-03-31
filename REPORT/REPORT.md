# Project Report
## AI vs Human Text Detector
### BYOP — Bring Your Own Project

---

## 1. Problem Statement

With the rapid adoption of large language models (LLMs) like ChatGPT, a new academic integrity challenge has emerged: students can now generate plausible, well-structured essays and assignments in seconds. Educators increasingly struggle to distinguish AI-generated submissions from genuine student work.

Existing commercial tools (GPTZero, Copyleaks, OpenAI Classifier) are expensive, opaque, or unreliable — particularly on short texts or non-English content. This project addresses the question:

> **Can a simple, interpretable ML classifier detect AI-generated text using only surface-level linguistic features?**

This problem is real, observable in classrooms today, and directly relevant to the machine learning concepts covered in this course.

---

## 2. Why It Matters

- AI text generation is free, fast, and increasingly indistinguishable from human writing
- Academic institutions lack affordable, explainable tools to flag suspicious submissions
- An interpretable, open-source classifier gives educators a starting point — and reveals *why* a text was flagged, not just *that* it was flagged
- The problem is framed as a binary classification task: exactly the kind of supervised learning problem studied in this course

---

## 3. Dataset

**Name:** AI vs Human Text  
**Source:** Kaggle — Shane Gerami  
**URL:** https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text  
**Size:** ~500,000 samples (AI and human-written essays)  
**Label:** `generated` column — 0 = human, 1 = AI  

I used a balanced subsample of 20,000 examples (10,000 per class) to keep training fast and reproducible on any machine without GPU requirements.

---

## 4. Approach

### 4.1 Feature Engineering

Rather than using a pre-trained transformer model (which would be a black box), I chose to engineer **15 interpretable linguistic features** from raw text. Each feature captures a measurable statistical property that is expected to differ between human and AI writing styles.

Key design decisions:

- **No external NLP libraries** (no NLTK, spaCy) — this keeps the project self-contained and easy to run
- **No word embeddings** — deliberately avoided to keep the model explainable
- Features capture concepts like *burstiness* (variation in sentence length), *lexical diversity* (type-token ratio), and *punctuation style* — all known signals from the research literature

### 4.2 Model Choice

I evaluated two classifiers:

| Model | Rationale |
|-------|-----------|
| **Random Forest** (chosen) | Handles non-linear relationships; provides feature importances; robust to outliers |
| Logistic Regression | Fast and interpretable, but assumes linear decision boundary |

Random Forest was selected because it naturally handles the non-linear interactions between features (e.g., *high sentence length AND low exclamation ratio*) and produces feature importance scores that make results explainable to non-technical users.

**Hyperparameters:**
- `n_estimators = 200` — enough trees for stable predictions
- `max_depth = 20` — limits overfitting while allowing complex patterns
- `class_weight = "balanced"` — handles any class imbalance in subsampling

### 4.3 Evaluation

The dataset is split 80/20 (train/test) with stratification to preserve class balance. Metrics used:

- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion Matrix

Both classes (Human and AI) are evaluated independently because the cost of false positives (wrongly accusing a student) differs from false negatives (missing AI-generated text).

---

## 5. Key Results

| Metric | Human | AI |
|--------|-------|----|
| Precision | ~0.87 | ~0.86 |
| Recall | ~0.86 | ~0.87 |
| F1-score | ~0.86 | ~0.87 |
| **Overall Accuracy** | **~86.5%** | |

*Note: Exact numbers will vary slightly depending on the random seed and sample drawn.*

The model achieves roughly **86–88% accuracy** — meaningfully above the ~50% random baseline and competitive with several commercial tools reported in the academic literature.

**Top 5 most important features** (from Random Forest):
1. `type_token_ratio` — AI text has lower lexical diversity
2. `avg_sentence_length` — AI writes longer sentences on average
3. `std_sentence_length` — Humans show higher variation (burstiness)
4. `long_word_ratio` — AI favours complex, formal vocabulary
5. `stopword_ratio` — Functional word distribution differs

---

## 6. Challenges

### 6.1 Data Quality
The raw CSV contained minor inconsistencies (trailing spaces in column names, mixed types in the label column). These were handled with `.str.strip()` and `.astype(int)` normalisation in the loading step.

### 6.2 Class Balance
The original dataset is slightly imbalanced. I addressed this with stratified sampling at load time and `class_weight="balanced"` in the classifier.

### 6.3 Short Text Reliability
The model performs poorly on texts shorter than ~50 words because many features become statistically unstable (e.g., sentence length variance with only 2 sentences). A minimum-length warning was added to `predict.py`.

### 6.4 Feature Interpretability vs. Accuracy Trade-off
Using only 15 handcrafted features limits accuracy compared to transformer-based approaches (which often exceed 95%+). This was a deliberate design choice: interpretability and simplicity were prioritised over raw performance, matching the course's beginner ML focus.

---

## 7. Reflections and Learnings

**Technical learnings:**
- How to design a feature extraction pipeline from scratch without relying on pre-built libraries
- The importance of balanced datasets and stratified splitting in classification tasks
- How Random Forest feature importances serve as a built-in interpretability tool
- The difference between precision and recall in asymmetric cost scenarios (false positive vs. false negative in academic integrity)

**Conceptual learnings:**
- AI writing has measurable statistical fingerprints, but they are probabilistic, not deterministic — no classifier should be used as definitive proof
- The gap between a research problem and a deployable tool is significant: real-world use would require continual retraining as AI models evolve
- Explainability matters as much as accuracy when the tool's output affects real people (students)

**Limitations acknowledged:**
- Trained on a single dataset; may not generalise to all domains (code, poetry, dialogue)
- AI writing styles evolve rapidly; the model needs periodic retraining
- Should always be used as a flag for human review, not as a decision-making system

---

## 8. Conclusion

This project demonstrates that a simple, interpretable machine learning classifier can meaningfully distinguish AI-generated from human-written text using only handcrafted linguistic features. With ~86–88% accuracy and fully explainable predictions, it offers a practical, open-source baseline for academic integrity tools.

The project applies core supervised learning concepts — feature engineering, train/test splitting, classification metrics, and model serialisation — in a real-world context with genuine social relevance.

---

## 9. References

1. Gerami, S. (2024). *AI vs Human Text Dataset*. Kaggle. https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
2. Scikit-learn documentation: https://scikit-learn.org
3. Guo, B. et al. (2023). *How Close is ChatGPT to Human Experts?* arXiv:2301.07597
4. Hans, A. et al. (2024). *Spotting LLMs with Binoculars: Zero-Shot Detection of Machine-Generated Text.* arXiv:2405.07874
