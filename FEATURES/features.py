"""
features.py — Handcrafted linguistic features for AI vs Human text detection
=============================================================================
All features are language-agnostic statistics that do NOT rely on heavy NLP
libraries, making this runnable with just  scikit-learn + numpy.

Features extracted (15 total):
  1.  avg_word_length        — AI tends toward longer, formal vocabulary
  2.  avg_sentence_length    — AI produces more uniform sentence lengths
  3.  std_sentence_length    — humans have higher burstiness
  4.  type_token_ratio       — lexical diversity (unique / total words)
  5.  punctuation_ratio      — AI uses punctuation differently
  6.  comma_ratio            — AI over-uses commas in lists
  7.  exclamation_ratio      — humans use more exclamations
  8.  question_ratio         — relative question frequency
  9.  uppercase_ratio        — ALL CAPS words (emotion, emphasis)
  10. digit_ratio            — AI inserts statistics more often
  11. avg_paragraph_length   — AI paragraphs are more uniform
  12. stopword_ratio         — functional word density
  13. long_word_ratio        — words > 6 chars
  14. sentence_count         — total number of sentences
  15. whitespace_ratio       — extra spaces / formatting artefacts
"""

import re
import string
from typing import List


# ──────────────────────────────────────────────────────────
# A small set of common English stopwords (no NLTK needed)
# ──────────────────────────────────────────────────────────
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they",
    "them", "their", "theirs", "themselves", "what", "which", "who",
    "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now", "s", "t", "re", "ve", "ll", "d", "m",
}


def _sentences(text: str) -> List[str]:
    """Split text into sentences on  .  !  ?"""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _words(text: str) -> List[str]:
    """Tokenise into lowercase words (letters only)."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def extract_features(text: str) -> List[float]:
    """Return a 1-D list of 15 numeric features for *text*."""
    if not text or not text.strip():
        return [0.0] * 15

    words      = _words(text)
    sentences  = _sentences(text)
    chars      = list(text)
    n_words    = max(len(words), 1)
    n_chars    = max(len(chars), 1)
    n_sent     = max(len(sentences), 1)

    # 1. avg_word_length
    avg_word_len = sum(len(w) for w in words) / n_words

    # 2 & 3. avg / std sentence length (in words)
    sent_lens = [len(_words(s)) for s in sentences]
    avg_sent_len = sum(sent_lens) / n_sent
    variance     = sum((l - avg_sent_len) ** 2 for l in sent_lens) / n_sent
    std_sent_len = variance ** 0.5

    # 4. type-token ratio  (unique words / total words)
    ttr = len(set(words)) / n_words

    # 5–8. punctuation ratios
    punct_count = sum(1 for c in text if c in string.punctuation)
    comma_count = text.count(",")
    excl_count  = text.count("!")
    quest_count = text.count("?")

    punct_ratio = punct_count / n_chars
    comma_ratio = comma_count / n_words
    excl_ratio  = excl_count  / n_words
    quest_ratio = quest_count / n_words

    # 9. uppercase word ratio
    raw_words   = text.split()
    upper_count = sum(1 for w in raw_words if w.isupper() and len(w) > 1)
    upper_ratio = upper_count / max(len(raw_words), 1)

    # 10. digit ratio
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / n_chars

    # 11. avg paragraph length (words per paragraph)
    paragraphs  = [p.strip() for p in text.split("\n\n") if p.strip()]
    n_para      = max(len(paragraphs), 1)
    para_lens   = [len(_words(p)) for p in paragraphs]
    avg_para_len = sum(para_lens) / n_para

    # 12. stopword ratio
    stop_count  = sum(1 for w in words if w in STOPWORDS)
    stop_ratio  = stop_count / n_words

    # 13. long word ratio  (words > 6 chars)
    long_count  = sum(1 for w in words if len(w) > 6)
    long_ratio  = long_count / n_words

    # 14. sentence count (raw)
    sent_count  = float(n_sent)

    # 15. extra whitespace ratio
    ws_count    = sum(1 for c in text if c in " \t\n")
    ws_ratio    = ws_count / n_chars

    return [
        avg_word_len,    # 1
        avg_sent_len,    # 2
        std_sent_len,    # 3
        ttr,             # 4
        punct_ratio,     # 5
        comma_ratio,     # 6
        excl_ratio,      # 7
        quest_ratio,     # 8
        upper_ratio,     # 9
        digit_ratio,     # 10
        avg_para_len,    # 11
        stop_ratio,      # 12
        long_ratio,      # 13
        sent_count,      # 14
        ws_ratio,        # 15
    ]


FEATURE_NAMES = [
    "avg_word_length",
    "avg_sentence_length",
    "std_sentence_length",
    "type_token_ratio",
    "punctuation_ratio",
    "comma_ratio",
    "exclamation_ratio",
    "question_ratio",
    "uppercase_ratio",
    "digit_ratio",
    "avg_paragraph_length",
    "stopword_ratio",
    "long_word_ratio",
    "sentence_count",
    "whitespace_ratio",
]
