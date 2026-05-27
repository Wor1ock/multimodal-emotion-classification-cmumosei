import os
import re
import nltk
import spacy
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

RE_SUFFIX = re.compile(r"\b(\d+)-[a-z]+\b")
RE_MIXED = re.compile(r"\b\d+[a-z]+\b")
RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?")

custom_exceptions = {
    "not", "no", "never", "nor", "neither",
    "but", "however", "although", "very",
    "too", "extremely", "again", "still",
    "only", "just", "against", "up", "down"
}
stop_words = set(stopwords.words("english")) - custom_exceptions
