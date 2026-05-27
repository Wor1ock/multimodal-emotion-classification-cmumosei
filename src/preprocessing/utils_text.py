import re
import spacy
from nltk.corpus import stopwords

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
