import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')
cv = CountVectorizer()

def clean_text(texts):
    normalized_texts = []

    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)

        cv.fit(normalized_texts)
        cleaned_texts = cv.transform(normalized_texts)              

    return cleaned_texts