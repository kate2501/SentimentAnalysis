from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
from sklearn.base import BaseEstimator, TransformerMixin

morph = pymorphy2.MorphAnalyzer()
stop_words = stopwords.words('russian')
stop_words.remove('не')


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.str.replace('http\S+|www.\S+|\d+|[^\w\s]', ' ')
        X = X.apply(lambda text: list(self.preprocess(text)))
        X = X.apply(lambda text: " ".join(text))
        return X

    @staticmethod
    def preprocess(text, stop_words=stop_words, lemmatizer=morph):
        tokens = word_tokenize(text.lower())
        clean_tokens = filter(lambda tok: not (tok in stop_words), tokens)
        lemmatized = list(map(lambda x: lemmatizer.parse(x)[0].normal_form, clean_tokens))
        return lemmatized

