"""Deterministic train-only TF-IDF + multinomial NB, no extra dependencies."""
from collections import Counter
import math
import re


def terms(text, ngram_max=2):
    words = re.findall(r"\b\w+\b", text.lower())
    return words + ([' '.join(pair) for pair in zip(words, words[1:])] if ngram_max == 2 else [])


class TfidfNB:
    def __init__(self, alpha=1.0, min_df=2, max_features=50000, ngram_max=2):
        if not math.isfinite(alpha) or alpha <= 0 or type(min_df) is not int or min_df < 1:
            raise ValueError('Positive alpha and integer min_df are required')
        if type(max_features) is not int or max_features < 1 or ngram_max not in (1, 2):
            raise ValueError('Invalid vocabulary/gram bounds')
        self.params = dict(alpha=alpha, min_df=min_df, max_features=max_features, ngram_max=ngram_max)

    def vector(self, text):
        counts = Counter(terms(text, self.params['ngram_max']))
        values = {term:(1 + math.log(count)) * self.idf[term]
                  for term,count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in values.values()))
        return {term:value / norm for term,value in values.items()} if norm else {}

    def fit(self, texts, targets):
        if len(texts) != len(targets) or not texts or set(targets) != {0, 1}:
            raise ValueError('Training requires aligned texts and both canonical classes')
        documents = Counter(term for text in texts for term in set(terms(text, self.params['ngram_max'])))
        vocabulary = sorted((term for term,df in documents.items() if df >= self.params['min_df']),
                            key=lambda term:(-documents[term], term))[:self.params['max_features']]
        if not vocabulary:
            raise ValueError('Training vocabulary is empty')
        self.idf = {term:math.log((1 + len(texts))/(1 + documents[term])) + 1 for term in vocabulary}
        counts = [Counter(), Counter()]
        for text, target in zip(texts, targets):
            counts[target].update(self.vector(text))
        alpha = self.params['alpha']
        class_counts = Counter(targets)
        self.log_prior = [math.log(class_counts[c]/len(targets)) for c in (0, 1)]
        self.log_probability = []
        for c in (0, 1):
            total = sum(counts[c].values()) + alpha * len(vocabulary)
            self.log_probability.append({term:math.log((counts[c][term] + alpha)/total) for term in vocabulary})
        return self

    def predict(self, texts):
        result = []
        for text in texts:
            vector = self.vector(text)
            scores = [self.log_prior[c] + sum(value*self.log_probability[c][term] for term,value in vector.items())
                      for c in (0, 1)]
            delta = scores[1] - scores[0]
            result.append(1/(1+math.exp(-delta)) if delta >= 0 else math.exp(delta)/(1+math.exp(delta)))
        return result

    def state(self):
        return dict(format='tfidf_multinomial_nb_v1', params=self.params, idf=self.idf,
                    log_prior=self.log_prior, log_probability=self.log_probability)

    @classmethod
    def restore(cls, state):
        if state['format'] != 'tfidf_multinomial_nb_v1':
            raise ValueError('Unknown text baseline format')
        model = cls(**state['params'])
        model.idf, model.log_prior, model.log_probability = state['idf'], state['log_prior'], state['log_probability']
        if len(model.log_prior) != 2 or len(model.log_probability) != 2:
            raise ValueError('Invalid class model')
        for mapping in model.log_probability:
            if set(mapping) != set(model.idf) or not all(math.isfinite(v) for v in mapping.values()):
                raise ValueError('Invalid vocabulary probabilities')
        return model
