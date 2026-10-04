"""The input contract shared by CSV training and standalone prediction."""
from collections.abc import Mapping
import pandas as pd

INPUT_FORMAT = 'news_fields_v1'


def visible_text(value):
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def build_news_text(news):
    """Only visible fields; text takes precedence over content when nonblank.

    A string is a body-only article. Missing metadata is optional. Literal words
    such as 'nan' are preserved; only actual null/NaN values are omitted.
    """
    if isinstance(news, str):
        news = {'text': news}
    if not isinstance(news, (Mapping, pd.Series)):
        raise TypeError('News must be a body string or a mapping of news fields')
    parts = []
    for field in ('title', 'subject', 'date'):
        value = visible_text(news.get(field))
        if value:
            parts.append(f'[{field.upper()}] {value}')
    body = visible_text(news.get('text')) or visible_text(news.get('content'))
    if body:
        parts.append(f'[CONTENT]\n{body}')
    if not parts:
        raise ValueError('News must contain at least one nonblank visible field')
    return '\n'.join(parts)


def tokenize_texts(tokenizer, texts, max_length):
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError('max_length must be a positive integer')
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    return tokenizer(texts, padding=True, truncation=True,
                     max_length=max_length, return_tensors='pt')
