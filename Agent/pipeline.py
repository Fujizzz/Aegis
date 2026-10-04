"""Two-stage results: probability and confidence never share one field."""
from probability import probability


def detect_item(item, first_layer, second_layer, cap=0.9):
    cap = probability(cap)
    stage = 'screening'
    args = {key: item.get(key, '') for key in ('title', 'content', 'subject', 'date')}
    try:
        p, label = first_layer.analyze_news(**args)
        p = probability(p)
        if label not in ('real', 'fake', 'uncertain'):
            raise ValueError('Invalid screening label')
        explanation = 'First-layer specialist consensus.'
        if label == 'uncertain' or (label == 'fake' and p < cap):
            stage = 'verification'
            explanation, label, p = second_layer.analyze_news(**args)
            p = probability(p)
        if label not in ('fake', 'real') or (label == 'fake') != (p >= .5):
            raise ValueError('Label and p_fake disagree')
        return dict(predicted_label=label, p_fake=p, class_confidence=p if label == 'fake' else 1-p,
                    stage=stage, status='success', error_type=None, explanation=explanation)
    except Exception as error:
        return dict(predicted_label='abstain', p_fake=None, class_confidence=None,
                    stage=stage, status='error', error_type=type(error).__name__, explanation='')
