"""Use the neural experiment input/probability/archive contracts for baselines."""
from pathlib import Path
from utils.freeze_pheme import read_csv
from utils.label_schema import schema_from_config
from utils.predictions import prediction_rows, write_predictions, sample_identity
from utils.run_archive import sha256, write_json
from utils.text_input import build_news_text
from evaluate import evaluate_files, write_report


def split_data(config, split):
    path = Path(config['data'][split + '_file'])
    rows = read_csv(path)
    if not rows:
        raise ValueError('Empty baseline split')
    schema = schema_from_config(config)
    return rows, [build_news_text(r) for r in rows], [schema.encode_source(r['label']) for r in rows]


def export(config, split, probabilities, checkpoint):
    rows, _, targets = split_data(config, split)
    source = config['data'][split + '_file']
    source_sha = sha256(source)
    identities = [dict(sample_id=sample_identity(r, i, source_sha)[0],
                       identity_source='source_sample_id', event=r.get('event_name', config['experiment']['event']))
                  for i,r in enumerate(rows)]
    predictions = prediction_rows(identities, targets, probabilities, schema_from_config(config), config,
                                  split, 'sha256:' + sha256(checkpoint))
    root = Path(config['runtime']['save_dir'])
    path = root / (split + '_predictions.csv')
    if path.exists():
        if not path.with_suffix('.manifest.json').is_file():
            raise ValueError('Existing prediction CSV is missing its provenance manifest')
        expected = [{k:str(v) for k,v in row.items()} for row in predictions]
        if read_csv(path) != expected:
            raise ValueError('Existing baseline predictions differ from the recovered response/model')
    else:
        write_predictions(path, predictions, checkpoint, source)
    report = evaluate_files([path])
    report_dir = root / (split + '_evaluation')
    if not report_dir.exists():
        write_report(report, report_dir)
    else:
        import json
        if json.loads((report_dir/'metrics.json').read_text(encoding='utf-8')) != report:
            raise ValueError('Existing baseline evaluation differs from predictions')
    write_json(root / (split + '_metrics.json'), report['sample_pooled'])
    return report['sample_pooled']


def run_text(config):
    from baselines.tfidf_nb import TfidfNB
    import time
    root = Path(config['runtime']['save_dir'])
    _, train_texts, targets = split_data(config, 'train')
    start = time.perf_counter()
    model = TfidfNB(**config.get('baseline', {}).get('tfidf', {})).fit(train_texts, targets)
    fit_seconds = time.perf_counter() - start
    checkpoint = root / 'best'
    checkpoint.mkdir()
    write_json(checkpoint / 'model.json', model.state())
    write_json(checkpoint / 'checkpoint_manifest.json', dict(format='tfidf_multinomial_nb_v1',
        sha256={'model.json':sha256(checkpoint / 'model.json')}, run_id=config['runtime']['run_id'],
        selection='Fixed hyperparameters; fitted on training texts/labels only; no validation/test tuning'))
    # Predictions use the serialized representation, not an unarchived in-memory model.
    import json
    restored = TfidfNB.restore(json.loads((checkpoint / 'model.json').read_text(encoding='utf-8')))
    metrics = {}
    for split in ('val', 'test'):
        if config['data'].get(split + '_file'):
            _, texts, _ = split_data(config, split)
            metrics[split] = export(config, split, restored.predict(texts), checkpoint / 'checkpoint_manifest.json')
    write_json(root / 'baseline_cost.json', dict(fit_seconds=fit_seconds, vocabulary_size=len(model.idf),
        input_contract='news_fields_v1; word unigrams/bigrams, no Qwen token truncation', pretrained_model=False))
    return metrics
