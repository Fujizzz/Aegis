"""Standalone inference from an explicit best/last checkpoint directory."""
import argparse
import json
from pathlib import Path
import yaml
from inference.predictor import Predictor
from utils.checkpoints import checkpoint_config, checkpoint_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', required=True, help='Exact run/best or run/last directory')
    ap.add_argument('--config', help='Optional compatible config; defaults to checkpoint snapshot')
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--text', help='Article body')
    inputs.add_argument('--input_json', help='UTF-8 JSON news object')
    ap.add_argument('--title')
    ap.add_argument('--subject')
    ap.add_argument('--date')
    ap.add_argument('--device', help='Override all devices, e.g. cpu or cuda:0')
    ap.add_argument('--output', help='Save prediction as UTF-8 JSON')
    args = ap.parse_args()
    if args.input_json and any(v is not None for v in (args.title, args.subject, args.date)):
        ap.error('Metadata flags require --text')
    cfg = checkpoint_config(args.checkpoint_dir)
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8-sig'))
    if args.device:
        cfg['hardware']['device_map'] = {key: args.device for key in cfg['hardware']['device_map']}
    if args.input_json:
        news = json.loads(Path(args.input_json).read_text(encoding='utf-8-sig'))
        if not isinstance(news, dict):
            ap.error('--input_json must contain one news object')
    else:
        news = dict(text=args.text, title=args.title, subject=args.subject, date=args.date)
    result = Predictor(cfg, checkpoint_paths(args.checkpoint_dir)).predict(news)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + '\n', encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
