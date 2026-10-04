"""Download a pinned model snapshot or fingerprint an existing local snapshot."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Aegis'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Existing local Hugging Face snapshot')
    parser.add_argument('--download', action='store_true', help='Download the pinned Qwen snapshot')
    parser.add_argument('--output', type=Path, default=ROOT/'models/Qwen2.5-7B')
    parser.add_argument('--manifest', type=Path, default=ROOT/'models/model_manifest.json')
    args = parser.parse_args()
    if args.source and args.download:
        parser.error('Choose --source or --download')
    if not (args.source or args.download):
        print(json.dumps(dict(model='Qwen/Qwen2.5-7B', revision='d149729398750b98c0af14eb82c78cfe92750796',
            output=str(args.output), status='planned')))
        return
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id='Qwen/Qwen2.5-7B', revision='d149729398750b98c0af14eb82c78cfe92750796',
                          local_dir=str(args.output))
    from utils.run_archive import model_manifest, write_json
    manifest = model_manifest(args.source or args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        if json.loads(args.manifest.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Existing model manifest differs; choose a new --manifest')
    else:
        write_json(args.manifest, manifest)
    print(json.dumps(dict(model_root=str((args.source or args.output).resolve()),
                         manifest=str(args.manifest), identity=manifest['identity']), indent=2))


if __name__ == '__main__':
    main()
