"""Build the deduplicated binary dataset from local JSONL files."""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(raw_dir, output_dir, seed=42):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    groups = defaultdict(list)
    hashes = {}
    seen_ids = set()
    for source in ('vihsd', 'victsd'):
        for split in ('train', 'validation', 'test'):
            path = raw_dir / f'{source}_{split}.jsonl'
            hashes[path.name] = sha(path)
            with path.open(encoding='utf-8') as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    text = row['text'] if source == 'vihsd' else row['Comment']
                    label = row['label'] if source == 'vihsd' else row['Toxicity']
                    if not isinstance(text, str) or not text.strip() or type(label) is not int or label not in ((0, 1, 2) if source == 'vihsd' else (0, 1)):
                        raise ValueError(f'Invalid record in {path}:{index + 1}')
                    origin = f"{source}:{row['id']}" if source == 'vihsd' else f"{source}:{split}:{row['_idx']}"
                    if origin in seen_ids:
                        raise ValueError(f'Duplicate source ID: {origin}')
                    seen_ids.add(origin)
                    groups[text].append((origin, source, int(label > 0)))
    records, manifest = [], {}
    conflicts = 0
    for text, origins in groups.items():
        if len({entry[2] for entry in origins}) > 1:
            conflicts += 1
            for origin, _, _ in origins:
                manifest[origin] = {'id': None, 'split': None, 'reason': 'conflicting_labels'}
            continue
        origins.sort()
        origin, source, label = origins[0]
        record_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
        records.append({'id': record_id, 'text': text, 'label': label, 'source': source})
        for original, _, _ in origins:
            manifest[original] = {'id': record_id, 'split': None}
    if conflicts != 205:
        raise ValueError(f'Expected 205 conflicting groups; found {conflicts}')
    records.sort(key=lambda row: row['id'])
    strata = [f"{r['source']}:{r['label']}" for r in records]
    normalized_keys = [hashlib.sha256((re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', r['text']).casefold()) or r['id']).encode('utf-8')).hexdigest() for r in records]
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(folds.split(records, strata, groups=normalized_keys)):
        split = 'train' if fold < 14 else 'val' if fold < 17 else 'test'
        for i in indices:
            records[i]['split'] = split
    assignments = {row['id']: row['split'] for row in records}
    for entry in manifest.values():
        if entry['id'] is not None:
            entry['split'] = assignments[entry['id']]
    normalized = defaultdict(set)
    for row in records:
        key = re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', row['text']).casefold()) or row['id']
        normalized[key].add(row['split'])
    near_overlap = sum(len(splits) > 1 for splits in normalized.values())
    output_dir.mkdir(parents=True)
    for split in ('train', 'val', 'test'):
        pd.DataFrame([r for r in records if r['split'] == split]).to_parquet(output_dir / f'{split}.parquet', index=False)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')
    counts = Counter((r['split'], r['source'], r['label']) for r in records)
    info = {'seed': seed, 'raw_sha256': hashes, 'conflicting_groups': conflicts,
            'excluded_records': sum(entry['id'] is None for entry in manifest.values()),
            'counts': [{'split': s, 'source': o, 'label': l, 'count': n} for (s, o, l), n in sorted(counts.items())],
            'normalized_cross_split_groups': near_overlap,
            'near_duplicate_method': 'NFKC, casefold, remove non-word characters; not a general near-duplicate search',
            'test_blind': False,
            'output_sha256': {p.name: sha(p) for p in sorted(output_dir.iterdir())}}
    (output_dir / 'dataset_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output-dir', default='data/processed/binary_n2_v1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.raw_dir, args.output_dir, args.seed), ensure_ascii=False, indent=2))
