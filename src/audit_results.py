"""실행 결과의 수치와 검증 임계값을 대조하고 원본 계열 중복을 점검한다."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    root = args.results
    env = json.loads((root / 'environment.json').read_text())
    selected = json.loads((root / 'selected_thresholds.json').read_text())
    table = pd.read_csv(root / 'test_onoff_results.csv')
    scores = pd.read_csv(root / 'image_onoff_scores.csv')
    validation = pd.read_csv(root / 'val_threshold_scores.csv')
    assert selected['selected_on'] == 'val'
    for filename, key in [('split_manifest.csv', 'manifest_sha256'),
                          ('evaluation_manifest.csv', 'evaluation_manifest_sha256')]:
        digest = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        assert digest == env[key] == selected[key], filename
    for _, row in table.iterrows():
        name = row['class']
        candidates = validation[validation['class'] == name]
        best_f1 = candidates['f1'].max()
        threshold = candidates.loc[candidates['f1'] == best_f1, 'threshold'].min()
        assert threshold == selected['thresholds'][name] == row['threshold']
        images = scores[(scores['split'] == 'test') & (scores['class'] == name)]
        actual = {
            'TP': (images['gt_on'] & images['pred_on']).sum(),
            'FP': (~images['gt_on'] & images['pred_on']).sum(),
            'FN': (images['gt_on'] & ~images['pred_on']).sum(),
            'TN': (~images['gt_on'] & ~images['pred_on']).sum(),
        }
        assert all(row[k] == actual[k] for k in actual), name
        assert len(images) == sum(actual.values()) == env['test_images']
    print('Validation thresholds and test ON/OFF counts verified.')
    print('Test macro F1:', round(float(table['f1'].mean()), 6))

    manifest = pd.read_csv(root / 'split_manifest.csv')
    pattern = r'^([A-Z]+\d+_\d{4}_\d{2}_\d{2}_\d{3}_\d+)(?:_|$)'
    def family(key):
        stem = Path(key).name
        match = re.match(pattern, stem)
        return match.group(1) if match else stem
    manifest['source_family'] = manifest['source_key'].map(family)
    counts = manifest.groupby('source_family')['split'].nunique()
    overlap = manifest[manifest['source_family'].isin(counts[counts > 1].index)]
    summary = overlap.groupby(['source_family', 'split']).agg(
        file_count=('source_key', 'size'), example_source_key=('source_key', 'first')
    ).reset_index()
    summary.to_csv(root / 'source_family_overlap.csv', index=False, encoding='utf-8-sig')
    print('Source families shared across splits:', int((counts > 1).sum()))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
