"""Score two completed blinded reviewer CSV files without external packages."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ANSWER_VALUES = {'yes', 'no', 'unclear'}
RELATION_VALUES = {'same', 'opposite', 'neither', 'unclear'}
ACCEPTABLE_VALUES = {'yes', 'no', 'unclear'}


def load_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return {row['uid']: {k: v.strip().lower() for k, v in row.items()} for row in rows}


def load_mapping(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return {(x['reviewer'], x['uid']): x for x in payload['mapping']}


def kappa(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('Kappa vectors must be non-empty and equal length.')
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[x] / len(a)) * (cb[x] / len(b)) for x in labels)
    value = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)
    return {'n': len(a), 'agreement': observed, 'expected_agreement': expected, 'cohen_kappa': value}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviewer-1', required=True)
    parser.add_argument('--reviewer-2', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    reviewers = {
        'reviewer_1': load_csv(args.reviewer_1),
        'reviewer_2': load_csv(args.reviewer_2),
    }
    mapping = load_mapping(args.mapping)
    uids = sorted(reviewers['reviewer_1'])
    if len(uids) != 81 or set(uids) != set(reviewers['reviewer_2']):
        raise ValueError('Both reviewer files must contain the same 81 unique uids.')

    normalized = {}
    for reviewer, rows in reviewers.items():
        for uid in uids:
            row = rows[uid]
            for field in ('answer_original', 'answer_version_1', 'answer_version_2'):
                if row[field] not in ANSWER_VALUES:
                    raise ValueError(f'{reviewer} {uid}: invalid or blank {field}={row[field]!r}')
            for field in ('relation_version_1_to_original', 'relation_version_2_to_original'):
                if row[field] not in RELATION_VALUES:
                    raise ValueError(f'{reviewer} {uid}: invalid or blank {field}={row[field]!r}')
            if row['item_acceptable'] not in ACCEPTABLE_VALUES:
                raise ValueError(f'{reviewer} {uid}: invalid or blank item_acceptable')

            m = mapping[(reviewer, uid)]
            arm_to_version = {m['version_1_arm']: 'version_1', m['version_2_arm']: 'version_2'}
            normalized[(reviewer, uid)] = {
                'answer_o': row['answer_original'],
                'answer_a': row[f"answer_{arm_to_version['a']}"],
                'answer_b': row[f"answer_{arm_to_version['b']}"],
                'relation_a': row[f"relation_{arm_to_version['a']}_to_original"],
                'relation_b': row[f"relation_{arm_to_version['b']}_to_original"],
                'item_acceptable': row['item_acceptable'],
                'notes': row['notes'],
                'gold_o': m['gold_o'],
                'gold_a': m['gold_a'],
                'gold_b': m['gold_b'],
            }

    fields = ['answer_o', 'answer_a', 'answer_b', 'relation_a', 'relation_b', 'item_acceptable']
    agreement = {}
    for field in fields:
        a = [normalized[('reviewer_1', uid)][field] for uid in uids]
        b = [normalized[('reviewer_2', uid)][field] for uid in uids]
        agreement[field] = kappa(a, b)

    item_rows = []
    adjudication = []
    valid_by_reviewer = Counter()
    for uid in uids:
        r1 = normalized[('reviewer_1', uid)]
        r2 = normalized[('reviewer_2', uid)]
        row = {'uid': uid}
        disagreements = []
        for field in fields:
            row[f'r1_{field}'] = r1[field]
            row[f'r2_{field}'] = r2[field]
            if r1[field] != r2[field]:
                disagreements.append(field)

        for reviewer, values in [('reviewer_1', r1), ('reviewer_2', r2)]:
            valid = (
                values['answer_o'] == values['gold_o']
                and values['answer_a'] == values['gold_a']
                and values['answer_b'] == values['gold_b']
                and values['relation_a'] == 'same'
                and values['relation_b'] == 'opposite'
                and values['item_acceptable'] == 'yes'
            )
            row[f'{reviewer}_expected_pattern_valid'] = valid
            valid_by_reviewer[reviewer] += int(valid)
        row['both_reviewers_expected_pattern_valid'] = bool(
            row['reviewer_1_expected_pattern_valid'] and row['reviewer_2_expected_pattern_valid']
        )
        row['disagreement_fields'] = ';'.join(disagreements)
        item_rows.append(row)
        if disagreements or not row['both_reviewers_expected_pattern_valid']:
            adjudication.append(row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'human_validation_item_results.csv', item_rows)
    write_csv(out / 'adjudication_required.csv', adjudication)
    summary = {
        'n_items': len(uids),
        'independent_pre_adjudication_agreement': agreement,
        'reviewer_1_expected_pattern_valid_items': valid_by_reviewer['reviewer_1'],
        'reviewer_2_expected_pattern_valid_items': valid_by_reviewer['reviewer_2'],
        'both_reviewers_expected_pattern_valid_items': sum(x['both_reviewers_expected_pattern_valid'] for x in item_rows),
        'items_requiring_adjudication': len(adjudication),
        'note': 'Cohen kappa is computed before adjudication. Final consensus validity must be added only after human adjudication.',
    }
    (out / 'human_validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = [
        '| Field | Agreement (%) | Cohen’s κ |',
        '|---|---:|---:|',
    ]
    for field in fields:
        lines.append(f"| {field} | {100 * agreement[field]['agreement']:.1f} | {agreement[field]['cohen_kappa']:.3f} |")
    lines.extend([
        '',
        f"Expected pattern valid for both reviewers: {summary['both_reviewers_expected_pattern_valid_items']}/81",
        f"Items requiring adjudication: {summary['items_requiring_adjudication']}/81",
    ])
    (out / 'human_validation_markdown_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
