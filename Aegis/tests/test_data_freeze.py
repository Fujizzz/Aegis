"""Leakage, identity, pseudo lineage and immutable-release regression checks."""
import copy
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import freeze_pheme as f


def row(i, text, event='charliehebdo', label='0'):
    sid = str(550000000000000000 + i)
    return dict(sample_id='pheme:' + sid, id=sid, thread_id=sid, reply_to_id='',
                event_name=event, label=label, title='', subject='', date='', text=text,
                source_member='source/' + sid, source_json_sha256=f.digest(sid),
                text_sha256=f.digest(text), normalized_text_sha256=f.digest(f.normalize(text)))


class DataFreezeTests(unittest.TestCase):
    def test_identity_never_uses_label_to_resolve_ambiguous_text(self):
        rows = [row(1, 'same text', label='0'), row(2, 'same text', label='1')]
        method, matches = f.Matcher(rows).match('charliehebdo', ' same\ntext ')
        self.assertEqual(method, 'normalized_exact')
        self.assertEqual(len(matches), 2)
        self.assertEqual(f.Matcher(rows).match('sydneysiege', 'same text')[1], [])

    def test_whitespace_repair_is_distinguished_from_exact_pseudo_reuse(self):
        matcher = f.Matcher([row(1, 'one two\nthree')])
        self.assertEqual(matcher.match('charliehebdo', 'onetwothree')[0], 'whitespace_repaired')
        self.assertEqual(matcher.match('charliehebdo', 'one two three')[0], 'normalized_exact')
        self.assertEqual(matcher.match('charliehebdo', '')[0], 'empty_text')

    def test_historical_bad_ids_invalid_rows_and_text_repair_are_accounted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'old.csv'
            f.write_csv(path, [dict(id='5.5e17', label='0', content='first text'),
                               dict(id='5.5e17', label='1', content='secondtext'),
                               dict(id='5.5e17', label='date', content='bad text')], ['id', 'label', 'content'])
            official = [row(1, 'first text'), row(2, 'second text', label='1')]
            with patch.object(f, 'source_files', return_value=[('charliehebdo', 'raw', path)]):
                mappings, _, inputs = f.recover_history(root, official)
            self.assertEqual([m['status'] for m in mappings], ['recovered', 'recovered', 'invalid_legacy_label'])
            self.assertNotEqual(mappings[0]['sample_id'], mappings[1]['sample_id'])
            self.assertEqual(mappings[2]['sample_id'], '')
            self.assertTrue(mappings[2]['fallback_content_id'].startswith('text:'))
            self.assertEqual(inputs[0]['sha256'], f.file_hash(path))

    def test_near_search_matches_exhaustive_jaccard_reference(self):
        texts = ['Breaking: a long factual report about an event in central London today',
                 'RT @someone: Breaking: a long factual report about an event in central London today https://t.co/abc',
                 'Breaking: a long factual report about an event in central London tonight',
                 'Completely different weather report from another country', 'short', '',
                 'Breaking: a long factual report about an event in central London today!']
        rows = [row(i, t) for i, t in enumerate(texts)]
        grams = []
        for text in texts:
            s = f.skeleton(text)
            grams.append({s[k:k+5] for k in range(len(s)-4)} if len(s) >= 20 else set())
        expected = set()
        for a, b in itertools.combinations(range(len(rows)), 2):
            if grams[a] and grams[b] and len(grams[a] & grams[b]) / len(grams[a] | grams[b]) >= .85:
                expected.add((a, b))
        self.assertEqual({(a, b) for a, b, _ in f.near_pairs(rows)}, expected)

    def test_global_transitive_groups_and_input_order_independent_splits(self):
        rows = [row(1, 'identical source text', label='0'),
                row(2, 'identical source text', event='sydneysiege', label='1'),
                row(3, 'a completely unrelated reply')]
        rows[2]['reply_to_id'] = rows[1]['id']
        f.group_rows(rows)
        self.assertEqual(len({r['group_id'] for r in rows}), 1)
        f.assign_splits(rows, 42)
        other = list(reversed(copy.deepcopy(rows)))
        f.group_rows(other)
        f.assign_splits(other, 42)
        self.assertEqual({r['id']: r['split'] for r in rows}, {r['id']: r['split'] for r in other})

    def test_pseudo_conflicts_and_nonfinite_scores_are_not_reused(self):
        rows = [row(i, str(i)) for i in range(4)]
        def pseudo(score):
            return {k: v for d in f.DIMS for k, v in [(d+'_label', score), (d+'_parse_ok', '1')]}
        m = {'legacy_record_id': 'legacy:1'}
        data = {rows[0]['sample_id']: [(m, pseudo('.4'))],
                rows[1]['sample_id']: [(m, pseudo('.4')), (m, pseudo('.6'))],
                rows[2]['sample_id']: [(m, pseudo('nan'))]}
        f.migrate_pseudo(rows, data)
        self.assertEqual([r['pseudo_status'] for r in rows], ['reused_identical_normalized_text',
                         'conflicting_legacy_pseudo', 'invalid_legacy_pseudo', 'regeneration_required'])
        self.assertEqual(rows[1]['emotion_label'], '')

    def test_checker_rejects_identity_and_text_leakage(self):
        rows = [row(1, 'same'), row(2, 'same')]
        for r, split in zip(rows, ('train', 'test')):
            r.update(split=split, group_id=r['sample_id'])
        with self.assertRaisesRegex(ValueError, 'Cross-split overlap'):
            f.check_disjoint(rows, [])

    def test_build_reproducible_seal_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'archive'
            archive.write_bytes(b'fixture')
            rows = [row(i*40+j, f.digest(f'{event}:{j}'), event, str(j % 2))
                    for i, event in enumerate(f.EVENTS) for j in range(40)]
            mapping = dict(legacy_record_id='legacy:fixture', source_kind='raw', sample_id=rows[0]['sample_id'],
                           status='recovered', method='normalized_exact')
            for name in ('one', 'two'):
                with patch.object(f, 'load_official', return_value=copy.deepcopy(rows)), \
                     patch.object(f, 'recover_history', return_value=([dict(mapping)], {}, [])):
                    f.build(root, archive, root / name)
                self.assertEqual(f.verify(root / name, root)['status'], 'passed')
            self.assertEqual((root/'one/FREEZE.sha256').read_bytes(), (root/'two/FREEZE.sha256').read_bytes())
            with self.assertRaises(FileExistsError):
                f.build(root, archive, root / 'one')
            with (root/'one/split_charliehebdo/test.csv').open('a') as out:
                out.write('tamper')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                f.verify(root / 'one')


if __name__ == '__main__':
    unittest.main()
