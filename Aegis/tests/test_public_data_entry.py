"""Public data ingestion without archived development files."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils import freeze_pheme as f
from test_data_freeze import row


class PublicDataTests(unittest.TestCase):
    def test_official_archive_build_without_legacy_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'pheme.tar.gz'
            archive.write_bytes(b'offline fixture')
            rows = [row(i + 100*j, f'{event} distinct {i} sample {i*i}', event=event, label=str(i%2))
                    for j,event in enumerate(f.EVENTS) for i in range(20)]
            with patch.object(f, 'load_official', return_value=rows):
                result = f.build(None, archive, root/'frozen')
            self.assertEqual(result['total_rows'], 100)
            self.assertEqual(result['legacy_record_count'], 0)
            self.assertEqual(f.verify(root/'frozen')['status'], 'passed')
            self.assertEqual(len(f.read_csv(root/'frozen/pseudo_regeneration_queue.csv')), 100)


if __name__ == '__main__':
    unittest.main()
