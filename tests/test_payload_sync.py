"""Behavioral checks for deployment updates, without model assets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sync_runtime_payload import synchronize


class PayloadSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.target = self.root/'source', self.root/'target'
        self.source.mkdir(); self.target.mkdir()
        (self.source/'app.py').write_text('new')
        (self.target/'app.py').write_text('old')
        (self.source/'added.py').write_text('added')
        (self.target/'unlisted.txt').write_text('keep')
        self.manifest = self.root/'files.json'
        self.backup, self.report = self.root/'backup', self.root/'report.json'

    def run_update(self, files, dry_run=False):
        self.manifest.write_text(json.dumps({'files': files}))
        return synchronize(self.source, self.target, self.manifest, self.backup, self.report, dry_run)

    def test_preview_does_not_write(self):
        result = self.run_update(['app.py', 'added.py'], True)
        self.assertEqual(len(result['changes']), 2)
        self.assertEqual((self.target/'app.py').read_text(), 'old')
        self.assertFalse(self.backup.exists())
        self.assertFalse(self.report.exists())
        self.assertFalse((self.target/'added.py').exists())

    def test_apply_backs_up_and_records_additions(self):
        result = self.run_update(['app.py', 'added.py'])
        self.assertTrue(result['passed'])
        self.assertEqual((self.backup/'app.py').read_text(), 'old')
        self.assertEqual((self.target/'app.py').read_text(), 'new')
        self.assertEqual((self.target/'added.py').read_text(), 'added')
        self.assertEqual((self.target/'unlisted.txt').read_text(), 'keep')
        self.assertIsNone(result['changes'][1]['before_sha256'])
        with self.assertRaises(FileExistsError):
            self.run_update(['app.py'])

    def test_missing_late_source_does_not_partially_update(self):
        with self.assertRaises(FileNotFoundError):
            self.run_update(['app.py', 'absent.py'])
        self.assertEqual((self.target/'app.py').read_text(), 'old')
        self.assertFalse(self.backup.exists())

    def test_unsafe_and_duplicate_paths(self):
        for paths in (['../app.py'], ['C:/app.py'], ['app.py:stream'], ['app.py', 'APP.py']):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.run_update(paths)
        self.assertFalse(self.backup.exists())

    def test_parent_collision_does_not_partially_update(self):
        (self.source/'blocked').mkdir()
        (self.source/'blocked/child.py').write_text('child')
        (self.target/'blocked').write_text('blocking-file')
        with self.assertRaises(ValueError):
            self.run_update(['app.py', 'blocked/child.py'])

    def test_overlap_rejected(self):
        self.manifest.write_text('{"files":["app.py"]}')
        with self.assertRaises(ValueError):
            synchronize(self.source,self.target,self.manifest,self.target/'backup',self.report)


if __name__ == '__main__':
    unittest.main()
