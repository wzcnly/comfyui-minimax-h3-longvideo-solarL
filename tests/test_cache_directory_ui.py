import tempfile
import unittest
from pathlib import Path
import test_sampler
from longvideo.cache_directory_ui import is_local_request, writable_directory


class DirectoryTests(unittest.TestCase):
    def test_local_and_remote_requests(self):
        self.assertTrue(is_local_request('127.0.0.1', 'http://localhost:7820', 'localhost:7820'))
        self.assertTrue(is_local_request('::1', None, '[::1]:7820'))
        self.assertFalse(is_local_request('192.168.1.2', None, 'localhost:7820'))
        self.assertFalse(is_local_request('127.0.0.1', 'https://example.com', 'localhost:7820'))
        self.assertFalse(is_local_request('invalid', None, 'localhost:7820'))

    def test_writable_folder_no_leftover_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(writable_directory(directory), str(Path(directory).resolve()))
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaises(OSError):
                writable_directory(str(Path(directory)/'missing'))
