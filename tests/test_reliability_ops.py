"""Reliability ops tests: idempotent index, delete flow, validation, logging."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), 1.0, 0.0]


class FakeDoc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = dict(metadata or {})
        self.id = None


def make_chunks():
    return [
        FakeDoc("lease text one", {"source_path": "/t/a.pdf",
                                   "source": "/t/a.pdf",
                                   "file_name": "a.pdf", "page": 0,
                                   "document_id": "doc-A", "chunk_id": "c1"}),
        FakeDoc("lease text two", {"source_path": "/t/a.pdf",
                                   "source": "/t/a.pdf",
                                   "file_name": "a.pdf", "page": 1,
                                   "document_id": "doc-A", "chunk_id": "c2"}),
        FakeDoc("other doc", {"source_path": "/t/b.pdf",
                              "source": "/t/b.pdf",
                              "file_name": "b.pdf", "page": 0,
                              "document_id": "doc-B", "chunk_id": "c3"}),
    ]


class TestChromaIdempotent(unittest.TestCase):
    def test_reindex_same_folder_no_duplicates(self):
        from vector_store import ChromaVectorStore
        tmp = tempfile.mkdtemp(prefix="dedupe_")
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        vs = ChromaVectorStore(persist_directory=os.path.join(tmp, "db"),
                               embedding_function=DummyEmbeddings())
        vs.build_index(make_chunks())
        first = vs.get_status()["chunk_count"]
        self.assertEqual(first, 3)
        vs.build_index(make_chunks())
        self.assertEqual(vs.get_status()["chunk_count"], 3)

    def test_delete_by_document_removes_only_target(self):
        from vector_store import ChromaVectorStore
        tmp = tempfile.mkdtemp(prefix="deldoc_")
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        vs = ChromaVectorStore(persist_directory=os.path.join(tmp, "db"),
                               embedding_function=DummyEmbeddings())
        vs.build_index(make_chunks())
        self.assertEqual(vs.delete_by_document("doc-A"), 2)
        st = vs.get_status()
        self.assertEqual(st["chunk_count"], 1)
        self.assertEqual(vs.delete_by_document("doc-missing"), 0)
        self.assertEqual(vs.delete_by_document(""), 0)

    def test_backend_delete_flow(self):
        import backend
        import config
        tmp = tempfile.mkdtemp(prefix="bdel_")
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        from vector_store import ChromaVectorStore
        vs = ChromaVectorStore(persist_directory=os.path.join(tmp, "db"),
                               embedding_function=DummyEmbeddings())
        vs.build_index(make_chunks())
        res = backend.delete_indexed_document(
            "doc-B", config=config.load_config(), vector_store=vs)
        self.assertEqual(res, {"vectors_removed": 1, "file_removed": False})
        self.assertEqual(vs.get_status()["chunk_count"], 2)


class TestPathValidation(unittest.TestCase):
    def test_valid_folder(self):
        from document_storage import validate_local_path
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(validate_local_path(tmp),
                             os.path.abspath(tmp))

    def test_rejects_empty_missing_file(self):
        from document_storage import validate_local_path
        with self.assertRaises(ValueError):
            validate_local_path("")
        with self.assertRaises(ValueError):
            validate_local_path(os.path.join(tempfile.gettempdir(),
                                             "no-such-dir-xyz"))
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(ValueError):
                validate_local_path(f.name)

    def test_no_traversal(self):
        from document_storage import LocalDocumentStorage
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalDocumentStorage(tmp)
            with self.assertRaises(ValueError):
                store.save_document("../evil.txt", b"x")
            with self.assertRaises(ValueError):
                store.get_document(os.path.join(tmp, "..", "evil.txt"))

    def test_s3_prefix_validation(self):
        from document_storage import validate_s3_prefix
        self.assertEqual(validate_s3_prefix("documents"), "documents/")
        self.assertEqual(validate_s3_prefix("/documents//"), "documents/")
        with self.assertRaises(ValueError):
            validate_s3_prefix("../secret")


class TestNoPrints(unittest.TestCase):
    MODULES = ["backend.py", "document_loader.py", "model_warmup.py", "app.py"]

    def test_library_code_uses_logging(self):
        import logging
        root = os.path.join(os.path.dirname(__file__), "..")
        for name in self.MODULES:
            with open(os.path.join(root, name), encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith("print("):
                        self.fail(f"{name}:{i} uses print(): {stripped[:60]}")
        for mod in ("backend", "document_loader", "model_warmup"):
            self.assertIsInstance(logging.getLogger(mod),
                                  logging.Logger)


if __name__ == "__main__":
    unittest.main()
