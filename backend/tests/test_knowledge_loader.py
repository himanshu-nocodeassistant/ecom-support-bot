import tempfile
import unittest
from pathlib import Path

from backend.app.data_loader import chunk_knowledge_documents, load_knowledge_documents


class KnowledgeLoaderTests(unittest.TestCase):
    def test_load_knowledge_documents_reads_markdown_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shipping-policy.md"
            path.write_text(
                "# Shipping policy\n\nOrders usually ship within 2 business days.\n",
                encoding="utf-8",
            )

            documents = load_knowledge_documents(temp_dir)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].id, "shipping-policy")
            self.assertEqual(documents[0].title, "Shipping policy")
            self.assertIn("ship within 2 business days", documents[0].content)

    def test_chunk_knowledge_documents_splits_large_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "portable-blender-guide.md"
            path.write_text(
                "# Portable blender guide\n\n"
                "The portable blender has a 400ml jar.\n\n"
                "It includes a safety lock that prevents blending when the lid is not secured.\n\n"
                "Customers asking why the blender will not start should check the lid alignment first.\n",
                encoding="utf-8",
            )

            documents = load_knowledge_documents(temp_dir)
            chunks = chunk_knowledge_documents(documents, target_chunk_size=90)

            self.assertGreaterEqual(len(chunks), 2)
            self.assertEqual(chunks[0].document_id, "portable-blender-guide")
            self.assertIn("chunk_index", chunks[0].metadata)
