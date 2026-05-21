import csv
import tempfile
import unittest
from pathlib import Path

from backend.app.data_loader import export_orders_csv, load_olist_orders


class DataLoaderTests(unittest.TestCase):
    def test_load_olist_orders_normalizes_minimal_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._write_csv(
                temp_path / "olist_orders_dataset.csv",
                [
                    "order_id",
                    "customer_id",
                    "order_status",
                    "order_delivered_carrier_date",
                    "order_estimated_delivery_date",
                ],
                [["ORD-1", "CUST-1", "delivered", "2024-01-01", "2024-01-05"]],
            )
            self._write_csv(
                temp_path / "olist_order_items_dataset.csv",
                ["order_id", "product_id"],
                [["ORD-1", "PROD-1"]],
            )
            self._write_csv(
                temp_path / "olist_products_dataset.csv",
                ["product_id", "product_category_name"],
                [["PROD-1", "electronics"]],
            )
            self._write_csv(
                temp_path / "olist_customers_dataset.csv",
                ["customer_id", "customer_unique_id"],
                [["CUST-1", "customer-001"]],
            )
            self._write_csv(
                temp_path / "product_category_name_translation.csv",
                ["product_category_name", "product_category_name_english"],
                [["electronics", "electronics"]],
            )

            rows = load_olist_orders(temp_dir)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].order_id, "ORD-1")
            self.assertEqual(rows[0].customer_name, "customer-001")
            self.assertEqual(rows[0].item, "electronics")
            self.assertTrue(rows[0].delivered)

    def test_export_orders_csv_writes_normalized_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self._write_csv(
                temp_path / "olist_orders_dataset.csv",
                [
                    "order_id",
                    "customer_id",
                    "order_status",
                    "order_delivered_carrier_date",
                    "order_estimated_delivery_date",
                ],
                [["ORD-1", "CUST-1", "shipped", "", "2024-01-05"]],
            )
            self._write_csv(
                temp_path / "olist_order_items_dataset.csv",
                ["order_id", "product_id"],
                [["ORD-1", "PROD-1"]],
            )
            self._write_csv(
                temp_path / "olist_products_dataset.csv",
                ["product_id", "product_category_name"],
                [["PROD-1", "kitchen"]],
            )
            self._write_csv(
                temp_path / "olist_customers_dataset.csv",
                ["customer_id", "customer_unique_id"],
                [["CUST-1", "customer-001"]],
            )
            self._write_csv(
                temp_path / "product_category_name_translation.csv",
                ["product_category_name", "product_category_name_english"],
                [["kitchen", "kitchen"]],
            )

            output_path = temp_path / "normalized_orders.csv"
            count = export_orders_csv(temp_dir, str(output_path))

            self.assertEqual(count, 1)
            self.assertTrue(output_path.exists())

    def _write_csv(self, path: Path, headers: list[str], rows: list[list[str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
