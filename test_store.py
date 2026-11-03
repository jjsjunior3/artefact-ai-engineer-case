import csv
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pypdf import PdfWriter

from catalog import recommend
from policies import get_policy_page, load_policy_pages


class CatalogTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data = Path(directory.name)
        self.write_table("categories", ["category_id", "name"], [(1, "Guitars")])
        products = [
            (number, number * 100, 1, "active", 1)
            for number in range(1, 8)
        ]
        products.extend([(8, 50, 1, "active", 0), (9, 60, 1, "inactive", 1)])
        self.write_table(
            "products",
            ["product_id", "price_brl", "category_id", "status", "stock_quantity"],
            products,
        )

    def write_table(self, name, columns, rows):
        path = self.data / f"desafio_tecnico_ai_eng - {name}.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_counts_all_matches_and_selects_four(self):
        total, products, extra = recommend(self.data, "guitars", Decimal("550"))
        self.assertEqual(total, 5)
        self.assertEqual([row["product_id"] for row in products], ["1", "2", "3", "5"])
        self.assertEqual(extra["product_id"], "6")

    def test_exact_budget_is_included_and_large_upsell_is_excluded(self):
        total, products, extra = recommend(self.data, "Guitars", Decimal("200"))
        self.assertEqual(total, 2)
        self.assertEqual([row["price_brl"] for row in products], ["100", "200"])
        self.assertIsNone(extra)

    def test_invalid_ceiling(self):
        with self.assertRaises(ValueError):
            recommend(self.data, "Guitars", Decimal("NaN"))


class PolicyTests(unittest.TestCase):
    def test_pdf_pages_are_numbered_from_one(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manual.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.add_blank_page(width=100, height=100)
            writer.write(path)
            self.assertEqual(load_policy_pages(path), {1: "", 2: ""})

    def test_evidence_includes_source_page(self):
        pages = {4: "Online returns within seven days."}
        self.assertEqual(get_policy_page(pages, 4), {"page": 4, "text": pages[4]})
        self.assertEqual(get_policy_page(pages, 9), {"page": 9, "text": ""})

class AgentRoutingTests(unittest.TestCase):
    setUp = CatalogTests.setUp
    write_table = CatalogTests.write_table

    def test_routes_catalog_and_policy_calls(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from agent import build_graph

        class FakeModel:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                if isinstance(messages[-1], ToolMessage):
                    return AIMessage(content="Done")
                return AIMessage(content="", tool_calls=[
                    {"name": "search_catalog", "args": {"category": "Guitars", "ceiling": "550"}, "id": "catalog"},
                    {"name": "read_policy", "args": {"page": 4}, "id": "policy"},
                ])

        import json

        graph = build_graph(FakeModel(), self.data, {4: "Returns policy"})
        result = graph.invoke({"messages": [HumanMessage(content="Show guitars and return policy")]})
        evidence = {
            message.tool_call_id: json.loads(message.content)
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        }
        self.assertEqual(evidence["catalog"]["total"], 5)
        self.assertEqual(evidence["policy"], {"page": 4, "text": "Returns policy"})
        self.assertEqual(result["messages"][-1].content, "Done")

