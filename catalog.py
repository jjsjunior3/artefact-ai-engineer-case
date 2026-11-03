import json
import csv
from decimal import Decimal
from pathlib import Path


def read_rows(data_dir: Path, table: str) -> list[dict[str, str]]:
    path = data_dir / f"desafio_tecnico_ai_eng - {table}.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def recommend(
    data_dir: Path, category: str, ceiling: Decimal, *, strings: str | None = None, amplified: bool = False
) -> tuple[int, list[dict[str, str]], dict[str, str] | None]:
    if not ceiling.is_finite() or ceiling < 0:
        raise ValueError("The price ceiling must be finite and nonnegative.")

    category_ids = {
        row["category_id"]
        for row in read_rows(data_dir, "categories")
        if row["name"].casefold() == category.strip().casefold()
    }
    def matches_specs(row):
        if strings is None and not amplified:
            return True
        specs = json.loads(row.get("specs") or "{}")
        return (
            (strings is None or specs.get("strings") == strings)
            and (not amplified or specs.get("electronics") == "yes")
        )

    available = sorted(
        (
            row for row in read_rows(data_dir, "products")
            if row["category_id"] in category_ids
            and row["status"] == "active"
            and int(row["stock_quantity"]) > 0
            and matches_specs(row)
        ),
        key=lambda row: (Decimal(row["price_brl"]), int(row["product_id"])),
    )
    within = [row for row in available if Decimal(row["price_brl"]) <= ceiling]
    above = next(
        (row for row in available if ceiling < Decimal(row["price_brl"]) <= ceiling * Decimal("1.10")),
        None,
    )
    if not within:
        return 0, [], available[0] if available else None
    if len(within) <= 4:
        return len(within), within, above

    from statistics import median

    midpoint = median(Decimal(row["price_brl"]) for row in within)
    middle = min(within[2:-1], key=lambda row: abs(Decimal(row["price_brl"]) - midpoint))
    return len(within), [*within[:2], middle, within[-1]], above
def find_products(data_dir: Path, name: str) -> list[dict[str, str]]:
    """Find active products by case-insensitive name or model substring."""
    query = name.strip().casefold()
    if not query:
        return []
    return [
        row for row in read_rows(data_dir, "products")
        if row["status"] == "active" and query in row["name"].casefold()
    ]
