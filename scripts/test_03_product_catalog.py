"""Test script for 03_product_catalog — CompositeNormalization, cost budget, checkpoints."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

from catchfly import Pipeline
from catchfly.demo import load_samples
from catchfly.extraction import LLMDirectExtraction
from catchfly.normalization import (
    CompositeNormalization,
    DictionaryNormalization,
    LLMCanonicalization,
)

MODEL = "gpt-4.1-mini"


# --- Schema ---
class ProductListing(BaseModel):
    product_name: str = Field(description="Full product name")
    brand: str = Field(description="Manufacturer brand name")
    category: str = Field(description="Product category, e.g. Smartphones, Audio, Laptops")
    rating: float | None = Field(default=None, description="Rating normalized to 0-10 scale")
    price: float | None = Field(default=None, description="Price in USD")
    pros: list[str] = Field(default_factory=list, description="Positive attributes")
    cons: list[str] = Field(default_factory=list, description="Negative attributes")


# --- CompositeNormalization ---
normalization = CompositeNormalization(
    field_strategies={
        "brand": DictionaryNormalization(
            mapping={
                "samsung": "Samsung", "SAMSUNG": "Samsung",
                "asus": "ASUS", "Asus": "ASUS",
                "lg": "LG", "Lg": "LG",
                "apple": "Apple", "APPLE": "Apple",
            },
            case_insensitive=True,
            passthrough_unmapped=True,
        ),
        "category": LLMCanonicalization(model=MODEL),
        "pros": LLMCanonicalization(model=MODEL),
        "cons": LLMCanonicalization(model=MODEL),
    },
)

# --- Load & run ---
docs = load_samples("product_reviews")
print(f"Loaded {len(docs)} product reviews\n")

pipeline = Pipeline(
    extraction=LLMDirectExtraction(model=MODEL, on_error="collect"),
    normalization=normalization,
    verbose=True,
)

# Estimate cost
estimate = pipeline.estimate_cost(docs)
print(f"Cost estimate: {estimate}\n")

results = pipeline.run(
    docs,
    schema=ProductListing,
    normalize_fields=["brand", "category", "pros", "cons"],
    max_cost_usd=2.0,
)

# --- Records ---
print("=" * 60)
print(f"EXTRACTED RECORDS ({len(results.records)} total, {len(results.errors)} errors)")
print("=" * 60)
for record in results.records[:5]:
    print(f"\n  {record.brand} {record.product_name}: ${record.price}, {record.rating}/10")
    print(f"    Pros: {record.pros}")
    print(f"    Cons: {record.cons}")

# --- Normalization ---
print(f"\n{'=' * 60}")
print("NORMALIZATION RESULTS")
print("=" * 60)
for field_name in ["brand", "category", "pros", "cons"]:
    norm = results.normalizations.get(field_name)
    if not norm or not norm.clusters:
        print(f"\n  '{field_name}': no clusters")
        continue
    n_multi = sum(1 for m in norm.clusters.values() if len(m) > 1)
    print(f"\n  '{field_name}' — {len(norm.clusters)} groups ({n_multi} with merges):")
    for canonical, members in norm.clusters.items():
        if len(members) > 1:
            print(f"    {canonical}: {members}")

# --- Export ---
print(f"\n{'=' * 60}")
print("EXPORT")
print("=" * 60)
df = results.to_dataframe()
print(df[["product_name", "brand", "category", "rating", "price"]].to_string())

print(f"\nCost: ${results.report.total_cost_usd:.4f}")
print(f"Tokens: {results.report.total_input_tokens + results.report.total_output_tokens:,}")
