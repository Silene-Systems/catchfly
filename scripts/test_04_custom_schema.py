"""Test script for 04_custom_schema — user-provided Pydantic schema + auto field selection."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

from catchfly import Pipeline
from catchfly.demo import load_samples

MODEL = "gpt-4.1-mini"


# --- Schema ---
class SupportTicket(BaseModel):
    subject: str = Field(description="Brief summary of the issue")
    category: str = Field(description="e.g., Authentication, Billing, Bug, Feature Request")
    priority: str = Field(description="Critical, High, Medium, or Low")
    product_area: str = Field(description="Which product area is affected")
    customer_email: str | None = Field(default=None, description="Customer contact email if mentioned")
    is_resolved: bool = Field(default=False, description="Whether the issue appears resolved")


# --- Load & run with auto field selection ---
docs = load_samples("support_tickets")
print(f"Loaded {len(docs)} support tickets\n")

pipeline = Pipeline.quick(model=MODEL)

# No normalize_fields — LLMFieldSelector auto-detects
results = pipeline.run(docs, schema=SupportTicket)

print(f"Extracted {len(results.records)} tickets")
print(f"Auto-normalized fields: {list(results.normalizations.keys())}\n")

# --- Records ---
print("=" * 60)
print("EXTRACTED RECORDS")
print("=" * 60)
for ticket in results.records[:5]:
    email = f" ({ticket.customer_email})" if ticket.customer_email else ""
    print(f"  [{ticket.priority}] {ticket.category}: {ticket.subject}{email}")

# --- Normalization ---
print(f"\n{'=' * 60}")
print("NORMALIZATION RESULTS")
print("=" * 60)
for field_name, norm in results.normalizations.items():
    if not norm.clusters:
        continue
    print(f"\n  '{field_name}':")
    for canonical, members in norm.clusters.items():
        print(f"    {canonical}: {members}")

# --- Cost ---
print(f"\nTotal cost: ${results.report.total_cost_usd:.4f}")
