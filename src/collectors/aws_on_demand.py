import os
import json
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

# Load .env (for AWS creds if you use them that way)
load_dotenv()

# ---- CONFIG SECTION ---------------------------------------------------------

# GPU instance types we care about
GPU_INSTANCE_TYPES = [
    "p5.48xlarge",    # 8x H100
    "p4d.24xlarge",   # 8x A100
    "g5.xlarge",      # 1x A10G
    "g5.2xlarge",     # 1x A10G
    "g5.12xlarge",    # 4x A10G
    "g5.48xlarge",    # 8x A10G
    "g4dn.xlarge",    # 1x T4
    "g4dn.12xlarge",  # 4x T4
]


AWS_PRICING_REGION = "us-east-1"

# Map human-readable "location" from Pricing API to AWS region code
LOCATION_TO_REGION = {
    "US East (N. Virginia)": "us-east-1",
    "US East (Ohio)": "us-east-2",
    "US West (Oregon)": "us-west-2",
    "EU (Ireland)": "eu-west-1",
    "EU (Frankfurt)": "eu-central-1",
    "Asia Pacific (Sydney)": "ap-southeast-2",
    "Asia Pacific (Tokyo)": "ap-northeast-1",
    "Asia Pacific (Singapore)": "ap-southeast-1",
    # Add more if you need them
}

# Approximate FP16 TFLOPS per GPU model (can refine later)
GPU_TFLOPS_FP16 = {
    "H100": 950,
    "A100": 312,
    "V100": 112,
    "L40S": 181,
    "L40": 181,
    "A10G": 72,
    "T4": 8,
}

# Output path
OUTPUT_PATH = Path("data/raw/aws_on_demand_prices.csv")


# ---- HELPER FUNCTIONS -------------------------------------------------------


def get_pricing_client():
    """
    AWS Pricing API only exists in us-east-1, regardless of where your workloads live.
    """
    return boto3.client("pricing", region_name=AWS_PRICING_REGION)


def infer_gpu_model(instance_type: str) -> str:
    """
    Infer GPU model from instance_type naming convention.
    This is approximate but good enough for our initial pricing engine.
    """
    if instance_type.startswith("p5."):
        return "H100"
    if instance_type.startswith("p4d."):
        return "A100"
    if instance_type.startswith("p3."):
        return "V100"
    if instance_type.startswith("g5."):
        return "A10G"
    if instance_type.startswith("g4dn."):
        return "T4"
    # If we don't recognise it, return empty string
    return ""


def get_tflops_total(instance_type: str, gpu_count: int) -> float:
    model = infer_gpu_model(instance_type)
    per_gpu = GPU_TFLOPS_FP16.get(model, 0)
    return per_gpu * gpu_count


def ensure_output_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# ---- MAIN LOGIC -------------------------------------------------------------


def fetch_on_demand_gpu_prices(instance_types):
    """
    Fetch on-demand prices for a list of instance types.
    We do not filter by location; instead we read the 'location' attribute from each product,
    then map it to an AWS region code (if available).
    """
    client = get_pricing_client()
    rows = []

    for itype in instance_types:
        paginator = client.get_paginator("get_products")
        page_iterator = paginator.paginate(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": itype},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
        )

        for page in page_iterator:
            for price_str in page["PriceList"]:
                product = json.loads(price_str)
                attr = product["product"]["attributes"]

                instance_type = attr.get("instanceType")
                location = attr.get("location")
                # locationType can be "AWS Region", "AWS Outpost", etc
                location_type = attr.get("locationType")

                # Many GPU instances have 'gpu' as a string like "1", "4", "8"
                raw_gpu = attr.get("gpu", "0")
                try:
                    gpu_count = int(raw_gpu)
                except ValueError:
                    gpu_count = 0

                # Map Pricing API name -> region code if possible
                region_code = LOCATION_TO_REGION.get(location, None)

                # Extract OnDemand terms
                terms = product.get("terms", {}).get("OnDemand", {})
                for term_code, term in terms.items():
                    for dim_code, dim in term["priceDimensions"].items():
                        price_per_unit = dim.get("pricePerUnit", {})
                        usd_str = price_per_unit.get("USD")
                        if usd_str is None:
                            # Skip dimensions without a USD price
                            continue

                        try:
                            price_per_hour = float(usd_str)
                        except (TypeError, ValueError):
                            continue

                        unit = dim.get("unit", "Hrs")

                        rows.append(
                            {
                                "provider": "AWS",
                                "market_type": "on-demand",
                                "instance_type": instance_type,
                                "location": location,
                                "location_type": location_type,
                                "region": region_code,
                                "gpu_count": gpu_count,
                                "price_per_hour_usd": price_per_hour,
                                "unit": unit,
                            }
                        )

    df = pd.DataFrame(rows)
    return df


def enrich_with_tflops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add GPU model, TFLOPS, and cost_per_tflop_hour_usd.
    """
    df = df.copy()

    df["gpu_model"] = df["instance_type"].apply(infer_gpu_model)
    df["tflops_total"] = df.apply(
        lambda row: get_tflops_total(row["instance_type"], row["gpu_count"]), axis=1
    )

    def cost_per_tflop(row):
        t = row["tflops_total"]
        if t and t > 0:
            return row["price_per_hour_usd"] / t
        return None

    df["cost_per_tflop_hour_usd"] = df.apply(cost_per_tflop, axis=1)

    return df


def main():
    print("Fetching AWS on-demand GPU prices from Pricing API...")
    df_raw = fetch_on_demand_gpu_prices(GPU_INSTANCE_TYPES)
    print(f"Fetched {len(df_raw)} price rows.")

    df_enriched = enrich_with_tflops(df_raw)

    ensure_output_dir(OUTPUT_PATH)
    df_enriched.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved enriched on-demand GPU prices to: {OUTPUT_PATH.resolve()}")
    print(df_enriched.head())


if __name__ == "__main__":
    main()
