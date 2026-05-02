#!/usr/bin/env python
"""Test script to verify metrics endpoints."""

import httpx
import json
from datetime import datetime

BASE_URL = "http://localhost:8082"
AUTH_TOKEN = "freecc"  # ANTHROPIC_AUTH_TOKEN


async def test_metrics():
    """Test the metrics endpoints."""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

        # Get full metrics (by model and hour)
        print("=" * 60)
        print("📊 FULL METRICS (by model and hour)")
        print("=" * 60)
        response = await client.get(f"{BASE_URL}/metrics", headers=headers)
        if response.status_code == 200:
            metrics = response.json()
            print(json.dumps(metrics, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)

        # Get model summary
        print("\n" + "=" * 60)
        print("📈 MODEL SUMMARY (total requests per model)")
        print("=" * 60)
        response = await client.get(f"{BASE_URL}/metrics/summary", headers=headers)
        if response.status_code == 200:
            summary = response.json()
            print(json.dumps(summary, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)

        # Get hourly summary
        print("\n" + "=" * 60)
        print("⏰ HOURLY SUMMARY (total requests per hour)")
        print("=" * 60)
        response = await client.get(f"{BASE_URL}/metrics/hourly", headers=headers)
        if response.status_code == 200:
            hourly = response.json()
            print(json.dumps(hourly, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)


if __name__ == "__main__":
    import asyncio

    print(f"Testing metrics endpoints at {BASE_URL}\n")
    print(f"Auth token: {AUTH_TOKEN}\n")
    asyncio.run(test_metrics())
