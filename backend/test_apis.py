
import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from dotenv import load_dotenv
load_dotenv() # Load from root .env

from tools.gov_apis import search_plans, get_eligibility_estimates, resolve_drug_rxcui

async def test_api_calls():
    print("--- Testing API Live Status ---")
    
    # 1. Test ZIP to FIPS (usually local/cached but calls CMS if missing)
    print("\n[1] Testing Plan Search (CMS API)...")
    try:
        # Using a well-known ZIP for testing (Chicago)
        plans = search_plans(zip_code="60601", age=30, income=45000, fips="17031", state="IL")
        if plans:
            print(f"✅ SUCCESS: Found {len(plans)} plans via CMS API.")
            print(f"Sample Plan: {plans[0]['name']} - Premium: ${plans[0]['premium']}")
            # Check if it looks like mock data (mock data often has specific IDs or names)
            if "BlueCross" in plans[0]['name'] or "Aetna" in plans[0]['name']:
                print("Note: Plan names match common real carriers.")
        else:
            print("❌ FAILURE: No plans returned.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # 2. Test Eligibility (CMS API)
    print("\n[2] Testing Eligibility Estimate (CMS API)...")
    try:
        est = get_eligibility_estimates(income=35000, age=30, fips="17031", zip_code="60601", state="IL")
        if est and est.get("aptc") is not None:
            print(f"✅ SUCCESS: APTC Estimate: ${est['aptc']}")
        else:
            print("❌ FAILURE: No eligibility data returned.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

    # 3. Test Drug Resolution (RxNorm/CMS API)
    print("\n[3] Testing Drug Resolution (RxNorm/CMS API)...")
    try:
        drug = resolve_drug_rxcui("Ozempic")
        if drug:
            print(f"✅ SUCCESS: Resolved Ozempic to RxCUI: {drug['rxcui']}")
        else:
            print("❌ FAILURE: Could not resolve drug.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_calls())
