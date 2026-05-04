import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from dotenv import load_dotenv
load_dotenv()

from agents.adk_orchestrator import ADKOrchestrator

async def test_states():
    orchestrator = ADKOrchestrator()
    
    # Test cases: IL (Chicago), TX (Houston), FL (Miami)
    test_profiles = [
        {"user_id": "test_il", "zip_code": "60601", "age": 30, "income": 45000, "household_size": 1, "drugs": ["Ozempic"], "doctors": []},
        {"user_id": "test_tx", "zip_code": "77002", "age": 45, "income": 60000, "household_size": 2, "drugs": [], "doctors": []},
        {"user_id": "test_fl", "zip_code": "33101", "age": 25, "income": 30000, "household_size": 1, "drugs": [], "doctors": []},
    ]

    for profile in test_profiles:
        print(f"\n--- Testing ZIP {profile['zip_code']} (User: {profile['user_id']}) ---")
        try:
            result = await orchestrator.analyze(profile)
            plans = result.get("plans", [])
            print(f"✅ Success! Found {len(plans)} plans.")
            if plans:
                print(f"   Top plan: {plans[0].get('name')} | Issuer: {plans[0].get('issuer')} | Net Premium: ${plans[0].get('premium_w_credit')}")
            print(f"   Recommendation length: {len(result.get('recommendation', ''))} chars")
            print(f"   Monthly APTC Subsidy: ${result.get('subsidy', {}).get('monthly_aptc', 0)}/mo")
        except Exception as e:
            print(f"❌ Error testing ZIP {profile['zip_code']}: {e}")

if __name__ == "__main__":
    asyncio.run(test_states())
