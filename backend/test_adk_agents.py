
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))
load_dotenv()

from agents.adk_orchestrator import ADKOrchestrator

async def test_adk_analysis():
    print("--- Testing ADK Orchestrator Analysis ---")
    orchestrator = ADKOrchestrator()
    
    sample_profile = {
        "user_id": "test_user_adk",
        "zip_code": "60601",
        "age": 30,
        "income": 45000,
        "household_size": 1,
        "drugs": ["Ozempic"],
        "doctors": ["Dr. Sarah Chen"],
        "utilization": "sometimes",
        "tobacco_use": False
    }
    
    try:
        print("\n[1] Running Full Analysis...")
        result = await orchestrator.analyze(sample_profile)
        
        if result.get("recommendation"):
            print(f"✅ SUCCESS: Analysis generated.")
            # print(f"Recommendation Preview: {result['recommendation'][:200]}...")
        else:
            print("❌ FAILURE: No recommendation generated.")
            
        if result.get("plans") and len(result["plans"]) > 0:
            print(f"✅ SUCCESS: Found {len(result['plans'])} plans.")
            # Check if plan names are real
            plan_name = result["plans"][0]["name"]
            print(f"Sample Plan: {plan_name}")
            if "Blue" in plan_name or "Aetna" in plan_name or "UHC" in plan_name:
                print("Note: Plan names look like live API data.")
        else:
            print("❌ FAILURE: No plans in result.")
            
        if result.get("drugs") and result["drugs"].get("resolved_drugs"):
            print(f"✅ SUCCESS: Medication check returned data.")
        else:
            print("❌ FAILURE: Medication check failed.")

        # Test Chat
        print("\n[2] Testing Follow-up Chat...")
        chat_reply = await orchestrator.chat("test_user_adk", "Does the Blue plan cover my Ozempic?")
        if chat_reply.get("reply"):
            print(f"✅ SUCCESS: Chat reply: {chat_reply['reply'][:100]}...")
        else:
            print("❌ FAILURE: No chat reply.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_adk_analysis())
