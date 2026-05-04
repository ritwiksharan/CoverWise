import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from dotenv import load_dotenv
load_dotenv()

from tools.gov_apis import search_plans, get_fips_from_zip, _fips_to_state

def test_10_states():
    # Diverse set of ZIP codes across the US
    # Note: Some states (like CA, NY) run their own exchanges and might return 0 plans on CMS.
    # We will test a mix of federal exchange states (TX, FL, IL, NC, GA, OH, PA, MI, AZ, TN)
    test_zips = [
        ("77002", "Texas (Houston)"),
        ("33101", "Florida (Miami)"),
        ("60601", "Illinois (Chicago)"),
        ("28202", "North Carolina (Charlotte)"),
        ("30301", "Georgia (Atlanta)"),
        ("43215", "Ohio (Columbus)"),
        ("15201", "Pennsylvania (Pittsburgh)"),
        ("48201", "Michigan (Detroit)"),
        ("85001", "Arizona (Phoenix)"),
        ("37201", "Tennessee (Nashville)")
    ]

    print("--- Testing CMS Plan Search for 10 States ---")
    
    success_count = 0
    mock_count = 0
    error_count = 0

    for zip_code, location in test_zips:
        print(f"\n[{location} - ZIP: {zip_code}]")
        try:
            fips = get_fips_from_zip(zip_code)
            state = _fips_to_state(fips) if fips else "US"
            
            # Use typical demographics
            age = 35
            income = 50000
            
            plans = search_plans(zip_code=zip_code, age=age, income=income, fips=fips, state=state)
            
            if not plans:
                print("❌ No plans returned.")
                error_count += 1
                continue
                
            print(f"✅ Found {len(plans)} plans.")
            
            # Check if it's mock data or real data
            # Real CMS data usually has many plans and specific issuers.
            first_plan_name = plans[0].get("name", "")
            issuer = plans[0].get("issuer", "")
            
            if "Mock" in first_plan_name or "Example" in first_plan_name:
                print(f"⚠️ Returned MOCK data: {first_plan_name}")
                mock_count += 1
            else:
                print(f"✅ REAL CMS Data -> Top Plan: {first_plan_name} | Issuer: {issuer}")
                success_count += 1
                
        except Exception as e:
            print(f"❌ Error during search: {e}")
            error_count += 1

    print("\n--- Summary ---")
    print(f"Real API Results: {success_count}/10")
    print(f"Mock Data Used:   {mock_count}/10")
    print(f"Errors:           {error_count}/10")

if __name__ == "__main__":
    test_10_states()
