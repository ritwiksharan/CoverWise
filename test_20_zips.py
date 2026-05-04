import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from dotenv import load_dotenv
load_dotenv()

from tools.gov_apis import get_fips_from_zip, _fips_to_state

def test_20_dynamic_zips():
    # 20 ZIP codes strictly NOT in the KNOWN_FIPS cache
    # All are from federal marketplace states to ensure we get FIPS codes successfully
    test_zips = [
        ("35203", "Birmingham, AL"),
        ("72201", "Little Rock, AR"),
        ("32301", "Tallahassee, FL"),
        ("31401", "Savannah, GA"),
        ("46204", "Indianapolis, IN"),
        ("50309", "Des Moines, IA"),
        ("67202", "Wichita, KS"),
        ("70112", "New Orleans, LA"),
        ("39201", "Jackson, MS"),
        ("63101", "St. Louis, MO"),
        ("68508", "Lincoln, NE"),
        ("27601", "Raleigh, NC"),
        ("58102", "Fargo, ND"),
        ("45202", "Cincinnati, OH"),
        ("73102", "Oklahoma City, OK"),
        ("29401", "Charleston, SC"),
        ("57104", "Sioux Falls, SD"),
        ("84101", "Salt Lake City, UT"),
        ("23219", "Richmond, VA"),
        ("53202", "Milwaukee, WI")
    ]

    print("--- Testing CMS FIPS Lookup for 20 Dynamic ZIP Codes ---")
    
    success_count = 0
    error_count = 0

    for zip_code, location in test_zips:
        try:
            # This directly calls the CMS API because these are NOT in KNOWN_FIPS
            fips = get_fips_from_zip(zip_code)
            
            if fips:
                state = _fips_to_state(fips)
                print(f"✅ {location} ({zip_code}) -> FIPS: {fips} | State: {state}")
                success_count += 1
            else:
                print(f"❌ {location} ({zip_code}) -> FAILED to resolve FIPS")
                error_count += 1
                
        except Exception as e:
            print(f"❌ Error during search for {zip_code}: {e}")
            error_count += 1

    print("\n--- Summary ---")
    print(f"Dynamic API Lookups Succeeded: {success_count}/20")
    print(f"Dynamic API Lookups Failed:    {error_count}/20")

if __name__ == "__main__":
    test_20_dynamic_zips()
