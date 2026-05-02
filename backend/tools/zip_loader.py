"""
ZIP to FIPS loader - reads Census ZCTA file + fills gaps using nearby ZIP inference
"""

import os

_zip_fips_cache = {}
_loaded = False

def _load():
    global _zip_fips_cache, _loaded
    if _loaded:
        return

    paths = [
        os.path.join(os.path.dirname(__file__), "..", "data", "tab20_zcta520_county20_natl.txt"),
        os.path.join(os.path.dirname(__file__), "..", "tab20_zcta520_county20_natl.txt"),
        "/app/data/tab20_zcta520_county20_natl.txt",
    ]

    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8-sig") as f:
                    header = f.readline().strip().split("|")
                    name_col = header.index("NAMELSAD_ZCTA5_20")
                    fips_col = header.index("GEOID_COUNTY_20")
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) > max(name_col, fips_col):
                            name = parts[name_col].strip()
                            fips = parts[fips_col].strip()
                            if name.startswith("ZCTA5 "):
                                z = name[6:]
                                if len(z) == 5 and len(fips) == 5 and z not in _zip_fips_cache:
                                    _zip_fips_cache[z] = fips
                print("Loaded " + str(len(_zip_fips_cache)) + " ZIP-FIPS mappings from census file")
                _loaded = True
                return
            except Exception as e:
                print("Census file error: " + str(e))

    _loaded = True


def get_fips_for_zip(zip_code: str):
    _load()

    # Direct lookup
    if zip_code in _zip_fips_cache:
        return _zip_fips_cache[zip_code]

    # Infer from nearby ZIPs (PO Box ZIPs are gaps in census data)
    # Try incrementing/decrementing the last digits
    try:
        num = int(zip_code)
        for delta in [1, -1, 2, -2, 5, -5, 10, -10, 20, -20, 50, -50]:
            nearby = str(num + delta).zfill(5)
            # Only look in same prefix range (same city area)
            if nearby[:3] == zip_code[:3] and nearby in _zip_fips_cache:
                result = _zip_fips_cache[nearby]
                _zip_fips_cache[zip_code] = result  # cache the inferred result
                print("FIPS inferred for " + zip_code + " from nearby " + nearby + ": " + result)
                return result
    except Exception:
        pass

    return None


def total_zips_loaded():
    _load()
    return len(_zip_fips_cache)
