import os
import sys
import pandas as pd

#Get path of this current file and root directory
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)

#Add root/data to list of directories for this file to search, to allow imports from data folder
sys.path.append(ROOT_DIRECTORY)

from data.loader import get_data

#Create test cache in data folder
TEST_CACHE = os.path.join(ROOT_DIRECTORY, 'data', 'test_cache.csv')

def run_tests():
    #Clear previous cache
    if os.path.exists(TEST_CACHE):
        os.remove(TEST_CACHE)
        print("Cleared old cache.")

    tickers = ["AAPL", "INTC"]

    #Test 1: Normal Download
    print("Test 1: Normal Download")
    df1 = get_data(tickers, "2026-01-05", "2026-01-16", cache_path=TEST_CACHE)
    print(f"[TEST] Dataframe size is {df1.shape[0]} rows, {df1.shape[1]} columns. Expected: 10 rows, 10 columns.\n")

    #Test 2: Pull data only from cache
    print("Test 2: Pull data only from cache")
    df2 = get_data(tickers, "2026-01-05", "2026-01-16", cache_path=TEST_CACHE)
    print("[TEST] Check if loader.py printed 'Loaded ticker entirely from cache.'\n")

    #Test 3: Requesting for data later than what exists in cache
    print("Test 3: Requesting for data later than what exists in cache")
    df3 = get_data(tickers, "2026-01-05", "2026-01-23", cache_path=TEST_CACHE)
    print(f"[TEST] Check if loader.py prints 'Downloading ticker from 2026-01-16 to 2026-01-23'")
    print(f"New cache size: {df3.shape[0]} rows and {df3.shape[1]} columns. Expected: 14 rows, 10 \n")

    #Test 4: Creating and detecting internal gap
    print("Test 4: Creating and detecting internal gap")
    df4_1 = pd.read_csv(TEST_CACHE, index_col=0, parse_dates=True)

    #Remove Jan 2 to Jan 14
    gap_mask = (df4_1.index > "2026-01-06") & (df4_1.index < "2026-01-22")
    df4_1 = df4_1[~gap_mask]
    df4_1.to_csv(TEST_CACHE)

    df4_2 = get_data(tickers, "2026-01-05", "2026-01-23", cache_path=TEST_CACHE)

    #Check if gaps have been patched
    if not df4_2.isnull().values.any():
        print("[TEST] Gaps have been patched. Check for 'Internal data gap detected' message from loader.py.")
    else:
        print("[TEST] Gaps were not patched successfully")

if __name__ == "__main__":
    run_tests()


