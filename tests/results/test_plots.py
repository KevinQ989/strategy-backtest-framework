import pickle
import os
from strategy_backtester.results import generate_dashboard

if __name__ == "__main__": 
    current_folder = os.path.dirname(__file__)
    root_folder = os.path.abspath(os.path.join(current_folder,"..",".."))
    print(root_folder)
    cache_path = os.path.join(root_folder, "tests", "results", "perm_test_data.pkl")

    try:
        with open(cache_path, "rb") as file:
            cached_results = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: could not find {cache_path}. Run main.py file first to generate pickle file.")
        exit()

    print("Test data loaded. Generating dashboard...")
    generate_dashboard(cached_results, rolling_window = 126)