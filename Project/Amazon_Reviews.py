import json
from tkinter import filedialog
import tkinter as tk
import pickle
import os
CACHE_FILE = "data_cache.pkl"


data = []




def filter_data():
    filtered_data = []
    for item in data:
        filtered_item = {
            'rating': item['rating'],
            'text': item['text']
        }
        filtered_data.append(filtered_item)
    return filtered_data

    

def load_data(LIMIT):
    global data
    
    if os.path.exists(CACHE_FILE):
        print("Loading from cache...")
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        data = data[:LIMIT]  # slice after loading
        print(f"Loaded {len(data)} records from cache")
        return

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select JSONL File",
        filetypes=[("JSON Lines", "*.jsonl"), ("All Files", "*.*")]
    )
    if file_path:
        print(f"Loading: {file_path} ...")
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= LIMIT:
                    break
                data.append(json.loads(line))
        print(f"Loaded {len(data)} records")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(data, f)
        print("Saved to cache")
    else:
        print("No file selected")
    root.destroy()
    


def get_data():
    return filter_data()


