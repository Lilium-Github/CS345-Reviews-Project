import json
from tkinter import filedialog
import tkinter as tk


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

    

def load_data():

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select JSONL File",
        filetypes=[("JSON Lines", "*.jsonl"), ("All Files", "*.*")]
    )

    if file_path:
        print(f"Loading: {file_path} ...")
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
    
        print(f"Loaded {len(data)} records from {file_path}")
   
    else:
        print("No file selected")


    root.destroy()

    


def get_data():
    return filter_data()


