

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import yaml

# Function to load the config data from the YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Function to save the config data to the YAML file
def save_config(file_path, config_data):
    with open(file_path, 'w') as file:
        yaml.dump(config_data, file)

# Function to update the config data from the GUI entries and save it
def save_changes():
    # Update config data from the entries
    for section, entries in entry_widgets.items():
        for key, entry in entries.items():
            if isinstance(config_data[section][key], bool):
                config_data[section][key] = bool(entry.get())
            elif isinstance(config_data[section][key], int):
                config_data[section][key] = int(entry.get())
            elif isinstance(config_data[section][key], float):
                config_data[section][key] = float(entry.get())
            else:
                config_data[section][key] = entry.get()

    # Save updated config data
    save_config(config_path, config_data)
    messagebox.showinfo("Info", "Configuration saved successfully!")

# Load the config data
config_path = r'C:\Users\nati\Desktop\Implementations\FinalProject\FinalProject\config.yaml'
config_data = load_config(config_path)

# Create the tkinter GUI
root = tk.Tk()
root.title("Edit Config")

# Create the Notebook (tab container)
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

entry_widgets = {}

# Create a tab for each section in the config file
for idx, section in enumerate(config_data):
    tab_frame = ttk.Frame(notebook)
    notebook.add(tab_frame, text=section)

    # Adjust the column weight for the first tab
    if idx == 0:
        tab_frame.grid_columnconfigure(0, weight=1)
        tab_frame.grid_columnconfigure(1, weight=2)

    # Add a canvas to allow scrolling
    canvas = tk.Canvas(tab_frame)
    canvas.grid(row=0, column=0, sticky='news')

    # Add a scrollbar
    scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
    scrollbar.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame for the section content
    content_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=content_frame, anchor="nw")

    entry_widgets[section] = {}

    for i, (key, value) in enumerate(config_data[section].items()):
        tk.Label(content_frame, text=f"{key}:").grid(row=i, column=0, sticky="e", padx=5, pady=5)

        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            entry = tk.Checkbutton(content_frame, variable=var)
        else:
            entry = tk.Entry(content_frame)
            entry.insert(0, value)

        entry.grid(row=i, column=1, padx=5, pady=5)
        entry_widgets[section][key] = entry

    # Update the scroll region after content is added
    content_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))


# Add Save button
save_button = ttk.Button(root, text="Save", command=save_changes)
save_button.pack(pady=10)

# Start the tkinter main loop
root.mainloop()
