import os
import sys
import tkinter as tk
from tkinter import messagebox

from data_manager import load_and_clean
from analysis import compute_analytics
from recommendation_engine import get_recommendations, get_warnings
from gui import launch

def main():
    BASE_DIR = r'C:\Users\Sanu\Desktop\Sanjog\Advanced Python\Project\SmartSpend'
    CSV_PATH = os.path.join(BASE_DIR, 'cleaned_transactions.csv')

    try:
        df_user, df_full, top_cc = load_and_clean(CSV_PATH)
    except FileNotFoundError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(e))
        root.destroy()
        sys.exit(1)

    stats = compute_analytics(df_user)
    recommendations = get_recommendations(df_user, stats)
    warnings = get_warnings(df_user, stats)

    root = launch(df_user, df_full, top_cc, stats, recommendations, warnings)
    root.mainloop()

if __name__ == "__main__":
    main()