import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_column(df, col, include_zeros):
    data = df[col].dropna()

    # Optionally exclude zeros
    if not include_zeros:
        data = data[data != 0]

    if data.empty:
        print("⚠️ No non-zero data available in this column.")
        return

    Q1 = np.percentile(data, 25)
    Q2 = np.median(data)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    data_min = data.min()
    data_max = data.max()

    # Print statistics
    print(f"\n📊 Analyzing Column: {col}")
    print(f"  ➤ Q1: {Q1:.2f}")
    print(f"  ➤ Q2 (Median): {Q2:.2f}")
    print(f"  ➤ Q3: {Q3:.2f}")
    print(f"  ➤ IQR: {IQR:.2f}")
    print(f"  ➤ Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
    print(f"  ✅ Min Value: {data_min}, Max Value: {data_max}")
    print(f"  ❗ Outliers: {outliers.values if not outliers.empty else 'None'}")

    # Box Plot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data, color='lightblue', flierprops=dict(marker='o', color='red', markersize=6))
    plt.title(f"Box Plot: {col}")
    plt.xlabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    file_path = input("📂 Enter full path to your file (.xlsx, .xls, or .csv): ").strip()

    if not os.path.isfile(file_path):
        print("❌ File does not exist. Please check the path.")
        return

    try:
        # Auto-detect file type
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            print("❌ Unsupported file format. Use .csv, .xls, or .xlsx")
            return

        print(f"✅ File loaded: {file_path}")
        numeric_cols = df.select_dtypes(include=np.number).columns

        if numeric_cols.empty:
            print("⚠️ No numeric columns found in the file.")
            return

        print("\n📌 Available Numeric Columns:")
        for i, col in enumerate(numeric_cols):
            print(f"  {i}: {col}")

        col_index = input("\n🔎 Enter the column number you want to analyze: ")

        try:
            col_index = int(col_index)
            if 0 <= col_index < len(numeric_cols):
                selected_col = numeric_cols[col_index]
                zero_choice = input(f"\nInclude zero values in '{selected_col}'? (yes/no): ").strip().lower()
                include_zeros = zero_choice == 'yes'
                analyze_column(df, selected_col, include_zeros)
            else:
                print("❌ Invalid column number.")
        except ValueError:
            print("❌ Please enter a valid number.")

    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    main()
