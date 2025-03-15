import pandas as pd
import os
import re

def merge_pv_files(input_folder, output_file):
    all_data = []
    
    for file in os.listdir(input_folder):
        if file.endswith(".xlsx"):
            file_path = os.path.join(input_folder, file)
            df = pd.read_excel(file_path)
            
            # Extract year and month from file name
            match = re.search(r"(\d{4})-(\d{2})", file)
            if not match:
                print(f"Name error: {file}")
                continue
            year, month = match.groups()
            
            # Check if the file contains the required columns
            if 'Time' in df.columns and 'PV Energy(kWh)' in df.columns:
                df = df[['Time', 'PV Energy(kWh)']].copy()
                
                # Convert Time to datetime
                df['date'] = pd.to_datetime(df['Time'].astype(str) + f'-{month}-{year}', format='%d-%m-%Y', errors='coerce')
                df.drop(columns=['Time'], inplace=True)
                df.rename(columns={'PV Energy(kWh)': 'pv_energy'}, inplace=True)
                
                # Remove rows with missing date
                df = df.dropna(subset=['date'])

                # Order columns
                df = df[['date', 'pv_energy']]
                
                all_data.append(df)
            else:
                print(f"Column error: {file}")
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)

        # Sort by date
        merged_df = merged_df.sort_values(by='date')

        merged_df.to_csv(output_file, index=False)
        print(f"File saved: {output_file}")
    else:
        print("No data to merge")

# Main
test = False

if not test:
    input_folder = "./data/raw/pv"
    output_file = "./data/raw/pv_merged.csv"
else:
    input_folder = "./data/raw/pv_test"
    output_file = "./data/raw/pv_merged_test.csv"
    
merge_pv_files(input_folder, output_file)