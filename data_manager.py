import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_and_clean_data(file_path='credit_card_transactions.csv'):
    """
    Loads the credit card transaction data, cleans it up,
    adds some useful columns, and saves the result to a new file.
    Returns the clean data and a summary of what was done.
    """

    # make sure the file actually exists before trying to open it
    if not os.path.exists(file_path):
        print(f"Error: Could not find the file '{file_path}'.")
        exit(1)

    # load the csv into a dataframe
    df = pd.read_csv(file_path)

    # print a quick overview of what we loaded
    print("---- Dataset Overview ----")
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    print("Columns and Data Types:")
    for col in df.columns:
        print(f"  {col} : {df[col].dtype}")
    print("Completeness Summary:")
    for col in df.columns:
        filled_pct = (1 - df[col].isnull().sum() / len(df)) * 100
        print(f"  {col} : {filled_pct:.1f}% populated")
    print()

    # remove transactions with amounts that dont make sense
    # anything below a penny or above 10 thousand is probably bad data
    too_low = df[df['amt'] < 0.01]
    too_high = df[df['amt'] > 10000]
    bad_amounts = pd.concat([too_low, too_high])
    num_bad_amounts = len(bad_amounts)

    print(f"Found {num_bad_amounts} transactions with unrealistic amounts.")
    if num_bad_amounts > 0:
        print("Sample of removed rows:")
        print(bad_amounts.head(5)[['amt', 'category', 'merchant']].to_string())
    print()

    # keep only rows where amount is between $0.01 and $10,000
    df = df[(df['amt'] >= 0.01) & (df['amt'] <= 10000)]
    print(f"Removed {num_bad_amounts} unrealistic transaction records.")
    print()

    # save the starting count so we can report how much was removed later
    start_count = len(df)
    print(f"Records before cleaning: {start_count}")

    # show which columns have missing values before we fix them
    print("Missing values per column before imputation:")
    found_missing = False
    for col in df.columns:
        num_missing = df[col].isnull().sum()
        if num_missing > 0:
            print(f"  {col} : {num_missing}")
            found_missing = True
    if not found_missing:
        print("  No missing values found.")

    # drop rows where the date is missing since we cant guess what date it was
    df = df.dropna(subset=['trans_date_trans_time'])
    print(f"Dropped {start_count - len(df)} rows where date was missing.")

    # fill missing amounts with the average amount
    num_missing_amt = df['amt'].isnull().sum()
    if num_missing_amt > 0:
        avg_amt = df['amt'].mean()
        df['amt'] = df['amt'].fillna(avg_amt)
        print(f"Amount: filled {num_missing_amt} missing values with mean ${avg_amt:.2f}")

    # fill missing text columns with the most common value in that column
    text_cols_to_fill = [
        'category', 'merchant', 'first', 'last',
        'gender', 'street', 'city', 'state', 'job', 'trans_num'
    ]
    for col in text_cols_to_fill:
        if col in df.columns:
            num_missing_col = df[col].isnull().sum()
            if num_missing_col > 0:
                most_common_val = df[col].mode()[0]
                df[col] = df[col].fillna(most_common_val)
                print(f"{col}: filled {num_missing_col} missing values with mode '{most_common_val}'")

    # fill missing number columns with their average
    # skip columns where filling with average wouldnt make sense
    cols_to_skip = [
        'amt', 'Unnamed: 0', 'cc_num', 'zip',
        'lat', 'long', 'city_pop', 'unix_time',
        'merch_lat', 'merch_long', 'is_fraud'
    ]
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col not in cols_to_skip:
            num_missing_col = df[col].isnull().sum()
            if num_missing_col > 0:
                avg_val = df[col].mean()
                df[col] = df[col].fillna(avg_val)
                print(f"{col}: filled {num_missing_col} missing values with mean {avg_val:.2f}")

    # remove exact duplicate rows
    num_dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate records removed: {num_dupes}")

    # remove any rows where amount ended up zero or negative
    num_bad_amt = len(df[df['amt'] <= 0])
    df = df[df['amt'] > 0]
    print(f"Invalid amount records removed: {num_bad_amt}")

    # trim spaces from all text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # make category names look consistent like "Grocery_Pos" instead of "grocery_pos"
    if 'category' in df.columns:
        df['category'] = df['category'].str.title()

    # convert the date column from a string to an actual datetime object
    df['date'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    num_bad_dates = df['date'].isnull().sum()
    df = df.dropna(subset=['date'])
    print(f"Dropped {num_bad_dates} rows with dates that couldnt be converted.")

    # print a summary of everything that was cleaned
    print()
    print("---- Cleaning Summary ----")
    print(f"Records before cleaning: {start_count}")
    print(f"Duplicate records removed: {num_dupes}")
    print(f"Invalid amount records removed: {num_bad_amt}")
    print(f"Unrealistic amount records removed: {num_bad_amounts}")
    print(f"Final valid records: {len(df)}")
    print()

    # add month column like "2019-01"
    df['Month'] = df['date'].dt.strftime('%Y-%m')

    # add year as a number
    df['Year'] = df['date'].dt.year

    # add the day name like Monday, Tuesday etc
    df['DayOfWeek'] = df['date'].dt.day_name()

    # mark whether the transaction happened on a weekend
    df['IsWeekend'] = df['date'].dt.weekday >= 5

    # calculate some stats about the new columns
    month_start = df['Month'].min()
    month_end = df['Month'].max()
    year_start = df['Year'].min()
    year_end = df['Year'].max()
    weekend_pct = (df['IsWeekend'].sum() / len(df)) * 100
    weekday_pct = 100 - weekend_pct

    print("---- Feature Engineering Summary ----")
    print(f"Month column added: range {month_start} to {month_end}")
    print(f"Year column added: range {year_start} to {year_end}")
    print("DayOfWeek column added")
    print(f"IsWeekend column added: {weekend_pct:.1f}% weekend, {weekday_pct:.1f}% weekday")
    print(f"Total features now: {len(df.columns)}")
    print()

    # show basic stats about the amount column after everything is cleaned
    print("---- Amount Column Statistics ----")
    print(f"Min: ${df['amt'].min():.2f}")
    print(f"Max: ${df['amt'].max():.2f}")
    print(f"Mean: ${df['amt'].mean():.2f}")
    print(f"Std Dev: ${df['amt'].std():.2f}")
    print()

    # save the cleaned version to a new file
    df.to_csv('cleaned_transactions.csv', index=False)
    print("Cleaned data saved to cleaned_transactions.csv")

    # put all the cleaning stats into a dictionary so main.py can use them
    clean_summary = {
        'records_before': start_count,
        'duplicates_removed': num_dupes,
        'invalid_amt_removed': num_bad_amt,
        'unrealistic_amt_removed': num_bad_amounts,
        'final_records': len(df),
        'amount_stats': {
            'min': df['amt'].min(),
            'max': df['amt'].max(),
            'mean': df['amt'].mean(),
            'std': df['amt'].std()
        },
        'weekend_pct': weekend_pct,
        'weekday_pct': weekday_pct,
        'month_range': f"{month_start} to {month_end}",
        'year_range': f"{year_start} to {year_end}"
    }

    return df, clean_summary