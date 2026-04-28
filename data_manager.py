import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_and_clean(csv_path='credit_card_transactions.csv'):
    """
    Load and clean transaction data from CSV file.

    Args:
        csv_path (str): Path to the CSV file containing transaction data.

    Returns:
        tuple: (df_user, df_full, top_cc) where df_user is filtered to single user,
               df_full is the full cleaned dataset, and top_cc is the credit card number.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Transaction data file not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Parse trans_date_trans_time as datetime, drop NaT
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df = df.dropna(subset=['trans_date_trans_time'])

    # Filter amt <= 0, drop duplicates
    df = df[df['amt'] > 0].drop_duplicates()

    # Engineer features on full dataset
    df['month_num'] = df['trans_date_trans_time'].dt.month
    df['year'] = df['trans_date_trans_time'].dt.year
    df['quarter'] = df['trans_date_trans_time'].dt.quarter
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['amt_log'] = np.log1p(df['amt'])

    df_full = df.copy()

    # Filter to single user with most transactions
    top_cc = df['cc_num'].value_counts().index[0]
    df_user = df[df['cc_num'] == top_cc].copy()

    # Engineer on df_user only
    df_user['month_str'] = df_user['trans_date_trans_time'].dt.strftime('%b %Y')

    # Save df_user to cleaned_transactions.csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # Ensure directory exists
    df_user.to_csv(csv_path, index=False)

    # Print row counts and date range
    print(f"Full dataset: {len(df_full)} rows, date range: {df_full['trans_date_trans_time'].min()} to {df_full['trans_date_trans_time'].max()}")
    print(f"User dataset: {len(df_user)} rows, date range: {df_user['trans_date_trans_time'].min()} to {df_user['trans_date_trans_time'].max()}")

    return df_user, df_full, top_cc