import pandas as pd

def compute_analytics(df):
    """
    Compute analytics from transaction data.

    Args:
        df (pd.DataFrame): Cleaned transaction data.

    Returns:
        dict: Statistics dictionary with various metrics.
    """
    # Basic stats
    total_spent = df['amt'].sum()
    total_txns = len(df)
    avg_txn = df['amt'].mean()
    avg_daily = total_spent / ((df['trans_date_trans_time'].max() - df['trans_date_trans_time'].min()).days + 1)
    avg_monthly = df.groupby(df['trans_date_trans_time'].dt.to_period('M'))['amt'].sum().mean()

    # Top category
    top_category = df.groupby('category')['amt'].sum().idxmax()

    # Fraud count (assuming is_fraud column exists)
    fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0

    # Category spending
    category_spending = df.groupby('category')['amt'].sum().to_dict()

    # Monthly spending
    monthly_spending = df.groupby(df['trans_date_trans_time'].dt.to_period('M'))['amt'].sum().to_dict()

    # Day of week spending
    dow_spending = df.groupby('day_of_week')['amt'].sum().to_dict()

    # Hourly spending
    hourly_spending = df.groupby('hour')['amt'].sum().to_dict()

    # Last 3 months average
    last_3_months = df[df['trans_date_trans_time'] >= df['trans_date_trans_time'].max() - pd.DateOffset(months=3)]
    last_3_avg = last_3_months['amt'].sum() / 3 if len(last_3_months) > 0 else 0

    stats = {
        'total_spent': total_spent,
        'total_txns': total_txns,
        'avg_txn': avg_txn,
        'avg_daily': avg_daily,
        'avg_monthly': avg_monthly,
        'top_category': top_category,
        'fraud_count': fraud_count,
        'category_spending': category_spending,
        'monthly_spending': monthly_spending,
        'dow_spending': dow_spending,
        'hourly_spending': hourly_spending,
        'last_3_avg': last_3_avg
    }

    # Print summary
    print("SmartSpend Analytics Summary")
    print("=" * 30)
    print(f"Total Spent: ${total_spent:,.2f}")
    print(f"Total Transactions: {total_txns}")
    print(f"Average Transaction: ${avg_txn:.2f}")
    print(f"Daily Average: ${avg_daily:.2f}")
    print(f"Monthly Average: ${avg_monthly:.2f}")
    print(f"Top Category: {top_category}")
    print(f"Fraud Count: {fraud_count}")

    return stats