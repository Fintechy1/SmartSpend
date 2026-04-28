import pandas as pd

def get_recommendations(df, stats):
    """
    Generate budget recommendations based on spending data.

    Args:
        df (pd.DataFrame): Transaction data.
        stats (dict): Statistics from compute_analytics.

    Returns:
        dict or None: Category budget recommendations or None if insufficient data.
    """
    if stats['total_txns'] < 10:
        print("Warning: Insufficient transaction data for recommendations.")
        return None

    # Recommended budget per category = avg monthly spend * 1.10
    recommendations = {}
    for category, spend in stats['category_spending'].items():
        # Calculate monthly average for category
        category_monthly = df[df['category'] == category].groupby(df['trans_date_trans_time'].dt.to_period('M'))['amt'].sum()
        avg_monthly = category_monthly.mean() if len(category_monthly) > 0 else 0
        recommendations[category] = avg_monthly * 1.10

    return recommendations

def get_warnings(df, stats):
    """
    Generate spending warnings for categories exceeding thresholds.

    Args:
        df (pd.DataFrame): Transaction data.
        stats (dict): Statistics from compute_analytics.

    Returns:
        dict: Categories with overspending alerts.
    """
    warnings = {}

    # Get most recent month
    recent_month = df['trans_date_trans_time'].max().to_period('M')

    for category in stats['category_spending'].keys():
        category_data = df[df['category'] == category]
        monthly_spend = category_data.groupby(category_data['trans_date_trans_time'].dt.to_period('M'))['amt'].sum()

        if len(monthly_spend) > 1:  # Need at least 2 months for comparison
            avg_monthly = monthly_spend[:-1].mean()  # Exclude most recent
            recent_spend = monthly_spend.get(recent_month, 0)

            if recent_spend > avg_monthly * 1.20:
                warnings[category] = 'Overspending alert'

    return warnings