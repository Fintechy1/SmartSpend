import os
from datetime import datetime

def save_report(df, stats, recommendations, warnings, fraud_results=None, filepath=None):
    """
    Generate and save a comprehensive report.

    Args:
        df (pd.DataFrame): Transaction data.
        stats (dict): Statistics from compute_analytics.
        recommendations (dict): Budget recommendations.
        warnings (dict): Spending warnings.
        fraud_results (dict, optional): Fraud detection results.
        filepath (str, optional): Custom filepath.

    Returns:
        str: Path to saved report.
    """
    if filepath is None:
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d')
        filepath = f'reports/report_{timestamp}.txt'

    with open(filepath, 'w') as f:
        # Header
        f.write("SmartSpend Financial Report\n")
        f.write("=" * 40 + "\n\n")

        # Summary
        f.write("Summary\n")
        f.write("-" * 10 + "\n")
        f.write(f"Total Transactions: {stats['total_txns']}\n")
        f.write(f"Total Spent: ${stats['total_spent']:,.2f}\n")
        f.write(f"Average Transaction: ${stats['avg_txn']:.2f}\n")
        f.write(f"Daily Average: ${stats['avg_daily']:.2f}\n")
        f.write(f"Monthly Average: ${stats['avg_monthly']:.2f}\n")
        f.write(f"Top Category: {stats['top_category']}\n")
        f.write(f"Fraud Count: {stats['fraud_count']}\n\n")

        # Category Breakdown
        f.write("Category Breakdown\n")
        f.write("-" * 20 + "\n")
        for cat, amt in sorted(stats['category_spending'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"{cat}: ${amt:,.2f}\n")
        f.write("\n")

        # Monthly Trends
        f.write("Monthly Trends\n")
        f.write("-" * 15 + "\n")
        for period, amt in stats['monthly_spending'].items():
            f.write(f"{period}: ${amt:,.2f}\n")
        f.write("\n")

        # Budget Recommendations
        f.write("Budget Recommendations\n")
        f.write("-" * 25 + "\n")
        if recommendations:
            for cat, rec in recommendations.items():
                f.write(f"{cat}: ${rec:,.2f} recommended\n")
        else:
            f.write("No recommendations available (insufficient data)\n")
        f.write("\n")

        # Warnings
        f.write("Warnings\n")
        f.write("-" * 10 + "\n")
        if warnings:
            for cat, warning in warnings.items():
                f.write(f"{cat}: {warning}\n")
        else:
            f.write("No warnings\n")
        f.write("\n")

        # Fraud Detection Results
        if fraud_results:
            f.write("Fraud Detection Results\n")
            f.write("-" * 25 + "\n")

            # Best model
            best_model = max(fraud_results.keys(), key=lambda x: fraud_results[x]['auc'])
            f.write(f"Best Model: {best_model}\n")
            f.write(f"AUC: {fraud_results[best_model]['auc']:.3f}\n")
            f.write(f"F1 Score: {fraud_results[best_model]['f1']:.3f}\n\n")

            # Flagged transactions
            flagged = df[df['fraud_predicted'] == 1]
            if len(flagged) > 0:
                f.write("Flagged Transactions:\n")
                for _, row in flagged.iterrows():
                    f.write(f"  {row['trans_date_trans_time'].date()} - {row['merchant']} - ${row['amt']:.2f} - Prob: {row['fraud_prob']:.3f}\n")
            else:
                f.write("No transactions flagged as fraudulent\n")
            f.write("\n")

    print(f"Report saved to: {filepath}")
    return filepath