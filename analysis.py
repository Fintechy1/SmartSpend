import pandas as pd
from data_manager import load_and_clean_data

def run_eda(df):
    """
    Goes through the cleaned transaction data and pulls out
    useful stats and patterns. Returns everything in a dictionary
    so main.py can use it to write the summary file.
    """
    results = {}

    # count up the basic numbers we need for the summary
    print("---- Summary Statistics ----")
    num_transactions = len(df)
    total_spent = df['amt'].sum()
    avg_amount = df['amt'].mean()
    min_amount = df['amt'].min()
    max_amount = df['amt'].max()
    first_date = df['date'].min().date()
    last_date = df['date'].max().date()
    date_range = f"{first_date} to {last_date}"

    print(f"Total Transactions: {num_transactions:,}")
    print(f"Total Amount Spent: ${total_spent:,.2f}")
    print(f"Average Transaction Amount: ${avg_amount:.2f}")
    print(f"Minimum Transaction: ${min_amount:.2f}")
    print(f"Maximum Transaction: ${max_amount:.2f}")
    print(f"Date Range: {date_range}")

    # save these to the results dict so we can use them later
    results['summary'] = {
        'total_transactions': num_transactions,
        'total_amount': total_spent,
        'avg_transaction': avg_amount,
        'min_transaction': min_amount,
        'max_transaction': max_amount,
        'date_range': date_range
    }
    print()

    # group transactions by category and add up the totals
    print("---- Spending by Category ----")
    category_totals = df.groupby('category').agg(
        total_spent=('amt', 'sum'),
        num_transactions=('amt', 'count'),
        avg_per_transaction=('amt', 'mean')
    ).sort_values('total_spent', ascending=False)

    # print a simple table showing each category
    print(f"{'Category':<20} {'Total Spent':>12} {'Transactions':>14} {'Avg per Transaction':>20}")
    print("-" * 70)
    for cat, row in category_totals.iterrows():
        print(f"{cat:<20} ${row['total_spent']:>11,.2f} {int(row['num_transactions']):>14,} ${row['avg_per_transaction']:>19.2f}")

    # show just the top 3 categories
    top_three = category_totals.head(3)
    print("\nTop 3 Categories by Total Spending:")
    for i, (cat, row) in enumerate(top_three.iterrows(), 1):
        print(f"{i}. {cat}: ${row['total_spent']:,.2f}")

    results['category_analysis'] = {
        'total_spent': category_totals['total_spent'].to_dict(),
        'num_transactions': category_totals['num_transactions'].to_dict(),
        'avg_per_transaction': category_totals['avg_per_transaction'].to_dict()
    }
    print()

    # add up spending for each month and sort by date order
    print("---- Monthly Spending Trends ----")
    monthly_totals = df.groupby('Month')['amt'].sum().sort_index()

    print(f"{'Month':<12} {'Total Spent':>12}")
    print("-" * 26)
    for month, amount in monthly_totals.items():
        print(f"{month:<12} ${amount:>11,.2f}")

    # find the best and worst months
    highest_month = monthly_totals.idxmax()
    lowest_month = monthly_totals.idxmin()
    avg_monthly_spend = monthly_totals.mean()

    print(f"\nHighest Spending Month: {highest_month} (${monthly_totals[highest_month]:,.2f})")
    print(f"Lowest Spending Month: {lowest_month} (${monthly_totals[lowest_month]:,.2f})")
    print(f"Average Monthly Spending: ${avg_monthly_spend:,.2f}")

    results['monthly_trends'] = {
        'monthly_spending': monthly_totals.to_dict(),
        'highest_month': highest_month,
        'lowest_month': lowest_month,
        'avg_monthly': avg_monthly_spend
    }
    print()

    # look at which days of the week have the most spending
    print("---- Spending by Day of Week ----")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    spending_by_day = df.groupby('DayOfWeek')['amt'].sum().reindex(day_order)

    print(f"{'Day':<12} {'Total Spent':>12}")
    print("-" * 26)
    for day, amount in spending_by_day.items():
        print(f"{day:<12} ${amount:>11,.2f}")

    # compare weekend vs weekday average transaction size
    weekend_avg = df[df['IsWeekend']]['amt'].mean()
    weekday_avg = df[~df['IsWeekend']]['amt'].mean()

    print(f"\nWeekend vs Weekday:")
    print(f"Average Weekend Transaction: ${weekend_avg:.2f}")
    print(f"Average Weekday Transaction: ${weekday_avg:.2f}")

    # simple yes or no answer
    weekend_higher = "Yes" if weekend_avg > weekday_avg else "No"
    print(f"Weekends have higher average spending: {weekend_higher}")

    results['daily_weekly'] = {
        'spending_by_day': spending_by_day.to_dict(),
        'weekend_avg': weekend_avg,
        'weekday_avg': weekday_avg,
        'higher_weekend': weekend_higher
    }
    print()

    # figure out how much money is being spent per day on average
    print("---- Key Financial Metrics ----")
    num_days = (df['date'].max() - df['date'].min()).days + 1
    daily_burn_rate = total_spent / num_days
    monthly_burn_rate = daily_burn_rate * 30

    print(f"Daily Average Burn Rate: ${daily_burn_rate:,.2f}")
    print(f"Estimated Monthly Burn Rate: ${monthly_burn_rate:,.2f}")
    print("Note: Income data not available. Savings rate cannot be calculated.")

    results['burn_rate'] = {
        'daily_burn_rate': daily_burn_rate,
        'monthly_burn_rate': monthly_burn_rate
    }

    return results