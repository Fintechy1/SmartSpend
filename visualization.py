import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_all_charts(df):
    """
    Makes all the charts for the SmartSpend dashboard and saves
    them as PNG files in the charts/ folder.
    Returns a list of the file paths that were saved.
    """

    # create the charts folder if it doesnt exist yet
    os.makedirs('charts', exist_ok=True)

    saved_files = []

    # ---- Chart 1: How much was spent in each category ----
    plt.figure(figsize=(10, 6))

    # add up all spending per category and sort biggest to smallest
    spending_by_cat = df.groupby('category')['amt'].sum().sort_values(ascending=False)

    bars = plt.bar(spending_by_cat.index, spending_by_cat.values, color='steelblue')
    plt.title('Total Spending by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Amount Spent')
    plt.xticks(rotation=45, ha='right')

    # put the dollar amount on top of each bar
    for bar, amount in zip(bars, spending_by_cat.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'${amount:,.0f}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig('charts/category_bar.png')
    plt.close()
    print("Saved: charts/category_bar.png")
    saved_files.append('charts/category_bar.png')

    # ---- Chart 2: Pie chart showing what percentage each category is ----
    plt.figure(figsize=(8, 8))

    # any category that is less than 3% of total gets lumped into Other
    # this keeps the pie chart from getting too crowded
    total_spend = spending_by_cat.sum()
    three_pct_cutoff = total_spend * 0.03
    big_cats = spending_by_cat[spending_by_cat >= three_pct_cutoff].copy()
    small_cats = spending_by_cat[spending_by_cat < three_pct_cutoff]

    if len(small_cats) > 0:
        big_cats['Other'] = small_cats.sum()

    plt.pie(
        big_cats.values,
        labels=big_cats.index,
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title('Spending Distribution by Category')
    plt.axis('equal')
    plt.savefig('charts/category_pie.png')
    plt.close()
    print("Saved: charts/category_pie.png")
    saved_files.append('charts/category_pie.png')

    # ---- Chart 3: How spending changed month by month ----
    plt.figure(figsize=(10, 6))

    # group by month and sort so the line goes left to right in order
    spending_by_month = df.groupby('Month')['amt'].sum().sort_index()

    plt.plot(spending_by_month.index, spending_by_month.values, marker='o', color='steelblue')
    plt.title('Monthly Spending Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Amount Spent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('charts/monthly_trend.png')
    plt.close()
    print("Saved: charts/monthly_trend.png")
    saved_files.append('charts/monthly_trend.png')

    # ---- Chart 4: Which day of the week has the most spending ----
    plt.figure(figsize=(10, 6))

    # reindex so days go Monday through Sunday instead of alphabetical
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    spending_by_day = df.groupby('DayOfWeek')['amt'].sum().reindex(day_order)

    bars = plt.bar(spending_by_day.index, spending_by_day.values, color='mediumseagreen')
    plt.title('Spending by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Amount Spent')

    # put the dollar amount on top of each bar
    for bar, amount in zip(bars, spending_by_day.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'${amount:,.0f}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig('charts/day_of_week.png')
    plt.close()
    print("Saved: charts/day_of_week.png")
    saved_files.append('charts/day_of_week.png')

    # ---- Chart 5: Top 10 merchants by how much was spent ----
    # only make this chart if the merchant column actually exists
    if 'merchant' in df.columns:
        plt.figure(figsize=(10, 8))

        # get the top 10 merchants by total spending
        top_merchants = df.groupby('merchant')['amt'].sum().sort_values(ascending=False).head(10)

        # flip it so the biggest bar is at the top
        top_merchants = top_merchants.sort_values(ascending=True)

        plt.barh(top_merchants.index, top_merchants.values, color='coral')
        plt.title('Top 10 Merchants by Total Spending')
        plt.xlabel('Total Amount Spent')
        plt.ylabel('Merchant')
        plt.tight_layout()
        plt.savefig('charts/top_merchants.png')
        plt.close()
        print("Saved: charts/top_merchants.png")
        saved_files.append('charts/top_merchants.png')
    else:
        print("Merchant column not found. Skipping Chart 5.")

    # ---- Chart 6: Spending every single day over the whole time period ----
    plt.figure(figsize=(12, 6))

    # group by date to get total spent each day
    spending_per_day = df.groupby('date')['amt'].sum()

    plt.plot(spending_per_day.index, spending_per_day.values, color='steelblue', linewidth=0.8)

    # add a dashed line showing the average so its easy to see above and below average days
    avg_daily_spend = spending_per_day.mean()
    plt.axhline(
        y=avg_daily_spend,
        color='red',
        linestyle='--',
        label=f'Average Daily Spend: ${avg_daily_spend:,.2f}'
    )

    plt.title('Daily Spending Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Amount Spent per Day')
    plt.legend()
    plt.tight_layout()
    plt.savefig('charts/daily_spending.png')
    plt.close()
    print("Saved: charts/daily_spending.png")
    saved_files.append('charts/daily_spending.png')

    return saved_files