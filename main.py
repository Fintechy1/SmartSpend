import os
from data_manager import load_and_clean_data
from analysis import run_eda
from visualization import generate_all_charts

def main():
    """
    Main entry point for the SmartSpend application Phase 1.
    Loads and cleans data, performs EDA, generates charts, and creates summary.
    """
    # load and clean the data
    clean_data, clean_info = load_and_clean_data()
    # run analysis
    analysis_results = run_eda(clean_data)
    # make the charts
    chart_list = generate_all_charts(clean_data)

    # write the summary file
    with open('phase1_summary.txt', 'w') as summary_file:
        summary_file.write("SmartSpend – Phase 1 Summary\n")
        summary_file.write("=============================\n")
        summary_file.write("\n")
        summary_file.write("Dataset Overview\n")
        summary_file.write("----------------\n")
        summary_file.write(f"The dataset was loaded from credit_card_transactions.csv and consists of {analysis_results['summary']['total_transactions']} records and {len(clean_data.columns)} features. All columns were reviewed for completeness and data quality.\n")
        summary_file.write("\n")
        summary_file.write("Data Cleaning\n")
        summary_file.write("-------------\n")
        summary_file.write(f"During the cleaning process, {clean_info['duplicates_removed']} duplicate records were removed and {clean_info['invalid_amt_removed']} invalid amount records were filtered out. {clean_info['unrealistic_amt_removed']} unrealistic amount records were removed (amounts below $0.01 or above $10,000). Missing values were imputed using the mean for numeric columns and the mode for categorical columns. The final cleaned dataset contains {clean_info['final_records']} valid records.\n")
        summary_file.write("\n")
        summary_file.write("Feature Engineering\n")
        summary_file.write("-------------------\n")
        summary_file.write(f"The following features were added: Month, Year, DayOfWeek, and IsWeekend. The data spans from {clean_info['month_range']}. Approximately {clean_info['weekend_pct']:.1f}% of transactions occurred on weekends and {clean_info['weekday_pct']:.1f}% occurred on weekdays.\n")
        summary_file.write("\n")
        summary_file.write("Amount Statistics\n")
        summary_file.write("-----------------\n")
        summary_file.write(f"After cleaning, the transaction amount column has the following distribution: Minimum: ${clean_info['amount_stats']['min']:.2f}, Maximum: ${clean_info['amount_stats']['max']:.2f}, Mean: ${clean_info['amount_stats']['mean']:.2f}, Standard Deviation: ${clean_info['amount_stats']['std']:.2f}.\n")
        summary_file.write("\n")
        summary_file.write("Exploratory Data Analysis\n")
        summary_file.write("-------------------------\n")
        summary_file.write(f"The dataset contains {analysis_results['summary']['total_transactions']} total transactions with a total spend of ${analysis_results['summary']['total_amount']:.2f}. The average transaction amount is ${analysis_results['summary']['avg_transaction']:.2f}. The data covers the period from {analysis_results['summary']['date_range']}.\n")
        # find the top category
        top_cat_name = list(analysis_results['category_analysis']['total_spent'].keys())[0]
        top_cat_amt = list(analysis_results['category_analysis']['total_spent'].values())[0]
        summary_file.write(f"The top spending category is {top_cat_name} with a total of ${top_cat_amt:.2f}.\n")
        summary_file.write(f"The highest spending month was {analysis_results['monthly_trends']['highest_month']} at ${analysis_results['monthly_trends']['monthly_spending'][analysis_results['monthly_trends']['highest_month']]:.2f} and the lowest was {analysis_results['monthly_trends']['lowest_month']} at ${analysis_results['monthly_trends']['monthly_spending'][analysis_results['monthly_trends']['lowest_month']]:.2f}.\n")
        summary_file.write(f"The average monthly spending is ${analysis_results['monthly_trends']['avg_monthly']:.2f}.\n")
        summary_file.write(f"The estimated daily burn rate is ${analysis_results['burn_rate']['daily_burn_rate']:.2f} and the estimated monthly burn rate is ${analysis_results['burn_rate']['monthly_burn_rate']:.2f}.\n")
        summary_file.write(f"Weekend transactions have a higher average spend than weekday transactions: {analysis_results['daily_weekly']['higher_weekend']}.\n")
        summary_file.write("\n")
        summary_file.write("Visualizations Generated\n")
        summary_file.write("------------------------\n")
        summary_file.write("The following charts were saved to the charts/ folder:\n")
        for chart_file in chart_list:
            summary_file.write(f"- {os.path.basename(chart_file)}\n")

    print("Phase 1 complete. Cleaned data saved to cleaned_transactions.csv. All charts saved to the charts/ folder. Summary written to phase1_summary.txt.")

if __name__ == "__main__":
    main()