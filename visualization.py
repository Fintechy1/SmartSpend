import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import os

COLORS = {
    'bg':      '#F0F2F5',
    'panel':   '#FFFFFF',
    'accent1': '#1D4ED8',
    'accent2': '#6D28D9',
    'accent3': '#047857',
    'accent4': '#B45309',
    'accent5': '#B91C1C',
    'text':    '#0F172A',
    'subtext': '#475569',
    'grid':    '#CBD5E1',
}

CATEGORY_COLORS = [
    '#1D4ED8','#6D28D9','#047857','#B45309','#B91C1C',
    '#BE185D','#0369A1','#4D7C0F','#C2410C','#0E7490',
    '#7E22CE','#15803D','#A16207','#4338CA'
]


def setup_style():
    """Set global matplotlib rcParams for light theme."""
    plt.rcParams.update({
        'figure.facecolor':  COLORS['bg'],
        'axes.facecolor':    COLORS['panel'],
        'text.color':        COLORS['text'],
        'axes.labelcolor':   COLORS['subtext'],
        'xtick.color':       COLORS['subtext'],
        'ytick.color':       COLORS['subtext'],
        'axes.edgecolor':    COLORS['grid'],
        'grid.color':        COLORS['grid'],
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'axes.titlecolor':   COLORS['text'],
        'axes.titleweight':  'bold',
        'axes.titlesize':    12,
        'font.size':         10,
        'legend.facecolor':  COLORS['panel'],
        'legend.edgecolor':  COLORS['grid'],
        'legend.labelcolor': COLORS['text'],
    })


def style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply light theme styling to an axes object."""
    ax.set_facecolor(COLORS['panel'])
    ax.grid(True, axis='y', alpha=0.5, color=COLORS['grid'], linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['subtext'], labelcolor=COLORS['subtext'])
    if title:
        ax.set_title(title, color=COLORS['text'], fontweight='bold',
                     fontsize=12, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS['subtext'], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS['subtext'], fontsize=9)


def dollar_fmt(x, pos):
    """Format axis value as dollars."""
    return f'${x:,.0f}'


def dollar_fmt_k(x, pos):
    """Format axis value as dollars in thousands."""
    return f'${x/1000:,.0f}k'


def _ensure_charts_dir(path):
    """Create parent directory if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ── Milestone 3 Charts ─────────────────────────────────────────

def plot_category_bar(df, save_path='charts/category_bar.png'):
    """Bar chart of total spending per category."""
    _ensure_charts_dir(save_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax, 'Total Spending by Category', 'Category', 'Total Spent')

    cat = df.groupby('category')['amt'].sum().sort_values(ascending=False)
    bars = ax.bar(cat.index, cat.values,
                  color=CATEGORY_COLORS[:len(cat)],
                  edgecolor='#FFFFFF', linewidth=1)
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt_k))
    plt.xticks(rotation=45, ha='right', color=COLORS['subtext'])

    for bar, val in zip(bars, cat.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + cat.max() * 0.01,
                f'${val:,.0f}', ha='center', va='bottom',
                fontsize=7, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f'✅ Saved: {save_path}')


def plot_category_pie(df, save_path='charts/category_pie.png'):
    """Pie chart of spending distribution by category."""
    _ensure_charts_dir(save_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['panel'])

    cat = df.groupby('category')['amt'].sum().sort_values(ascending=False)
    total = cat.sum()
    big = cat[cat >= total * 0.03].copy()
    small = cat[cat < total * 0.03]
    if len(small):
        big['Other'] = small.sum()

    wedges, texts, autotexts = ax.pie(
        big.values, labels=big.index,
        autopct=lambda p: f'{p:.1f}%' if p > 4 else '',
        startangle=140,
        colors=CATEGORY_COLORS[:len(big)],
        wedgeprops={'edgecolor': '#FFFFFF', 'linewidth': 1.5}
    )
    plt.setp(texts, color=COLORS['text'], fontsize=9)
    plt.setp(autotexts, color='#FFFFFF', fontsize=8, fontweight='bold')
    ax.set_title('Spending Distribution by Category',
                 color=COLORS['text'], fontweight='bold', fontsize=12)
    ax.legend(facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'], loc='best', fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f' Saved: {save_path}')


def plot_monthly_trend(df, save_path='charts/monthly_trend.png'):
    """Bar chart of monthly spending with average reference line."""
    _ensure_charts_dir(save_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax, 'Monthly Spending Trend', 'Month', 'Total Spent')

    monthly = df.groupby('month_str')['amt'].sum()
    avg = monthly.mean()
    colors = ['#93C5FD' if v <= avg else COLORS['accent1']
              for v in monthly.values]
    ax.bar(range(len(monthly)), monthly.values,
           color=colors, edgecolor='#FFFFFF', linewidth=1)
    ax.axhline(avg, color=COLORS['accent4'], linestyle='--',
               linewidth=2, label=f'Avg ${avg:,.0f}')
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha='right',
                       color=COLORS['subtext'])
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt_k))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f' Saved: {save_path}')


def plot_dow_bar(df, save_path='charts/dow_bar.png'):
    """Bar chart of average spending by day of week."""
    _ensure_charts_dir(save_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax, 'Avg Spending by Day of Week', 'Day', 'Avg Spent')

    order = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']
    dow = df.groupby('day_of_week')['amt'].mean().reindex(order).fillna(0)
    max_day = dow.idxmax() if dow.notna().any() else None
    colors = [COLORS['accent5'] if d == max_day
              else COLORS['accent2'] for d in dow.index]
    bars = ax.bar(range(7), dow.values,
                  color=colors, edgecolor='#FFFFFF', linewidth=1)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                       color=COLORS['subtext'])
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))

    for bar, val in zip(bars, dow.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + dow.max() * 0.01,
                f'${val:,.0f}', ha='center', va='bottom',
                fontsize=8, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f' Saved: {save_path}')


def plot_top_merchants(df, save_path='charts/top_merchants.png'):
    """Horizontal bar chart of top 10 merchants by spending."""
    _ensure_charts_dir(save_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax, 'Top 10 Merchants by Spending', 'Total Spent', '')

    top = df.groupby('merchant')['amt'].sum().nlargest(10).sort_values()
    bars = ax.barh(top.index, top.values,
                   color=COLORS['accent3'],
                   edgecolor='#FFFFFF', linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(dollar_fmt_k))
    ax.tick_params(axis='y', labelcolor=COLORS['subtext'], labelsize=8)

    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + top.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'${val:,.0f}', ha='left', va='center',
                fontsize=8, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f' Saved: {save_path}')


def plot_daily_spending(df, save_path='charts/daily_spending.png'):
    """Line chart of daily spending with average reference line."""
    _ensure_charts_dir(save_path)
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    style_ax(ax, 'Daily Spending with Average Reference',
             'Date', 'Daily Spent')

    daily = df.groupby(df['trans_date_trans_time'].dt.date)['amt'].sum()
    avg = daily.mean()
    ax.plot(daily.index, daily.values,
            color=COLORS['accent1'], linewidth=1.2, alpha=0.8)
    ax.fill_between(daily.index, daily.values,
                    alpha=0.1, color=COLORS['accent1'])
    ax.axhline(avg, color=COLORS['accent4'], linestyle='--',
               linewidth=2, label=f'Daily Avg ${avg:,.0f}')
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))
    ax.legend(fontsize=9)
    plt.xticks(rotation=45, ha='right', color=COLORS['subtext'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f' Saved: {save_path}')


# ── Main Dashboard ─────────────────────────────────────────────

def plot_dashboard(stats, top_cc, recommendations,
                   fraud_summary=None,
                   save_path='charts/smartspend_dashboard.png'):
    """
    Plot the full SmartSpend dashboard with KPI cards,
    category donut, monthly bar, day-of-week bar, budget panel,
    hourly bar, and fraud overview panel.
    """
    _ensure_charts_dir(save_path)
    setup_style()

    fig = plt.figure(figsize=(18, 24))
    fig.patch.set_facecolor(COLORS['bg'])
    gs = GridSpec(4, 2, figure=fig,
                  hspace=0.45, wspace=0.35,
                  top=0.95, bottom=0.04,
                  left=0.07, right=0.96)

    # ── Title ──────────────────────────────────────────────────
    fig.suptitle('SmartSpend — Personal Finance Dashboard',
                 fontsize=20, fontweight='bold',
                 color=COLORS['text'], y=0.98)

    # ══════════════════════════════════════════════════════════
    # ROW 0 — KPI Cards
    # ══════════════════════════════════════════════════════════
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(COLORS['bg'])
    fig.add_axes(ax_kpi)
    ax_kpi.axis('off')

    kpis = [
        ('Total Spent',     f"${stats['total_spent']:,.0f}",  COLORS['accent1']),
        ('Avg Monthly',     f"${stats['avg_monthly']:,.0f}",  COLORS['accent2']),
        ('Daily Burn Rate', f"${stats['avg_daily']:,.2f}",    COLORS['accent4']),
        ('Avg Transaction', f"${stats['avg_txn']:,.2f}",      COLORS['accent3']),
        ('Top Category',    str(stats['top_category']),        COLORS['accent5']),
        ('Fraud Flagged',   str(stats.get('fraud_count', 0)), COLORS['accent5']),
    ]

    card_w, card_h = 0.28, 0.38
    gap = 0.04
    start_x = 0.03

    for i, (label, value, color) in enumerate(kpis):
        col = i % 3
        row = i // 3
        x = start_x + col * (card_w + gap)
        y = 0.55 - row * (card_h + 0.06)

        # White card with colored border
        card = patches.FancyBboxPatch(
            (x, y), card_w, card_h,
            boxstyle='round,pad=0.02',
            facecolor=COLORS['panel'],
            edgecolor=color,
            linewidth=2,
            transform=ax_kpi.transAxes,
            clip_on=False
        )
        ax_kpi.add_patch(card)

        # Top colored accent bar on card
        accent_bar = patches.FancyBboxPatch(
            (x, y + card_h - 0.06), card_w, 0.05,
            boxstyle='round,pad=0.01',
            facecolor=color,
            edgecolor='none',
            transform=ax_kpi.transAxes,
            clip_on=False,
            alpha=0.15
        )
        ax_kpi.add_patch(accent_bar)

        # Label
        ax_kpi.text(x + card_w / 2, y + card_h * 0.68,
                    label,
                    ha='center', va='center',
                    fontsize=10, color=COLORS['subtext'],
                    transform=ax_kpi.transAxes)
        # Value
        ax_kpi.text(x + card_w / 2, y + card_h * 0.32,
                    value,
                    ha='center', va='center',
                    fontsize=16, color=color,
                    fontweight='bold',
                    transform=ax_kpi.transAxes)

    # ══════════════════════════════════════════════════════════
    # ROW 1 LEFT — Category Donut
    # ══════════════════════════════════════════════════════════
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor(COLORS['panel'])

    cat_spend = pd.Series(stats['category_spending'])
    total = cat_spend.sum()
    top_cats = cat_spend.nlargest(6).copy()
    other_val = total - top_cats.sum()
    if other_val > 0:
        top_cats['Other'] = other_val

    wedges, texts, autotexts = ax_pie.pie(
        top_cats.values,
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p > 4 else '',
        startangle=140,
        colors=CATEGORY_COLORS[:len(top_cats)],
        wedgeprops={'edgecolor': '#FFFFFF',
                    'linewidth': 2,
                    'width': 0.55}
    )
    plt.setp(autotexts, color='#FFFFFF', fontsize=8, fontweight='bold')

    # White center circle label
    ax_pie.text(0, 0, f'${total/1000:,.0f}k\nTotal',
                ha='center', va='center',
                fontsize=13, fontweight='bold',
                color=COLORS['text'])

    legend_labels = [
        f"{c}  ${v:,.0f}"
        for c, v in top_cats.items()
    ]
    ax_pie.legend(wedges, legend_labels,
                  loc='center left',
                  bbox_to_anchor=(1.0, 0.5),
                  fontsize=8,
                  frameon=True,
                  facecolor=COLORS['panel'],
                  edgecolor=COLORS['grid'],
                  labelcolor=COLORS['text'])
    ax_pie.set_title('Category Spending',
                     color=COLORS['text'],
                     fontweight='bold', fontsize=12, pad=12)

    # ══════════════════════════════════════════════════════════
    # ROW 1 RIGHT — Monthly Bar
    # ══════════════════════════════════════════════════════════
    ax_monthly = fig.add_subplot(gs[1, 1])
    style_ax(ax_monthly, 'Monthly Spending vs Average',
             'Month', 'Spending')

    monthly = pd.Series(stats['monthly_spending'])
    avg_m = monthly.mean()
    bar_colors = ['#93C5FD' if v <= avg_m else COLORS['accent1']
                  for v in monthly.values]
    ax_monthly.bar(range(len(monthly)), monthly.values,
                   color=bar_colors,
                   edgecolor='#FFFFFF', linewidth=1)
    ax_monthly.axhline(avg_m, color=COLORS['accent4'],
                       linestyle='--', linewidth=2,
                       label=f'Avg ${avg_m:,.0f}')
    if recommendations:
        budget_avg = sum(recommendations.values()) / len(recommendations)
        ax_monthly.axhline(budget_avg, color=COLORS['accent3'],
                           linestyle=':', linewidth=2,
                           label=f'Budget ${budget_avg:,.0f}')
    ax_monthly.set_xticks(range(len(monthly)))
    ax_monthly.set_xticklabels(monthly.index, rotation=45,
                                ha='right', fontsize=7,
                                color=COLORS['subtext'])
    ax_monthly.yaxis.set_major_formatter(FuncFormatter(dollar_fmt_k))
    ax_monthly.legend(fontsize=8, frameon=True,
                      facecolor=COLORS['panel'],
                      edgecolor=COLORS['grid'])

    # ══════════════════════════════════════════════════════════
    # ROW 2 LEFT — Day of Week
    # ══════════════════════════════════════════════════════════
    ax_dow = fig.add_subplot(gs[2, 0])
    style_ax(ax_dow, 'Avg Spend by Day of Week', 'Day', 'Avg Spent')

    order = ['Monday','Tuesday','Wednesday','Thursday',
             'Friday','Saturday','Sunday']
    dow = pd.Series(stats['dow_spending']).reindex(order).fillna(0)
    if dow.notna().any():
        max_day = dow.idxmax()
    else:
        max_day = None
    dow_colors = [COLORS['accent5'] if d == max_day
                  else COLORS['accent2'] for d in dow.index]
    bars = ax_dow.bar(range(7), dow.values,
                      color=dow_colors,
                      edgecolor='#FFFFFF', linewidth=1)
    ax_dow.set_xticks(range(7))
    ax_dow.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],
                            color=COLORS['subtext'], fontsize=9)
    ax_dow.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))

    for bar, val in zip(bars, dow.values):
        ax_dow.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + dow.max() * 0.01,
                    f'${val:,.0f}', ha='center', va='bottom',
                    fontsize=7, color=COLORS['text'])

    # ══════════════════════════════════════════════════════════
    # ROW 2 RIGHT — Budget Recommendation Panel
    # ══════════════════════════════════════════════════════════
    ax_budget = fig.add_subplot(gs[2, 1])
    ax_budget.set_facecolor(COLORS['panel'])
    ax_budget.axis('off')

    # Panel border
    border = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.02',
        facecolor=COLORS['panel'],
        edgecolor=COLORS['grid'],
        linewidth=1.5,
        transform=ax_budget.transAxes,
        clip_on=False
    )
    ax_budget.add_patch(border)

    ax_budget.text(0.5, 0.93, 'Budget Status',
                   ha='center', va='center',
                   fontsize=13, fontweight='bold',
                   color=COLORS['text'],
                   transform=ax_budget.transAxes)

    if recommendations:
        items = list(recommendations.items())[:6]
        row_h = 0.80 / len(items)
        for idx, (cat, rec) in enumerate(items):
            current = stats['category_spending'].get(cat, 0)
            pct = min((current / rec) * 100, 100) if rec > 0 else 0
            fill_color = (COLORS['accent3'] if pct < 85
                          else COLORS['accent4'] if pct < 100
                          else COLORS['accent5'])
            y_top = 0.86 - idx * row_h

            # Category label
            ax_budget.text(0.06, y_top,
                           cat[:16],
                           ha='left', va='center',
                           fontsize=9, color=COLORS['subtext'],
                           transform=ax_budget.transAxes)

            # Amount label
            ax_budget.text(0.94, y_top,
                           f'${current:,.0f} / ${rec:,.0f}',
                           ha='right', va='center',
                           fontsize=9, color=COLORS['text'],
                           fontweight='bold',
                           transform=ax_budget.transAxes)

            bar_y = y_top - 0.055
            bar_x = 0.06
            bar_w = 0.88
            bar_h = 0.03

            # Progress bar background — LIGHT GRAY
            ax_budget.add_patch(patches.FancyBboxPatch(
                (bar_x, bar_y), bar_w, bar_h,
                boxstyle='round,pad=0.005',
                facecolor=COLORS['grid'],      # #CBD5E1 light gray
                edgecolor='none',
                transform=ax_budget.transAxes,
                clip_on=False
            ))

            # Progress bar fill
            if pct > 0:
                ax_budget.add_patch(patches.FancyBboxPatch(
                    (bar_x, bar_y), bar_w * (pct / 100), bar_h,
                    boxstyle='round,pad=0.005',
                    facecolor=fill_color,
                    edgecolor='none',
                    transform=ax_budget.transAxes,
                    clip_on=False
                ))

            # Percent label
            ax_budget.text(bar_x + bar_w / 2,
                           bar_y + bar_h / 2,
                           f'{pct:.0f}%',
                           ha='center', va='center',
                           fontsize=7,
                           color='#FFFFFF' if pct > 20 else COLORS['text'],
                           fontweight='bold',
                           transform=ax_budget.transAxes)

            # Separator line
            if idx < len(items) - 1:
                ax_budget.axhline(
                    y=y_top - row_h + 0.01,
                    xmin=0.05, xmax=0.95,
                    color=COLORS['grid'],   # #CBD5E1 light gray
                    linewidth=0.8,
                )
    else:
        ax_budget.text(0.5, 0.5,
                       'No recommendations available\n(need 10+ transactions)',
                       ha='center', va='center',
                       fontsize=11, color=COLORS['subtext'],
                       transform=ax_budget.transAxes)

    # ══════════════════════════════════════════════════════════
    # ROW 3 LEFT — Hourly Bar
    # ══════════════════════════════════════════════════════════
    ax_hourly = fig.add_subplot(gs[3, 0])
    style_ax(ax_hourly, 'Avg Spend by Hour of Day', 'Hour', 'Avg Spent')

    hourly = pd.Series(stats['hourly_spending'])
    ax_hourly.bar(hourly.index, hourly.values,
                  color=COLORS['accent3'],
                  edgecolor='#FFFFFF', linewidth=1)
    ax_hourly.set_xticks(range(0, 24, 2))
    ax_hourly.set_xticklabels(
        [f'{h:02d}:00' for h in range(0, 24, 2)],
        rotation=45, ha='right', fontsize=7,
        color=COLORS['subtext']
    )
    ax_hourly.yaxis.set_major_formatter(FuncFormatter(dollar_fmt))

    # ══════════════════════════════════════════════════════════
    # ROW 3 RIGHT — Fraud Overview
    # ══════════════════════════════════════════════════════════
    ax_fraud = fig.add_subplot(gs[3, 1])
    ax_fraud.set_facecolor(COLORS['panel'])

    safe_count = None
    flagged_count = None
    threshold = None

    if isinstance(fraud_summary, dict):
        safe_count = fraud_summary.get('safe_count',
                     fraud_summary.get('safe', None))
        flagged_count = fraud_summary.get('flagged_count',
                        fraud_summary.get('flagged',
                        fraud_summary.get('fraud_count', None)))
        threshold = fraud_summary.get('threshold', None)
    elif hasattr(fraud_summary, 'columns'):
        if 'fraud_predicted' in fraud_summary.columns:
            flagged_count = int(fraud_summary['fraud_predicted'].sum())
            safe_count = len(fraud_summary) - flagged_count

    if safe_count is None or flagged_count is None:
        style_ax(ax_fraud, 'Fraud Overview')
        ax_fraud.text(0.5, 0.5,
                      'Fraud Detection Not Run\n\nClick "Run Fraud Detection"\nin the sidebar to analyse.',
                      ha='center', va='center',
                      fontsize=11, color=COLORS['subtext'],
                      transform=ax_fraud.transAxes,
                      linespacing=1.8)
    else:
        style_ax(ax_fraud, 'Fraud Overview', 'Status', 'Transactions')
        values = [safe_count, flagged_count]
        labels = ['Safe', 'Flagged']
        bar_colors = ['#93C5FD', COLORS['accent5']]
        bars = ax_fraud.bar(labels, values,
                            color=bar_colors,
                            edgecolor='#FFFFFF', linewidth=1,
                            width=0.4)
        if threshold is not None:
            ax_fraud.axhline(threshold,
                             color=COLORS['accent4'],
                             linestyle='--', linewidth=2,
                             label=f'Threshold {threshold:.2f}')
            ax_fraud.legend(fontsize=8)

        for bar, val in zip(bars, values):
            ax_fraud.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + max(values) * 0.01,
                          f'{val:,}', ha='center', va='bottom',
                          fontsize=11, color=COLORS['text'],
                          fontweight='bold')

    # ── Save ───────────────────────────────────────────────────
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print(f' Dashboard saved: {save_path}')
    return save_path