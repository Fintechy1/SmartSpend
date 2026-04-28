import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading
import os
from datetime import datetime

from data_manager import load_and_clean
from analysis import compute_analytics
from recommendation_engine import get_recommendations, get_warnings
from fraud_detector import train_models, score_user_transactions, plot_fraud_charts, print_summary_table
from visualization import (
    plot_dashboard,
    plot_category_bar,
    plot_category_pie,
    plot_monthly_trend,
    plot_dow_bar,
    plot_top_merchants,
    plot_daily_spending,
)
from report_generator import save_report

# ── Colors ────────────────────────────────────────────────────────────────────
BG         = '#F0F4FA'
PANEL      = '#FFFFFF'
BORDER     = '#D1D9E6'
HDR_BG     = '#1A56DB'
HDR_FG     = '#FFFFFF'
LABEL_FG   = '#1E293B'
SUB_FG     = '#64748B'
BLUE       = '#1A56DB'
GREEN      = '#059669'
AMBER      = '#D97706'
RED        = '#DC2626'
BTN_ACTIVE = '#1740A8'

F_TITLE    = ('Helvetica', 12, 'bold')
F_LABEL    = ('Helvetica', 10, 'bold')
F_BODY     = ('Helvetica', 10)
F_SMALL    = ('Helvetica', 9)
F_BIG      = ('Helvetica', 20, 'bold')


# ── Reusable widget builders ──────────────────────────────────────────────────
def make_card(parent):
    f = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    return f

def section_label(parent, text):
    tk.Label(parent, text=text, font=F_TITLE, bg=PANEL, fg=LABEL_FG).pack(
        anchor='w', padx=12, pady=(10, 2))
    tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=12, pady=(0, 6))

def flat_btn(parent, text, cmd, bg=None, fg='#FFFFFF'):
    bg = bg or BLUE
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, font=F_LABEL,
                  relief='flat', bd=0, cursor='hand2',
                  activebackground=BTN_ACTIVE, activeforeground=fg,
                  padx=10, pady=7)
    return b

def labeled_entry(parent, label_text, row, var):
    tk.Label(parent, text=label_text, font=F_BODY, bg=PANEL, fg=LABEL_FG).grid(
        row=row, column=0, sticky='w', pady=4, padx=(0, 8))
    e = tk.Entry(parent, textvariable=var, font=F_BODY,
                 bg='#F8FAFC', fg=LABEL_FG, relief='solid', bd=1,
                 highlightcolor=BLUE, highlightthickness=1)
    e.grid(row=row, column=1, sticky='ew', pady=4)
    return e


# ── Main GUI ──────────────────────────────────────────────────────────────────
class SmartSpendGUI:
    def __init__(self, df_user, df_full, top_cc, stats, recommendations, warnings):
        self.df_user         = df_user
        self.df_full         = df_full
        self.top_cc          = top_cc
        self.stats           = stats
        self.recommendations = recommendations
        self.warnings        = warnings
        self.fraud_results   = None

        self.root = tk.Tk()
        self.root.title('SmartSpend - Finance Dashboard')
        self.root.configure(bg=BG)
        self.root.geometry('1440x900')
        self.root.minsize(1100, 700)

        self._build_header()
        self._build_body()
        self._refresh_chart()

    # ── Header bar ────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HDR_BG, height=50)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text='SmartSpend', font=('Helvetica', 15, 'bold'),
                 bg=HDR_BG, fg=HDR_FG).pack(side='left', padx=18, pady=12)
        tk.Label(hdr, text=datetime.now().strftime('%A, %d %B %Y'),
                 font=F_SMALL, bg=HDR_BG, fg='#93C5FD').pack(side='right', padx=18)

    # ── Three-column body ─────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=10, pady=10)

        self.left  = tk.Frame(body, bg=BG, width=260)
        self.mid   = tk.Frame(body, bg=BG)
        self.right = tk.Frame(body, bg=BG, width=270)

        self.left.pack(side='left', fill='y', padx=(0, 8))
        self.left.pack_propagate(False)
        self.mid.pack(side='left', fill='both', expand=True)
        self.right.pack(side='right', fill='y', padx=(8, 0))
        self.right.pack_propagate(False)

        self._build_left()
        self._build_center()
        self._build_right()

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left(self):
        p = self.left

        # --- Add Transaction card ---
        card = make_card(p)
        card.pack(fill='x', pady=(0, 8))
        section_label(card, 'Add Transaction')

        form = tk.Frame(card, bg=PANEL)
        form.pack(fill='x', padx=12, pady=(0, 10))
        form.columnconfigure(1, weight=1)

        self.date_var     = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        self.category_var = tk.StringVar()
        self.desc_var     = tk.StringVar()
        self.amount_var   = tk.StringVar()

        labeled_entry(form, 'Date',        0, self.date_var)
        labeled_entry(form, 'Description', 2, self.desc_var)
        labeled_entry(form, 'Amount ($)',  3, self.amount_var)

        # Category dropdown
        tk.Label(form, text='Category', font=F_BODY, bg=PANEL, fg=LABEL_FG).grid(
            row=1, column=0, sticky='w', pady=4, padx=(0, 8))
        cats = list(self.stats.get('category_spending', {}).keys())
        cb = ttk.Combobox(form, textvariable=self.category_var, values=cats,
                          font=F_BODY, state='readonly')
        cb.grid(row=1, column=1, sticky='ew', pady=4)

        flat_btn(card, 'Add Transaction', self.add_transaction, bg=GREEN).pack(
            fill='x', padx=12, pady=(0, 12))

        # --- Actions card ---
        card2 = make_card(p)
        card2.pack(fill='x', pady=(0, 8))
        section_label(card2, 'Actions')

        flat_btn(card2, 'Export PDF Report', self.export_report).pack(
            fill='x', padx=12, pady=(0, 6))

        self.fraud_btn = flat_btn(card2, 'Run Fraud Detection', self.run_fraud_detection, bg=AMBER)
        self.fraud_btn.pack(fill='x', padx=12, pady=(0, 12))

        # --- Warnings card ---
        if self.warnings:
            card3 = make_card(p)
            card3.pack(fill='x', pady=(0, 8))
            section_label(card3, 'Warnings')
            for w in list(self.warnings.values())[:5]:
                row = tk.Frame(card3, bg=PANEL)
                row.pack(fill='x', padx=12, pady=2)
                tk.Label(row, text='!', font=F_LABEL, bg='#FEF3C7',
                         fg=AMBER, width=2).pack(side='left')
                tk.Label(row, text=w, font=F_SMALL, bg=PANEL, fg=LABEL_FG,
                         wraplength=195, justify='left').pack(side='left', padx=6)
            tk.Frame(card3, bg=PANEL, height=6).pack()

    # ── Center panel ──────────────────────────────────────────────────────────
    def _build_center(self):
        # Metric tiles row
        tile_row = tk.Frame(self.mid, bg=BG)
        tile_row.pack(fill='x', pady=(0, 8))

        metrics = [
            ('Total Spent',   f"${self.stats.get('total_spent', 0):,.2f}",   BLUE),
            ('Avg Txn',       f"${self.stats.get('avg_txn', 0):.2f}",         GREEN),
            ('Daily Burn',    f"${self.stats.get('avg_daily', 0):.2f}",        AMBER),
            ('Fraud Flagged', str(self.stats.get('fraud_count', 0)),            RED),
            ('Top Category',  str(self.stats.get('top_category', '-')),         SUB_FG),
        ]

        self.metric_labels = {}
        for i, (lbl, val, color) in enumerate(metrics):
            tile = make_card(tile_row)
            tile.grid(row=0, column=i, padx=4, sticky='ew')
            tile_row.columnconfigure(i, weight=1)
            tk.Label(tile, text=lbl, font=F_SMALL, bg=PANEL, fg=SUB_FG).pack(
                anchor='w', padx=10, pady=(8, 0))
            v_lbl = tk.Label(tile, text=val, font=F_BIG, bg=PANEL, fg=color)
            v_lbl.pack(anchor='w', padx=10, pady=(0, 8))
            self.metric_labels[lbl] = v_lbl

        # Chart canvas card
        chart_card = make_card(self.mid)
        chart_card.pack(fill='both', expand=True)
        section_label(chart_card, 'Spending Dashboard')

        self.chart_frame = tk.Frame(chart_card, bg=PANEL)
        self.chart_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right(self):
        p = self.right

        # Budget status card
        budget_card = make_card(p)
        budget_card.pack(fill='x', pady=(0, 8))
        section_label(budget_card, 'Budget vs Actual')

        self.budget_widgets = {}
        recs = self.recommendations or {}
        spending = self.stats.get('category_spending', {})

        for cat in list(recs.keys())[:8]:
            current = spending.get(cat, 0)
            budget  = recs[cat]
            pct     = min(current / budget * 100, 100) if budget > 0 else 0
            over    = current > budget

            row = tk.Frame(budget_card, bg=PANEL)
            row.pack(fill='x', padx=12, pady=3)

            tk.Label(row, text=cat[:16], font=F_SMALL, bg=PANEL, fg=LABEL_FG,
                     width=16, anchor='w').pack(side='left')

            bar_bg = tk.Frame(row, bg='#E2E8F0', height=10, width=100)
            bar_bg.pack(side='left', padx=6)
            bar_bg.pack_propagate(False)
            bar_color = RED if over else GREEN
            fill_w = max(1, int(pct))
            bar_fill = tk.Frame(bar_bg, bg=bar_color, height=10, width=fill_w)
            bar_fill.place(x=0, y=0, relheight=1, relwidth=pct / 100)

            amount_lbl = tk.Label(row, text=f'${current:,.0f}/${budget:,.0f}',
                                  font=F_SMALL, bg=PANEL,
                                  fg=RED if over else SUB_FG)
            amount_lbl.pack(side='right')

            self.budget_widgets[cat] = (bar_fill, amount_lbl)

        # Recent transactions card
        txn_card = make_card(p)
        txn_card.pack(fill='both', expand=True, pady=(0, 0))
        section_label(txn_card, 'Recent Transactions')

        cols = ('Date', 'Category', 'Amount')
        tree = ttk.Treeview(txn_card, columns=cols, show='headings', height=14)
        tree.heading('Date',     text='Date')
        tree.heading('Category', text='Category')
        tree.heading('Amount',   text='Amount')
        tree.column('Date',     width=80,  anchor='w')
        tree.column('Category', width=90,  anchor='w')
        tree.column('Amount',   width=70,  anchor='e')

        style = ttk.Style()
        style.configure('Treeview',
                         background=PANEL, foreground=LABEL_FG,
                         rowheight=22, fieldbackground=PANEL,
                         font=F_SMALL)
        style.configure('Treeview.Heading', font=F_LABEL,
                         background='#EFF6FF', foreground=LABEL_FG)
        style.map('Treeview', background=[('selected', '#DBEAFE')])

        recent = self.df_user.sort_values('trans_date_trans_time', ascending=False).head(20)
        for _, r in recent.iterrows():
            date = pd.to_datetime(r['trans_date_trans_time']).strftime('%m/%d/%y')
            cat  = str(r.get('category', ''))[:14]
            amt  = f"${r['amt']:,.2f}"
            tree.insert('', 'end', values=(date, cat, amt))

        sb = ttk.Scrollbar(txn_card, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side='left', fill='both', expand=True, padx=(12, 0), pady=(0, 8))
        sb.pack(side='right', fill='y', pady=(0, 8), padx=(0, 4))

    # ── Chart rendering ───────────────────────────────────────────────────────
    def _refresh_chart(self):
        for w in self.chart_frame.winfo_children():
            w.destroy()

        spending = self.stats.get('category_spending', {})
        monthly  = self.stats.get('monthly_spending', {})

        if not spending:
            tk.Label(self.chart_frame, text='No data available.',
                     font=F_BODY, bg=PANEL, fg=SUB_FG).pack(expand=True)
            return

        fig = plt.Figure(figsize=(9, 5), facecolor=PANEL, tight_layout=True)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12, wspace=0.35)

        ax1 = fig.add_subplot(1, 2, 1)   # bar: category spending
        ax2 = fig.add_subplot(1, 2, 2)   # line: monthly trend

        # --- Category bar chart ---
        cats   = list(spending.keys())[:10]
        values = [spending[c] for c in cats]
        colors = [GREEN if v < (self.recommendations or {}).get(c, float('inf'))
                  else RED for v, c in zip(values, cats)]

        bars = ax1.barh(cats, values, color=colors, height=0.55, zorder=3)
        ax1.set_facecolor('#F8FAFC')
        ax1.set_title('Spending by Category', fontsize=10, fontweight='bold',
                      color=LABEL_FG, pad=8)
        ax1.set_xlabel('Amount ($)', fontsize=8, color=SUB_FG)
        ax1.tick_params(axis='both', labelsize=8, colors=LABEL_FG)
        ax1.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax1.grid(axis='x', color='#E2E8F0', zorder=0)
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.spines[['left', 'bottom']].set_color(BORDER)
        for bar, v in zip(bars, values):
            ax1.text(v + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'${v:,.0f}', va='center', fontsize=7, color=LABEL_FG)

        # --- Monthly trend line ---
        if monthly:
            months = sorted(monthly.keys())
            totals = [monthly[m] for m in months]
            ax2.plot(range(len(months)), totals, color=BLUE, linewidth=2.2,
                     marker='o', markersize=5, zorder=3)
            ax2.fill_between(range(len(months)), totals, alpha=0.12, color=BLUE)
            ax2.set_facecolor('#F8FAFC')
            ax2.set_title('Monthly Spending Trend', fontsize=10, fontweight='bold',
                          color=LABEL_FG, pad=8)
            ax2.set_xticks(range(len(months)))
            ax2.set_xticklabels(
                [str(m)[-5:] if len(str(m)) > 4 else str(m) for m in months],
                rotation=40, ha='right', fontsize=7, color=LABEL_FG)
            ax2.tick_params(axis='y', labelsize=8, colors=LABEL_FG)
            ax2.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            ax2.grid(axis='y', color='#E2E8F0', zorder=0)
            ax2.spines[['top', 'right']].set_visible(False)
            ax2.spines[['left', 'bottom']].set_color(BORDER)
        else:
            ax2.text(0.5, 0.5, 'No monthly data', ha='center', va='center',
                     transform=ax2.transAxes, color=SUB_FG)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Save the current dashboard and supporting charts while running
        self._save_visualizations()

    def _save_visualizations(self):
        try:
            plot_dashboard(self.stats, self.top_cc, self.recommendations,
                           fraud_summary=self.fraud_results)
            plot_category_bar(self.df_user)
            plot_category_pie(self.df_user)
            plot_monthly_trend(self.df_user)
            plot_dow_bar(self.df_user)
            plot_top_merchants(self.df_user)
            plot_daily_spending(self.df_user)
        except Exception as e:
            print(f'Visualization save failed: {e}')

    # ── Add transaction ───────────────────────────────────────────────────────
    def add_transaction(self):
        try:
            date   = pd.to_datetime(self.date_var.get())
            cat    = self.category_var.get().strip()
            desc   = self.desc_var.get().strip()
            amount = float(self.amount_var.get())

            if amount <= 0:
                raise ValueError('Amount must be positive')
            if not cat:
                raise ValueError('Category is required')
            if not desc:
                raise ValueError('Description is required')

            new_row = {
                'trans_date_trans_time': date,
                'category': cat,
                'merchant': desc,
                'amt': amount,
                'cc_num': self.top_cc,
                'is_fraud': 0,
            }
            BASE_DIR = os.path.dirname(__file__)
            csv_path = os.path.join(BASE_DIR, 'cleaned_transactions.csv')
            pd.DataFrame([new_row]).to_csv(csv_path, mode='a', header=False, index=False)

            self.desc_var.set('')
            self.amount_var.set('')
            self.date_var.set(datetime.now().strftime('%Y-%m-%d'))
            self._reload_and_refresh()
            messagebox.showinfo('Success', 'Transaction added.')

        except Exception as e:
            messagebox.showerror('Error', f'Invalid input: {e}')

    # ── Export report ─────────────────────────────────────────────────────────
    def export_report(self):
        try:
            path = save_report(self.df_user, self.stats, self.recommendations,
                               self.warnings, self.fraud_results)
            messagebox.showinfo('Report Saved', f'Saved to:\n{path}')
        except Exception as e:
            messagebox.showerror('Error', f'Export failed: {e}')

    # ── Fraud detection ───────────────────────────────────────────────────────
    def run_fraud_detection(self):
        self.fraud_btn.config(text='Training models...', state='disabled', bg='#94A3B8')

        def worker():
            try:
                self.fraud_results = train_models(self.df_full)
                encoders = self.fraud_results[list(self.fraud_results.keys())[0]]['encoders']
                self.df_user = score_user_transactions(
                    self.df_user, self.fraud_results, encoders, None)
                plot_fraud_charts(self.fraud_results)
                print_summary_table(self.fraud_results)
                self.root.after(0, lambda: (
                    self.fraud_btn.config(text='Fraud Detection Done', state='normal', bg=GREEN),
                    self._reload_and_refresh()
                ))
            except Exception as e:
                self.root.after(0, lambda: (
                    messagebox.showerror('Error', f'Fraud detection failed: {e}'),
                    self.fraud_btn.config(text='Run Fraud Detection', state='normal', bg=AMBER)
                ))

        threading.Thread(target=worker, daemon=True).start()

    # ── Reload data and refresh all widgets ───────────────────────────────────
    def _reload_and_refresh(self):
        try:
            BASE_DIR = os.path.dirname(__file__)
            csv_path = os.path.join(BASE_DIR, 'cleaned_transactions.csv')
            self.df_user, _, _ = load_and_clean(csv_path)
            self.stats           = compute_analytics(self.df_user)
            self.recommendations = get_recommendations(self.df_user, self.stats)
            self.warnings        = get_warnings(self.df_user, self.stats)

            # Update metric tiles
            self.metric_labels['Total Spent'].config(
                text=f"${self.stats.get('total_spent', 0):,.2f}")
            self.metric_labels['Avg Txn'].config(
                text=f"${self.stats.get('avg_txn', 0):.2f}")
            self.metric_labels['Daily Burn'].config(
                text=f"${self.stats.get('avg_daily', 0):.2f}")
            self.metric_labels['Fraud Flagged'].config(
                text=str(self.stats.get('fraud_count', 0)))
            self.metric_labels['Top Category'].config(
                text=str(self.stats.get('top_category', '-')))

            # Update budget bars
            spending = self.stats.get('category_spending', {})
            for cat, (bar_fill, lbl) in self.budget_widgets.items():
                current = spending.get(cat, 0)
                budget  = (self.recommendations or {}).get(cat, 0)
                pct     = min(current / budget, 1.0) if budget > 0 else 0
                over    = current > budget
                bar_fill.place(x=0, y=0, relheight=1, relwidth=pct)
                bar_fill.config(bg=RED if over else GREEN)
                lbl.config(text=f'${current:,.0f}/${budget:,.0f}',
                           fg=RED if over else SUB_FG)

            # Redraw charts
            self._refresh_chart()

        except Exception as e:
            messagebox.showerror('Error', f'Refresh failed: {e}')


# ── Entry point ───────────────────────────────────────────────────────────────
def launch(df_user, df_full, top_cc, stats, recommendations, warnings):
    gui = SmartSpendGUI(df_user, df_full, top_cc, stats, recommendations, warnings)
    return gui.root