import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Parameters
BATTERY_CAPACITY = 10  # kWh
POWER_RATING = 1       # kW (change this to test different c-rates)
EFFICIENCY = 0.95

def load_data():
    df = pd.read_csv('energy.csv', skiprows=2)
    df = df.iloc[:, [0, 3]]
    df.columns = ['timestamp', 'price_eur_mwh']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price_eur_kwh'] = df['price_eur_mwh'] / 1000
    df = df.dropna().reset_index(drop=True)
    df['date'] = df['timestamp'].dt.date
    return df

def select_hours(daily, n_hours, before=None):
    """Select n_hours with lowest price before a certain time (or whole day if before=None)"""
    if before is not None:
        daily = daily[daily['timestamp'] < before]
    return daily.nsmallest(n_hours, 'price_eur_kwh')

def select_hours_desc(daily, n_hours, after=None):
    """Select n_hours with highest price after a certain time (or whole day if after=None)"""
    if after is not None:
        daily = daily[daily['timestamp'] > after]
    return daily.nlargest(n_hours, 'price_eur_kwh')

def optimize_day(daily, cycle_hours):
    # Try all possible splits: all charging hours before all discharging hours
    best = None
    for split in range(cycle_hours, len(daily) - cycle_hours + 1):
        charge_block = daily.iloc[:split]
        discharge_block = daily.iloc[split:]
        if len(charge_block) < cycle_hours or len(discharge_block) < cycle_hours:
            continue
        charges = charge_block.nsmallest(cycle_hours, 'price_eur_kwh')
        discharges = discharge_block.nlargest(cycle_hours, 'price_eur_kwh')
        charge_cost = charges['price_eur_kwh'].sum() * (BATTERY_CAPACITY / cycle_hours)
        discharge_revenue = discharges['price_eur_kwh'].sum() * (BATTERY_CAPACITY / cycle_hours) * EFFICIENCY
        profit = discharge_revenue - charge_cost
        if (best is None) or (profit > best['profit']):
            best = dict(
                charge_hours=charges,
                discharge_hours=discharges,
                profit=profit,
                charge_cost=charge_cost,
                discharge_revenue=discharge_revenue
            )
    return best

def run_bess(df):
    cycle_hours = math.ceil(BATTERY_CAPACITY / POWER_RATING)
    results = []
    for date in df['date'].unique():
        daily = df[df['date'] == date].copy().reset_index(drop=True)
        if len(daily) < 2 * cycle_hours:
            continue
        best = optimize_day(daily, cycle_hours)
        if best and best['profit'] > 0:
            results.append((date, best))
    # Mark actions in dataframe
    df['action'] = 'idle'
    df['soc'] = 0.0
    for date, op in results:
        soc = 0.0
        mask = df['date'] == date
        for idx, row in df[mask].iterrows():
            if row['timestamp'] in list(op['charge_hours']['timestamp']):
                energy = min(POWER_RATING, BATTERY_CAPACITY - soc)
                soc += energy
                df.at[idx, 'action'] = 'charge'
            elif row['timestamp'] in list(op['discharge_hours']['timestamp']):
                energy = min(POWER_RATING, soc)
                soc -= energy
                df.at[idx, 'action'] = 'discharge'
            df.at[idx, 'soc'] = soc
    return df, results

def plot_bess(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['timestamp'], df['price_eur_kwh'], label='Price (EUR/kWh)', color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['soc']/BATTERY_CAPACITY*100, label='SOC (%)', color='tab:red', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price (EUR/kWh)', color='tab:blue')
    ax2.set_ylabel('SOC (%)', color='tab:red')
    charge_mask = df['action'] == 'charge'
    discharge_mask = df['action'] == 'discharge'
    ax1.scatter(df[charge_mask]['timestamp'], df[charge_mask]['price_eur_kwh'], color='green', marker='^', s=100, label='Charge')
    ax1.scatter(df[discharge_mask]['timestamp'], df[discharge_mask]['price_eur_kwh'], color='red', marker='v', s=100, label='Discharge')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'BESS Optimal Arbitrage | {POWER_RATING} kW, {BATTERY_CAPACITY} kWh, C={POWER_RATING/BATTERY_CAPACITY:.2f}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    df, results = run_bess(df)
    plot_bess(df)
    df.to_csv('bess_results.csv', index=False)
    for date, op in results:
        print(f"{date}: Profit={op['profit']:.2f} EUR, Charge cost={op['charge_cost']:.2f}, Discharge revenue={op['discharge_revenue']:.2f}")
