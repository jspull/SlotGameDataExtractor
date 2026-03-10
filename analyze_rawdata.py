import pandas as pd
import re

df_raw = pd.read_excel(r'd:\SoundProgram\SlotGameDataExtractor\RawData.xlsx')
col = df_raw.columns[0]

records = []
for idx, row in df_raw.iterrows():
    text = str(row[col])
    m = re.match(r'\[(\d{2}:\d{2}:\d{2})\]\s+BALANCE:\s+([\d,.]+)\s+\|\s+WIN:\s+([\d,.]+)', text)
    if m:
        time_str = m.group(1)
        bal = float(m.group(2).replace(',',''))
        win = float(m.group(3).replace(',',''))
        records.append({'time': time_str, 'bal': bal, 'win': win})

df = pd.DataFrame(records)
print(f"Total raw records: {len(df)}")

# Find Win nonzero periods
print("\n=== Win > 0 periods ===")
prev_win = 0.0
for i, r in df.iterrows():
    if r['win'] > 0 and prev_win == 0:
        print(f"  WIN START at row {i}: {r['time']}  BAL={r['bal']:,.0f}  WIN={r['win']:,.0f}")
    if r['win'] == 0 and prev_win > 0:
        print(f"  WIN END   at row {i}: {r['time']}  BAL={r['bal']:,.0f}  (was WIN={prev_win:,.0f})")
        print()
    prev_win = r['win']

# Find balance drops >= 5000
print("\n=== Balance drops >= 5000 (potential bets) ===")
prev_bal = None
for i, r in df.iterrows():
    if prev_bal is not None:
        diff = prev_bal - r['bal']
        if diff >= 5000:
            print(f"  row {i}: {r['time']}  BAL: {prev_bal:,.0f} -> {r['bal']:,.0f}  drop={diff:,.0f}  WIN={r['win']:,.0f}")
    prev_bal = r['bal']

# Find balance increases (win payouts)
print("\n=== Balance increases (potential win payouts) ===")
prev_bal = None
for i, r in df.iterrows():
    if prev_bal is not None:
        diff = r['bal'] - prev_bal
        if diff > 0:
            print(f"  row {i}: {r['time']}  BAL: {prev_bal:,.0f} -> {r['bal']:,.0f}  gain={diff:,.0f}  WIN={r['win']:,.0f}")
    prev_bal = r['bal']

# Unique stable balance values with high count
print("\n=== Stable balance values (count >= 20) ===")
bal_groups = df.groupby('bal').size().sort_index()
for bal, cnt in bal_groups.items():
    if cnt >= 20:
        print(f"  BAL={bal:>15,.0f}  appear={cnt}")

# Show unique win values
print("\n=== Unique Win values ===")
win_groups = df.groupby('win').size().sort_index()
for win, cnt in win_groups.items():
    if cnt >= 1:
        print(f"  WIN={win:>12,.0f}  appear={cnt}")
