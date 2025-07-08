# src/outputs.py

import pandas as pd
from collections import defaultdict

def build_outputs(shifts, volunteers, shift_ids, x, solver2, prefs, seed, status2):
    points_dict = dict(zip(shifts['ShiftID'].astype(str), shifts['Points']))

    shift_vol_rows, max_vols_shift = [], 0
    for _, row in shifts.iterrows():
        sid = str(row['ShiftID'])
        role = row.get('Role', '')
        cap = int(row['Capacity'])
        pts = int(row['Points'])
        vols = [v for v in volunteers if solver2.BooleanValue(x[v, sid])]
        max_vols_shift = max(max_vols_shift, len(vols))
        shift_vol_rows.append([sid, role, cap, pts, *vols])
    for r in shift_vol_rows:
        r += [''] * (max_vols_shift - (len(r)-4))

    shift_vol_cols = ['ShiftID', 'Role', 'Capacity', 'Points'] + [f'Volunteer{i+1}' for i in range(max_vols_shift)]
    shift_vol_df = pd.DataFrame(shift_vol_rows, columns=shift_vol_cols)

    roster_rows, max_shifts_vol = [], 0
    for v in volunteers:
        my_shifts = [s for s in shift_ids if solver2.BooleanValue(x[v, s])]
        max_shifts_vol = max(max_shifts_vol, len(my_shifts))
        roster_rows.append([v, *my_shifts])
    for r in roster_rows:
        r += [''] * (max_shifts_vol - (len(r)-1))
    roster_cols = ['Volunteer'] + [f'Shift{i+1}' for i in range(max_shifts_vol)]
    roster_df = pd.DataFrame(roster_rows, columns=roster_cols)

    rank_cols = [f'# {i} hits' for i in range(1, 21)]
    audit_rows = []
    rank_hit_aggregate = defaultdict(int)

    for v in volunteers:
        my_shifts = [s for s in shift_ids if solver2.BooleanValue(x[v, s])]
        total_pts = sum(points_dict.get(s, 0) for s in my_shifts)
        hits = [0]*20
        for s in my_shifts:
            if s in prefs.columns:
                r = pd.to_numeric(prefs.at[v, s], errors='coerce')
                if 1 <= r <= 20:
                    hits[int(r)-1] += 1
        for i, h in enumerate(hits, start=1):
            if h > 0:
                rank_hit_aggregate[i] += 1
        audit_rows.append(
            {'Volunteer': v, 'TotalPoints': total_pts,
             **{rank_cols[i]: hits[i] for i in range(20)},
             'AssignedShifts': "; ".join(my_shifts)}
        )
    audit_df = pd.DataFrame(audit_rows).sort_values('Volunteer').reset_index(drop=True)
    summary_rows = []
    n_vols = len(volunteers)
    for i in range(1, 21):
        count = rank_hit_aggregate.get(i, 0)
        summary_rows.append({'Rank': i, 'VolsWithHit': count, 'Percentage': round(count/n_vols*100, 1)})
    summary_df = pd.DataFrame(summary_rows)
    blank_row = pd.DataFrame([{'Volunteer': ''}])
    audit_df = pd.concat([audit_df, blank_row, summary_df], ignore_index=True)
    footer = pd.DataFrame([{
        'Volunteer': '*** Seed used', 'TotalPoints': seed, '# 1 hits': 'Solver status',
        'AssignedShifts': solver2.StatusName(status2)
    }])
    audit_df = pd.concat([audit_df, footer], ignore_index=True)
    return shift_vol_df, roster_df, audit_df
