import streamlit as st

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import random
from collections import defaultdict
from tqdm import tqdm

# --------------------------------------------------------------------------------
# Title
# --------------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("FruitPOP Volunteer Shift Optimizer")

# --------------------------------------------------------------------------------
# Upload CSVs
# --------------------------------------------------------------------------------
st.header("Upload Your Data")

uploaded_shifts = st.file_uploader("Upload Shifts CSV", type="csv")
uploaded_prefs = st.file_uploader("Upload Preferences CSV", type="csv")
uploaded_settings = st.file_uploader("Upload Settings CSV", type="csv")

if uploaded_shifts and uploaded_prefs and uploaded_settings:

    # Read uploaded files
    shifts = pd.read_csv(uploaded_shifts)
    prefs_raw = pd.read_csv(uploaded_prefs)
    settings_row = pd.read_csv(uploaded_settings).iloc[0]

    # Convert StartTime and EndTime to datetime if present
    if "StartTime" in shifts.columns and "EndTime" in shifts.columns:
        shifts["StartTime"] = pd.to_datetime(shifts["StartTime"])
        shifts["EndTime"] = pd.to_datetime(shifts["EndTime"])

    # Process prefs
    volunteer_names = prefs_raw.iloc[:,0].astype(str).tolist()

    if 'MinPoints' in prefs_raw.columns:
        min_points_col = prefs_raw['MinPoints']
    else:
        min_points_col = pd.Series([np.nan]*len(volunteer_names), index=volunteer_names)

    prefs_raw.iloc[:,0] = volunteer_names
    prefs = prefs_raw.iloc[:,2:]
    prefs.columns = prefs.columns.astype(str)
    prefs.index = volunteer_names

    # Read Settings
    default_min_pts = float(settings_row.get("MIN_POINTS", 8.0))
    default_seed = int(settings_row.get("SEED", 69))
    default_max_over = float(settings_row.get("MAX_OVER", 1.0))

    # Create volunteer-specific min points dict
    vol_min_points = {}
    for v in volunteer_names:
        val = min_points_col.loc[v] if v in min_points_col.index else np.nan
        if pd.isna(val):
            vol_min_points[v] = default_min_pts
        else:
            vol_min_points[v] = float(val)

    # Display previews
    st.subheader("Data Previews")
    st.write("Shifts Data", shifts.head())
    st.write("Preferences Data", prefs_raw.head())
    st.write("Settings", settings_row.to_frame().T)

    # --------------------------------------------------------------------------------
    # User Input Parameters
    # --------------------------------------------------------------------------------
    st.header("Solver Settings")

    min_pts = st.number_input("Min Points", value=default_min_pts, step=0.5)
    max_over = st.number_input("Max Over", value=default_max_over, step=0.5)
    seed = st.number_input("Random Seed", value=default_seed, step=1)

    # --------------------------------------------------------------------------------
    # Run Solver
    # --------------------------------------------------------------------------------
    if st.button("Run Solver"):

        st.write("Running Solver...")

        # Identify overlapping and sequential pairs
        overlapping_pairs = []
        sequential_pairs = []

        if "Date" in shifts.columns:
            for i, s1 in shifts.iterrows():
                for j, s2 in shifts.iterrows():
                    if j <= i:
                        continue
                    if s1["Date"] != s2["Date"]:
                        continue
                    if (s1["StartTime"] < s2["EndTime"]) and (s2["StartTime"] < s1["EndTime"]):
                        overlapping_pairs.append((s1["ShiftID"], s2["ShiftID"]))

            for date, group in shifts.groupby("Date"):
                group_sorted = group.sort_values("StartTime")
                shift_ids_sorted = group_sorted["ShiftID"].tolist()
                for k in range(len(shift_ids_sorted) - 1):
                    s1 = shift_ids_sorted[k]
                    s2 = shift_ids_sorted[k + 1]
                    sequential_pairs.append((s1, s2))

        else:
            overlapping_pairs = []
            sequential_pairs = []

        shift_ids = shifts['ShiftID'].astype(str).tolist()
        volunteers = prefs.index.tolist()

        # Run solver
        status_name, assignments, solver2, x, status2, best_cut = run_shift_solver(
            volunteers=volunteers,
            shift_ids=shift_ids,
            shifts_df=shifts,
            prefs_input=prefs,
            vol_min_points=vol_min_points,
            min_points=min_pts,
            max_over=max_over,
            seed=seed,
            overlapping_pairs=overlapping_pairs,
            sequential_pairs=sequential_pairs
        )

        st.success(f"Solver status: {status_name}")

        # Build DataFrames
        shift_vol_df, roster_df, audit_df = build_outputs(
            shifts, volunteers, shift_ids, x, solver2, prefs, seed, status2
        )

        st.subheader("ShiftVols Table")
        st.dataframe(shift_vol_df)

        st.subheader("Roster Table")
        st.dataframe(roster_df)

        st.subheader("Audit Table")
        st.dataframe(audit_df)

        # Download buttons
        st.download_button("Download ShiftVols CSV", shift_vol_df.to_csv(index=False).encode(), "shiftvols.csv")
        st.download_button("Download Roster CSV", roster_df.to_csv(index=False).encode(), "roster.csv")
        st.download_button("Download Audit CSV", audit_df.to_csv(index=False).encode(), "audit.csv")

        # --------------------------------------------------------------------------------
        # Optional Monte Carlo
        # --------------------------------------------------------------------------------
        st.header("Monte Carlo Simulation (Optional)")

        run_mc = st.checkbox("Run Monte Carlo Simulation?")

        if run_mc:
            num_trials = st.number_input("Number of Monte Carlo Trials", value=1000, step=100)

            if st.button("Run Monte Carlo"):
                st.write("Running Monte Carlo...")

                run_monte_carlo(
                    volunteers,
                    shift_ids,
                    shifts,
                    vol_min_points,
                    min_pts,
                    max_over,
                    num_trials,
                    sequential_pairs
                )

else:
    st.info("Please upload all three required CSV files to proceed.")

def run_shift_solver(
    volunteers,
    shift_ids,
    shifts_df,
    prefs_input,
    vol_min_points,
    min_points,
    max_over,
    seed=69,
    overlapping_pairs=[],
    sequential_pairs=[]
):
    rand = random.Random(seed)

    SCALE = 10

    points_d = dict(zip(shifts_df['ShiftID'].astype(str), shifts_df['Points']))

    def get_rank(v, s):
        if isinstance(prefs_input, pd.DataFrame):
            val = prefs_input.at[v, s] if s in prefs_input.columns else float('inf')
            return int(val) if not pd.isna(val) else float('inf')
        return float('inf')

    m2 = cp_model.CpModel()
    x = {(v, s): m2.NewBoolVar(f'x_{v}_{s}') for v in volunteers for s in shift_ids}

    points_d_scaled = {s: int(round(float(points_d[s]) * SCALE)) for s in shift_ids}

    for s in shift_ids:
        cap = int(shifts_df.loc[shifts_df['ShiftID'].astype(str) == s, 'Capacity'].values[0])
        m2.Add(sum(x[v, s] for v in volunteers) <= cap)

    for v in volunteers:
        min_pts_v_scaled = int(np.floor(vol_min_points[v] * SCALE))
        max_over_scaled  = int(np.ceil(max_over * SCALE))
        total_pts = sum(x[v, s] * points_d_scaled[s] for s in shift_ids)
        m2.Add(total_pts >= min_pts_v_scaled)
        m2.Add(total_pts <= min_pts_v_scaled + max_over_scaled)

        eligible = [x[v, s] for s in shift_ids if get_rank(v, s) <= 5]
        m2.Add(sum(eligible) >= 1)

        for s1, s2 in overlapping_pairs:
            m2.Add(x[v, s1] + x[v, s2] <= 1)

    penalties_phase1 = []
    penalty_weight_phase1 = 10000
    for v in volunteers:
        for s1, s2 in sequential_pairs:
            seq_var = m2.NewBoolVar(f"seq_{v}_{s1}_{s2}")
            m2.Add(x[v, s1] + x[v, s2] == 2).OnlyEnforceIf(seq_var)
            m2.Add(x[v, s1] + x[v, s2] <= 1).OnlyEnforceIf(seq_var.Not())
            penalties_phase1.append(penalty_weight_phase1 * seq_var)

    obj_terms = []
    for v in volunteers:
        for s in shift_ids:
            rank = get_rank(v, s)
            if rank == float('inf'):
                continue
            weight = {1:300, 2:200, 3:100}.get(rank, 50)
            weight += rand.randint(0,9)
            obj_terms.append(weight * x[v, s])

    m2.Maximize(sum(obj_terms) - sum(penalties_phase1))

    solver2 = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds = 60
    status = solver2.Solve(m2)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = {
            v: s
            for v in volunteers
            for s in shift_ids
            if solver2.BooleanValue(x[v, s])
        }
        return solver2.StatusName(status), assignments, solver2, x, status, 5
    else:
        return solver2.StatusName(status), {}, solver2, x, status, None


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


def run_monte_carlo(volunteers, shift_ids, shifts, vol_min_points, min_pts, max_over, num_trials, sequential_pairs):
    results_by_rank = {rank: [] for rank in range(1, len(shift_ids)+1)}
    phase1_success = 0
    phase2_success = 0
    failed = 0
    for trial in tqdm(range(num_trials), desc="Running Monte Carlo"):
        prefs_random_df = pd.DataFrame(index=volunteers, columns=shift_ids, dtype=float)
        for v in volunteers:
            num_choices = random.randint(1, min(5, len(shift_ids)))
            chosen_shifts = random.sample(shift_ids, num_choices)
            for i, s in enumerate(chosen_shifts, start=1):
                prefs_random_df.at[v, s] = i
        for s in shift_ids:
            if prefs_random_df[s].isna().all():
                v_choice = random.choice(volunteers)
                prefs_random_df.at[v_choice, s] = 1
        status_name, assignments, solver2, x, status2, best_cut = run_shift_solver(
            volunteers=volunteers,
            shift_ids=shift_ids,
            shifts_df=shifts,
            prefs_input=prefs_random_df,
            vol_min_points=vol_min_points,
            min_points=min_pts,
            max_over=max_over,
            seed=random.randint(1, 1_000_000),
            sequential_pairs=sequential_pairs
        )
        if status_name in ["OPTIMAL", "FEASIBLE"]:
            if best_cut is not None:
                phase1_success += 1
            else:
                phase2_success += 1
        else:
            failed += 1
            continue
        rank_hits_per_volunteer = defaultdict(set)
        for v in volunteers:
            assigned_shifts = [s for s in shift_ids if solver2.BooleanValue(x[v, s])]
            for s in assigned_shifts:
                rank = prefs_random_df.at[v, s]
                if pd.notna(rank) and rank <= len(shift_ids):
                    rank_hits_per_volunteer[v].add(int(rank))
        for rank in range(1, len(shift_ids)+1):
            count = sum(1 for vols_ranks in rank_hits_per_volunteer.values() if rank in vols_ranks)
            pct = 100 * count / len(volunteers)
            results_by_rank[rank].append(pct)

    # Plot
    ranks_top10 = [r for r in range(1, 11) if any(results_by_rank[r])]
    if ranks_top10:
        means = [np.mean(results_by_rank[r]) for r in ranks_top10]
        lower = [np.percentile(results_by_rank[r], 5) for r in ranks_top10]
        upper = [np.percentile(results_by_rank[r], 95) for r in ranks_top10]

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(ranks_top10, means, yerr=[np.array(means)-np.array(lower), np.array(upper)-np.array(means)], capsize=5)
        ax.set_xlabel("Preference Rank")
        ax.set_ylabel("% of Volunteers with ≥1 Shift at Rank")
        ax.set_title(f"Monte Carlo ({num_trials} trials): Ranks 1–10")
        ax.grid(True)
        st.pyplot(fig)

    st.write(f"Phase 1 success rate: {phase1_success/num_trials:.1%}")
    st.write(f"Phase 2 success rate: {phase2_success/num_trials:.1%}")
    st.write(f"Failed trials: {failed/num_trials:.1%}")
