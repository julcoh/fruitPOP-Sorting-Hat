import streamlit as st
import pandas as pd
import numpy as np

from src.solver import run_shift_solver
from src.outputs import build_outputs
from src.monte_carlo import run_monte_carlo

st.set_page_config(layout="wide")
st.title("FruitPOP Volunteer Shift Optimizer")

# Upload data
st.header("Upload Your Data")

uploaded_shifts = st.file_uploader("Upload Shifts CSV", type="csv")
uploaded_prefs = st.file_uploader("Upload Preferences CSV", type="csv")
uploaded_settings = st.file_uploader("Upload Settings CSV", type="csv")

if uploaded_shifts and uploaded_prefs and uploaded_settings:
    # Load data
    shifts = pd.read_csv(uploaded_shifts)
    prefs_raw = pd.read_csv(uploaded_prefs)
    settings_row = pd.read_csv(uploaded_settings).iloc[0]

    # Convert times
    if "StartTime" in shifts.columns and "EndTime" in shifts.columns:
        shifts["StartTime"] = pd.to_datetime(shifts["StartTime"])
        shifts["EndTime"] = pd.to_datetime(shifts["EndTime"])

    # Process preferences
    volunteer_names = prefs_raw.iloc[:, 0].astype(str).tolist()
    if 'MinPoints' in prefs_raw.columns:
        min_points_col = prefs_raw['MinPoints']
    else:
        min_points_col = pd.Series([np.nan]*len(volunteer_names), index=volunteer_names)

    prefs_raw.iloc[:, 0] = volunteer_names
    prefs = prefs_raw.iloc[:, 2:]
    prefs.columns = prefs.columns.astype(str)
    prefs.index = volunteer_names

    # Settings
    default_min_pts = float(settings_row.get("MIN_POINTS", 8.0))
    default_seed = int(settings_row.get("SEED", 69))
    default_max_over = float(settings_row.get("MAX_OVER", 1.0))

    vol_min_points = {}
    for v in volunteer_names:
        val = min_points_col.loc[v] if v in min_points_col.index else np.nan
        if pd.isna(val):
            vol_min_points[v] = default_min_pts
        else:
            vol_min_points[v] = float(val)

    st.subheader("Data Previews")
    st.write("Shifts", shifts.head())
    st.write("Prefs", prefs_raw.head())
    st.write("Settings", settings_row.to_frame().T)

    # Solver inputs
    st.header("Solver Settings")
    min_pts = st.number_input("Min Points", value=default_min_pts, step=0.5)
    max_over = st.number_input("Max Over", value=default_max_over, step=0.5)
    seed = st.number_input("Random Seed", value=default_seed, step=1)

    # Run solver
    if st.button("Run Solver"):
        overlapping_pairs, sequential_pairs = [], []
        if "Date" in shifts.columns:
            for i, s1 in shifts.iterrows():
                for j, s2 in shifts.iterrows():
                    if j <= i:
                        continue
                    if s1["Date"] != s2["Date"]:
                        continue
                    if (s1["StartTime"] < s2["EndTime"]) and (s2["StartTime"] < s1["EndTime"]):
                        overlapping_pairs.append((str(s1["ShiftID"]), str(s2["ShiftID"])))

            for date, group in shifts.groupby("Date"):
                group_sorted = group.sort_values("StartTime")
                shift_ids_sorted = group_sorted["ShiftID"].astype(str).tolist()
                for k in range(len(shift_ids_sorted) - 1):
                    s1 = shift_ids_sorted[k]
                    s2 = shift_ids_sorted[k + 1]
                    sequential_pairs.append((s1, s2))

        shift_ids = shifts['ShiftID'].astype(str).tolist()
        volunteers = prefs.index.tolist()

        status_name, assignments, solver2, x, status2, best_cut = run_shift_solver(
            volunteers, shift_ids, shifts, prefs, vol_min_points,
            min_pts, max_over, seed, overlapping_pairs, sequential_pairs
        )

        st.success(f"Solver status: {status_name}")

        shift_vol_df, roster_df, audit_df = build_outputs(
            shifts, volunteers, shift_ids, x, solver2, prefs, seed, status2
        )

        st.subheader("ShiftVols Table")
        st.dataframe(shift_vol_df)

        st.subheader("Roster Table")
        st.dataframe(roster_df)

        st.subheader("Audit Table")
        st.dataframe(audit_df)

        # Downloads
        st.download_button("Download ShiftVols CSV", shift_vol_df.to_csv(index=False).encode(), "shiftvols.csv")
        st.download_button("Download Roster CSV", roster_df.to_csv(index=False).encode(), "roster.csv")
        st.download_button("Download Audit CSV", audit_df.to_csv(index=False).encode(), "audit.csv")

        # Monte Carlo
        st.header("Monte Carlo Simulation (Optional)")
        run_mc = st.checkbox("Run Monte Carlo Simulation?")
        if run_mc:
            num_trials = st.number_input("Number of Monte Carlo Trials", value=1000, step=100)
            if st.button("Run Monte Carlo"):
                run_monte_carlo(volunteers, shift_ids, shifts,
                                vol_min_points, min_pts, max_over,
                                num_trials, sequential_pairs)

else:
    st.info("Please upload all three required CSV files to proceed.")
