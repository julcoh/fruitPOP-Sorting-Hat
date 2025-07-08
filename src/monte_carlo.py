# src/monte_carlo.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import streamlit as st
from tqdm import tqdm
from src.solver import run_shift_solver

def run_monte_carlo(volunteers, shift_ids, shifts, vol_min_points,
                    min_pts, max_over, num_trials, sequential_pairs):

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
            volunteers, shift_ids, shifts, prefs_random_df, vol_min_points,
            min_pts, max_over, random.randint(1, 1_000_000),
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

    ranks_top10 = [r for r in range(1, 11) if any(results_by_rank[r])]
    if ranks_top10:
        means = [np.mean(results_by_rank[r]) for r in ranks_top10]
        lower = [np.percentile(results_by_rank[r], 5) for r in ranks_top10]
        upper = [np.percentile(results_by_rank[r], 95) for r in ranks_top10]

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(ranks_top10, means,
               yerr=[np.array(means)-np.array(lower), np.array(upper)-np.array(means)],
               capsize=5)
        ax.set_xlabel("Preference Rank")
        ax.set_ylabel("% of Volunteers with ≥1 Shift at Rank")
        ax.set_title(f"Monte Carlo ({num_trials} trials): Ranks 1–10")
        ax.grid(True)
        st.pyplot(fig)

    st.write(f"Phase 1 success rate: {phase1_success/num_trials:.1%}")
    st.write(f"Phase 2 success rate: {phase2_success/num_trials:.1%}")
    st.write(f"Failed trials: {failed/num_trials:.1%}")
