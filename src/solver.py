# src/solver.py

from ortools.sat.python import cp_model
import numpy as np
import random
import pandas as pd

def run_shift_solver(volunteers, shift_ids, shifts_df, prefs_input,
                     vol_min_points, min_points, max_over, seed,
                     overlapping_pairs=[], sequential_pairs=[]):
    rand = random.Random(seed)
    SCALE = 10
    points_d = dict(zip(shifts_df['ShiftID'].astype(str), shifts_df['Points']))

    def get_rank(v, s):
    if isinstance(prefs_input, pd.DataFrame):
        if s in prefs_input.columns:
            val = prefs_input.at[v, s]
            if pd.isna(val):
                return float('inf')
            try:
                return int(val)
            except (ValueError, TypeError):
                return float('inf')
        else:
            return float('inf')
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
