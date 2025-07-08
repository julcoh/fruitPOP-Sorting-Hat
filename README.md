# 📊 FruitPOP Volunteer Shift Optimizer

> **Optimize volunteer shift assignments and analyze outcomes with Monte Carlo simulations — all in an easy-to-use web app.**

---

## 🚀 What is FruitPOP?

FruitPOP helps volunteer coordinators **assign people to shifts** in the most efficient way possible, based on:

- **Volunteer preferences** (what shifts they want most)
- **Shift capacities** (how many people each shift can accept)
- **Point requirements** (each volunteer must work a certain number of “points” worth of shifts)

It also includes an **optional Monte Carlo simulation** to test how well your assignments perform under randomized preference scenarios.

---

## 🏗️ How the Code Works

### High-Level Architecture

The app is a Python web app built using **Streamlit**. The core architecture:

- **Frontend (Streamlit)**:
  - File uploads (CSV)
  - Parameter inputs (min points, max over, seed)
  - Interactive buttons
  - Displays tables and charts
  - Handles downloads of results

- **Backend logic**:
  - Runs an optimization solver (Google OR-Tools CP-SAT)
  - Constructs output tables:
    - ShiftVols (who’s in each shift)
    - Roster (what shifts each volunteer has)
    - Audit (summary of rank hits and total points)
  - Runs Monte Carlo simulations if requested
  - Generates plots

---

## 🔎 Optimization Approach

The solver works in **two phases**:

1. **Phase 1**
   - Assigns volunteers to shifts to maximize their preferences
   - Allows some shifts to remain under-filled if needed
   - Ensures each volunteer reaches at least their minimum points

2. **Phase 2 (if required)**
   - Forces all shifts to be filled
   - May override some preferences to meet hard constraints
   - Penalizes back-to-back shifts heavily

The objective is to:
- Maximize volunteer happiness (assign high-preference shifts)
- Avoid overlapping or sequential shifts where possible

---

## 🎲 Monte Carlo Simulation

Want to test your schedule’s resilience?

FruitPOP can:
- Randomly generate volunteer preferences
- Run thousands of optimization trials
- Report how often volunteers get one of their top choices
- Plot distributions for:
  - Rank hits
  - Total points assigned
  - Volunteer happiness
  - Sequential shifts

---

## 📂 Repo Layout

Here’s how the project is organized:

```
fruitpop-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│    ├── __init__.py
│    ├── solver.py         # OR-Tools optimization logic
│    ├── outputs.py        # Builds tables (ShiftVols, Roster, Audit)
│    └── monte_carlo.py    # Runs Monte Carlo simulations & plots
│
├── data/
│    ├── example_shifts.csv
│    ├── example_prefs.csv
│    └── example_settings.csv
│
└── .streamlit/
     └── config.toml
```

---

## ⚙️ How to Run the App

### 1. Clone the Repo

```
git clone https://github.com/yourusername/fruitpop-app.git
cd fruitpop-app
```

### 2. Install Python Requirements

We recommend a virtual environment.

```
pip install -r requirements.txt
```

### 3. Run Streamlit

```
streamlit run app.py
```

Your browser will open automatically (usually at http://localhost:8501).

---

## 🌐 Deploying Online

You can deploy this app to:

✅ **Streamlit Community Cloud** (free for public apps)  
✅ **Heroku**  
✅ **AWS / GCP / Azure**  
✅ **DigitalOcean**

All platforms support Python deployments.

---

## 🗂️ Input Data Format

You’ll upload **three CSV files**:

---

### Shifts CSV (`shifts.csv`)

| ShiftID | Role     | Date       | StartTime          | EndTime            | Capacity | Points |
|---------|----------|------------|--------------------|--------------------|----------|--------|
| 101     | Cashier  | 2025-08-01 | 2025-08-01 09:00   | 2025-08-01 12:00   | 3        | 2.0    |
| 102     | Greeter  | 2025-08-01 | 2025-08-01 12:30   | 2025-08-01 15:30   | 2        | 1.5    |

- `Points` can be fractional.
- `Date` helps detect overlapping or sequential shifts.

---

### Preferences CSV (`prefs.csv`)

| Volunteer  | MinPoints | 101 | 102 | 103 |
|------------|-----------|-----|-----|-----|
| Alice      | 8.0       | 1   | 3   |     |
| Bob        |           | 2   |     | 1   |

- **First column**: volunteer names
- **Second column**: optional individual min points
- Remaining columns:
    - Shift IDs as headers
    - Numbers = rank (1 = top choice)

---

### Settings CSV (`settings.csv`)

| MIN_POINTS | MAX_OVER | SEED |
|------------|----------|------|
| 8.0        | 1.0      | 69   |

- Default minimum points per volunteer
- Allowable buffer over minimum points
- Random seed for reproducibility

---

## 🎯 App Flow

✅ Upload shifts, preferences, and settings CSVs  
✅ Confirm solver parameters (or use defaults)  
✅ Click **Run Solver**  
✅ View results:
- ShiftVols table (per shift)
- Roster table (per volunteer)
- Audit summary

✅ Download any result as CSV

✅ (Optional) run Monte Carlo:
- Specify number of trials
- Visualize uncertainty in outcomes
- Check rank hit percentages and distributions

---

## 🛠️ Requirements

```
streamlit
pandas
numpy
matplotlib
ortools
tqdm
```

Install all with:

```
pip install -r requirements.txt
```

---

## 👩‍💻 Contributing

Feel free to fork this repo and send pull requests!

---

## 💡 Example Usage

A typical session might look like:
- Upload your shift schedule
- Upload volunteer preferences
- Run optimization
- Download results to share with your team
- Run a Monte Carlo sim to check how robust your plan is under random variations

---

## 📧 Questions?

Open an issue in the repo, or reach out!

Happy scheduling 🎉
