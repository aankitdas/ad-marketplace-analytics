# Ad Marketplace Analytics - Development Log

## Goal
Build an end-to-end ad marketplace simulation to demonstrate:
- Marketplace/auction mechanics
- Causal inference (A/B testing, difference-in-differences)
- Churn prediction
- Analytics dashboard

Target: Deployable on HF Spaces, completed in 2 days.

---

## Progress

### Step 1: Project Setup
- Initialized with `uv`
- Dependencies: numpy, pandas, scikit-learn, xgboost, streamlit, plotly, statsmodels, duckdb

### Step 2: Advertiser Generator
Built `advertiser_generator.py` to create synthetic advertiser profiles.

Key design decisions:
- **Budget**: Power law distribution (few big spenders, many small) — matches real ad marketplaces
- **Quality score**: Beta(5,2) distribution — skewed toward higher quality since bad advertisers quit
- **Churn probability**: Inversely related to budget — big spenders stick around longer

### Step 3: Auction System
Built `auction.py` implementing second-price auctions.

How it works:
- Effective bid = bid × quality_score (ad rank)
- Winner = highest effective bid
- Price paid = second-highest bid + $0.01

Why second-price? Encourages honest bidding. You pay less than you bid, so no incentive to game it.

### Step 4: Single Day Simulation
Built `marketplace.py` to run 5,000 auctions per day.

### Step 5: Full 90-Day Simulation
Built `full_simulation.py` with churn tracking.

---

## Bugs & Fixes

### Bug 1: Simulation too slow
**Problem**: Each day took forever to run.

**Cause**: This line inside the impression loop:
```python
eligible = active[active.apply(lambda r: daily_spend[r['advertiser_id']] < r['daily_budget'], axis=1)]
```
`.apply()` with `axis=1` is slow — basically a Python for-loop. 5,000 impressions × 200 advertisers × 90 days = 90 million operations.

**Fix**: Pre-compute budget map, use list comprehension:
```python
budget_map = dict(zip(active['advertiser_id'], active['daily_budget']))
eligible_ids = [adv_id for adv_id, spend in daily_spend.items() if spend < budget_map.get(adv_id, 0)]
```

### Bug 2: No progress feedback
**Problem**: Script ran silently, no idea if it was working.

**Fix**: Added progress prints every 10 days + per-day metrics.

### Bug 3: Misleading `active_advertisers` metric
**Problem**: Daily metrics showed `active_advertisers: 1` even when 30+ were active.

**Cause**: Was counting unique *winners*, not active advertisers. One big spender won everything.

**Fix**: Separated metrics:
- `active_advertisers`: Signed up and not churned
- `unique_winners`: Actually won at least one auction

### Bug 4: One advertiser dominates all auctions
**Problem**: From day 2 onward, only 1 unique winner per day.

**Cause**: Bid formula favored big budgets too heavily:
```python
base = 0.10 + (daily_budget / 1000) * 0.5
# $750 budget → $0.475 bid
# $50 budget → $0.125 bid
```
Big spender bids 4x higher, wins everything.

**Fix**: Flattened bid curve, added more randomness:
```python
base = 0.15 + (daily_budget / 5000) * 0.3
noise = self.rng.uniform(0.5, 1.5)  # Was (0.8, 1.2)
```

---

### Step 6: A/B Experiment
Added treatment vs control groups to simulate a pricing experiment.

- **Treatment**: Lower reserve price ($0.05)
- **Control**: Standard reserve price ($0.10)
- **Period**: Days 30-60

Implementation:
- Split advertisers 50/50 into groups at start
- During experiment period, run separate auctions for each group
- Track `group` column in events data

### Step 7: Causal Inference Module
Built `src/causal/experiment_analysis.py` with statistical analysis.

**Methods implemented:**
- T-test for revenue difference
- T-test for CTR difference
- Difference-in-Differences estimator
- Effect size (Cohen's d)
- 95% confidence intervals

**Results from our experiment:**
- Revenue lift: 5.5% ($1,375 more in treatment)
- P-value: 0.0000 (highly significant)
- Cohen's d: 0.340 (medium effect size)
- CTR lift: 11.5% (also significant, p=0.009)

### Step 8: Churn Prediction Model
Built `src/models/churn_model.py` using XGBoost.

**Features used:**
- daily_budget, quality_score, signup_day
- total_spend, avg_price, total_clicks, ctr
- total_conversions, total_impressions, active_days
- spend_per_day, budget_utilization, impressions_per_day
- vertical_encoded, bid_strategy_encoded

**Model performance:**
- AUC-ROC: 0.634
- Precision: 0.638
- Recall: 0.638

**Top predictors:**
1. active_days
2. avg_price
3. quality_score
4. daily_budget

### Step 9: Streamlit Dashboard
Built `dashboard/app.py` (later moved to `app.py` for HF Spaces).

**Pages:**
- Overview: Key metrics, daily revenue chart, advertiser breakdown
- Experiment Analysis: Treatment vs control comparison, statistical tests
- Churn Analysis: Churn rates by vertical/budget, timing distribution
- Raw Data: Data explorer for all tables

### Step 10: Deployment to HF Spaces

**Issue 1: Binary files rejected**
HF Spaces blocks binary files (parquet) by default.

Options considered:
- Git LFS
- Xet storage (HF's new system)
- Generate data on startup ← chose this

**Fix**: Modified `app.py` to generate data if not present:
```python
if not os.path.exists('data/events.parquet'):
    generate_data()
```

**Issue 2: Git history contained binary files**
Even after `git rm`, push was rejected because files were in history.

**Fix**: 
```bash
git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch data/" --prune-empty -- --all
```

**Issue 3: Docker dependencies**
`uv sync` wasn't installing packages correctly in Docker.

**Fix**: Switched to simple `requirements.txt` approach:
```dockerfile
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
```

---

## Bugs & Fixes (Deployment)

### Bug 5: Parquet files rejected by HF Spaces
**Problem**: `remote: Your push was rejected because it contains binary files`

**Cause**: HF Spaces requires Xet or LFS for binary files.

**Fix**: Generate data on app startup instead of uploading it.

### Bug 6: Binary files still in git history
**Problem**: After `git rm -r --cached data/`, push still rejected.

**Cause**: Files exist in previous commits.

**Fix**: Used `git filter-branch` to purge from all history.

### Bug 7: Plotly import error in HF Spaces
**Problem**: `ModuleNotFoundError: No module named 'plotly'`

**Cause**: `uv sync` in Dockerfile not installing dependencies properly.

**Fix**: Switched to `requirements.txt` with `pip install`.

---

## Final File Structure
```
ad-marketplace-analytics/
├── app.py                 # Streamlit dashboard (main entry)
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── README.md
├── DEVLOG.md
├── src/
│   ├── simulation/
│   │   ├── advertiser_generator.py
│   │   ├── auction.py
│   │   ├── marketplace.py
│   │   └── full_simulation.py
│   ├── causal/
│   │   └── experiment_analysis.py
│   └── models/
│       └── churn_model.py
└── dashboard/
    └── app.py             # Original dashboard location
```

---

## Key Learnings

1. **Pandas `.apply()` is slow** — vectorize or use list comprehensions
2. **Second-price auctions** encourage honest bidding (Vickrey auction)
3. **Power law distributions** better represent real-world advertiser spending
4. **Causal inference** requires proper experiment design (treatment/control split)
5. **HF Spaces deployment** has quirks with binary files — generate on startup if possible
6. **Git history** persists deleted files — use `filter-branch` to fully remove

---

## Resume Bullet Points

From this project, you can claim:

> Built end-to-end ad auction simulator modeling advertiser lifecycle, bidding dynamics, and matching algorithms; generated 600K+ synthetic impression events with realistic seasonality and churn patterns

> Implemented causal inference framework (A/B testing, DiD) to measure pricing policy impacts; demonstrated 5.5% revenue lift from reserve price optimization with p<0.001

> Developed churn prediction model (XGBoost, AUC 0.634) identifying key predictors: active days, average price paid, quality score

> Built interactive Streamlit dashboard for marketplace health monitoring; deployed to Hugging Face Spaces
