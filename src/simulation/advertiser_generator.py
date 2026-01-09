"""
Advertiser Generator - Creates synthetic advertiser profiles
"""

import numpy as np
import pandas as pd


class AdvertiserGenerator:
    
    VERTICALS = ['gaming', 'ecommerce', 'finance', 'travel', 'food', 'entertainment']
    BID_STRATEGIES = ['maximize_clicks', 'maximize_conversions', 'target_cpa']
    
    def __init__(self, n_advertisers: int = 500, seed: int = 42):
        self.n_advertisers = n_advertisers
        self.rng = np.random.default_rng(seed)
    
    def generate(self) -> pd.DataFrame:
        n = self.n_advertisers
        
        # Daily budget follows power law (few big spenders, many small)
        daily_budget = self.rng.pareto(1.5, n) * 100 + 50
        daily_budget = np.clip(daily_budget, 50, 50000)
        
        # Quality score (affects ad ranking)
        quality_score = self.rng.beta(5, 2, n)
        
        # Churn probability (big spenders less likely to churn)
        budget_normalized = (daily_budget - daily_budget.min()) / (daily_budget.max() - daily_budget.min())
        churn_probability = 0.15 * (1.5 - budget_normalized)
        
        advertisers = pd.DataFrame({
            'advertiser_id': [f'adv_{i:04d}' for i in range(n)],
            'vertical': self.rng.choice(self.VERTICALS, n),
            'daily_budget': daily_budget.round(2),
            'bid_strategy': self.rng.choice(self.BID_STRATEGIES, n),
            'quality_score': quality_score.round(3),
            'churn_probability': churn_probability.round(4),
            'signup_day': self.rng.integers(0, 30, n)
        })
        
        return advertisers


# Test it
if __name__ == '__main__':
    gen = AdvertiserGenerator(n_advertisers=100)
    df = gen.generate()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nBudget distribution:\n{df['daily_budget'].describe()}")