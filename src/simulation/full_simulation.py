"""
Full Simulation - Multiple days with churn
"""

import numpy as np
import pandas as pd
from advertiser_generator import AdvertiserGenerator
from auction import AuctionSimulator


class FullMarketplaceSimulator:
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.auction = AuctionSimulator(seed)
        self.base_ctr = 0.02
        self.base_cvr = 0.05
    
    def run(
        self,
        n_advertisers: int = 200,
        n_days: int = 90,
        impressions_per_day: int = 5000
    ) -> dict:
        """Run full simulation."""
        
        # Generate advertisers
        adv_gen = AdvertiserGenerator(n_advertisers=n_advertisers, seed=42)
        advertisers = adv_gen.generate()
        advertisers['is_churned'] = False
        advertisers['churn_day'] = None
        advertisers['group'] = self.rng.choice(['control', 'treatment'], size=n_advertisers)

        all_events = []
        daily_metrics = []
        
        for day in range(n_days):
            # Progress indicator
            if day % 10 == 0:
                active_count = ((advertisers['signup_day'] <= day) & (~advertisers['is_churned'])).sum()
                print(f"Day {day}/{n_days} | Active advertisers: {active_count}")
            
            # Simulate the day
            # events = self._simulate_day(advertisers, day, impressions_per_day)
            # Experiment runs from day 30-60
            experiment_active = 30 <= day < 60

            if experiment_active:
                # Treatment group: lower reserve price (0.05)
                treatment_advs = advertisers[advertisers['group'] == 'treatment']
                treatment_events = self._simulate_day(treatment_advs, day, impressions_per_day // 2, reserve_price=0.05)
                treatment_events['group'] = 'treatment'
    
                # Control group: normal reserve price (0.10)
                control_advs = advertisers[advertisers['group'] == 'control']
                control_events = self._simulate_day(control_advs, day, impressions_per_day // 2, reserve_price=0.10)
                control_events['group'] = 'control'
    
                events = pd.concat([treatment_events, control_events], ignore_index=True)
            else:
                events = self._simulate_day(advertisers, day, impressions_per_day)
                events['group'] = 'none'
            
            if len(events) > 0:
                all_events.append(events)
                active_today = ((advertisers['signup_day'] <= day) & (~advertisers['is_churned'])).sum()
                # Daily metrics
                daily_metrics.append({
                    'day': day,
                    'impressions': len(events),
                    'revenue': events['price_paid'].sum(),
                    'clicks': events['clicked'].sum(),
                    'conversions': events['converted'].sum(),
                    'active_advertisers': active_today,
                    'unique_winners': events['advertiser_id'].nunique()  # new metric
                })
            
            # Process churn at end of day
            advertisers = self._process_churn(advertisers, day)
            
            if (day + 1) % 30 == 0:
                churned = advertisers['is_churned'].sum()
                print(f"Day {day + 1}: {churned} advertisers churned so far")
        
        return {
            'events': pd.concat(all_events, ignore_index=True),
            'daily_metrics': pd.DataFrame(daily_metrics),
            'advertisers': advertisers
        }
    
    def _simulate_day(
        self,
        advertisers: pd.DataFrame,
        day: int,
        n_impressions: int,
        reserve_price: float = 0.10
    ) -> pd.DataFrame:
        """Simulate one day."""
        
        events = []
        
        # Active = signed up and not churned
        active = advertisers[
            (advertisers['signup_day'] <= day) & 
            (~advertisers['is_churned'])
        ].copy()
        
        if len(active) < 2:
            return pd.DataFrame()
        
        daily_spend = {adv_id: 0.0 for adv_id in active['advertiser_id']}
        
        budget_map = dict(zip(active['advertiser_id'], active['daily_budget']))
        for i in range(n_impressions):
            # Eligible = has budget remaining
            eligible_ids = [
                adv_id for adv_id, spend in daily_spend.items()
                if spend < budget_map.get(adv_id, 0)
            ]
            
            if len(eligible_ids) < 2:
                break
            
            eligible = active[active['advertiser_id'].isin(eligible_ids)].copy()
            
            eligible['bid'] = eligible.apply(self._generate_bid, axis=1)
            
            result = self.auction.run_auction(eligible, reserve_price=reserve_price)
            
            if result is None:
                continue
            
            daily_spend[result['winner_id']] += result['price_paid']
            
            clicked = self.rng.random() < self.base_ctr * result['quality_score']
            converted = clicked and (self.rng.random() < self.base_cvr)
            
            events.append({
                'day': day,
                'impression_id': i,
                'advertiser_id': result['winner_id'],
                'bid': result['bid'],
                'price_paid': result['price_paid'],
                'clicked': clicked,
                'converted': converted
            })
        
        print(f"  Day {day}: {len(events)} impressions, ${sum(e['price_paid'] for e in events):.2f} revenue")

        return pd.DataFrame(events)
    
    def _generate_bid(self, advertiser: pd.Series) -> float:
        base = 0.15 + (advertiser['daily_budget'] / 5000) * 0.3
        noise = self.rng.uniform(0.5, 1.5)
        return round(base * noise, 4)
    
    def _process_churn(self, advertisers: pd.DataFrame, day: int) -> pd.DataFrame:
        """Check if any advertisers churn today."""
        
        for idx, row in advertisers.iterrows():
            if row['is_churned'] or row['signup_day'] > day:
                continue
            
            # Daily churn probability (monthly rate / 30)
            daily_churn = row['churn_probability'] / 30
            
            if self.rng.random() < daily_churn:
                advertisers.at[idx, 'is_churned'] = True
                advertisers.at[idx, 'churn_day'] = day
        
        return advertisers


# Test it
if __name__ == '__main__':
    import time
    start = time.time()
    sim = FullMarketplaceSimulator(seed=42)
    results = sim.run(n_advertisers=200, n_days=90, impressions_per_day=1000)
    elapsed = time.time() - start
    print(f"\n=== TIMING ===")
    print(f"Simulation took {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    events = results['events']
    daily = results['daily_metrics']
    advertisers = results['advertisers']
    
    print(f"\n=== SIMULATION SUMMARY ===")
    print(f"Total impressions: {len(events):,}")
    print(f"Total revenue: ${events['price_paid'].sum():,.2f}")
    print(f"Total clicks: {events['clicked'].sum():,}")
    print(f"Total conversions: {events['converted'].sum():,}")
    
    print(f"\n=== CHURN ANALYSIS ===")
    churned = advertisers[advertisers['is_churned']]
    print(f"Advertisers churned: {len(churned)} / {len(advertisers)}")
    print(f"Churn rate: {len(churned)/len(advertisers)*100:.1f}%")
    
    print(f"\n=== DAILY METRICS (first 10 days) ===")
    print(daily[['day', 'revenue', 'active_advertisers', 'unique_winners']].head(10).to_string(index=False))
    # Check experiment data
    print(f"\n=== EXPERIMENT CHECK ===")
    print(events['group'].value_counts())

    # Compare treatment vs control during experiment (days 30-60)
    experiment_data = events[(events['day'] >= 30) & (events['day'] < 60)]
    if len(experiment_data) > 0:
        print(f"\nDuring experiment (days 30-60):")
        exp_summary = experiment_data.groupby('group').agg({
            'price_paid': ['sum', 'mean'],
            'clicked': 'sum',
            'impression_id': 'count'
        }).round(4)
        print(exp_summary)

    # Save data for analysis
    print(f"\n=== SAVING DATA ===")
    events.to_csv('data/events.csv', index=False)
    daily.to_csv('data/daily_metrics.csv', index=False)
    advertisers.to_csv('data/advertisers.csv', index=False)
    print("Saved to data/ folder")