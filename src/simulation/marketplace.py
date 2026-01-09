"""
Marketplace Simulator - Full day simulation
"""

import numpy as np
import pandas as pd
from advertiser_generator import AdvertiserGenerator
from auction import AuctionSimulator


class MarketplaceSimulator:
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.auction = AuctionSimulator(seed)
        
        # Marketplace parameters
        self.base_ctr = 0.02  # 2% click-through rate
        self.base_cvr = 0.05  # 5% conversion rate (of clicks)
    
    def simulate_day(
        self,
        advertisers: pd.DataFrame,
        day: int,
        n_impressions: int = 10000,
        reserve_price: float = 0.10
    ) -> pd.DataFrame:
        """
        Simulate one day of ad auctions.
        
        Returns DataFrame of impression events.
        """
        events = []
        
        # Filter to active advertisers (signed up by this day)
        active = advertisers[advertisers['signup_day'] <= day].copy()
        
        if len(active) == 0:
            return pd.DataFrame()
        
        # Track daily spend per advertiser
        daily_spend = {adv_id: 0.0 for adv_id in active['advertiser_id']}
        
        for i in range(n_impressions):
            # Filter advertisers with remaining budget
            eligible_ids = [
                adv_id for adv_id, spend in daily_spend.items()
                if spend < active[active['advertiser_id'] == adv_id]['daily_budget'].values[0]
            ]
            
            if len(eligible_ids) < 2:  # Need at least 2 for auction
                break
            
            eligible = active[active['advertiser_id'].isin(eligible_ids)].copy()
            
            # Generate bids (based on budget and some randomness)
            eligible['bid'] = eligible.apply(
                lambda row: self._generate_bid(row), axis=1
            )
            
            # Run auction
            result = self.auction.run_auction(eligible, reserve_price)
            
            if result is None:
                continue
            
            # Update spend
            daily_spend[result['winner_id']] += result['price_paid']
            
            # Simulate click and conversion
            clicked = self.rng.random() < self.base_ctr * result['quality_score']
            converted = clicked and (self.rng.random() < self.base_cvr)
            
            events.append({
                'day': day,
                'impression_id': i,
                'advertiser_id': result['winner_id'],
                'bid': result['bid'],
                'price_paid': result['price_paid'],
                'quality_score': result['quality_score'],
                'num_participants': result['num_participants'],
                'clicked': clicked,
                'converted': converted
            })
        
        return pd.DataFrame(events)
    
    def _generate_bid(self, advertiser: pd.Series) -> float:
        """Generate a bid based on advertiser profile."""
        # Base bid scales with budget
        base = 0.10 + (advertiser['daily_budget'] / 1000) * 0.5
        # Add randomness
        noise = self.rng.uniform(0.8, 1.2)
        return round(base * noise, 4)


# Test it
if __name__ == '__main__':
    # Generate advertisers
    adv_gen = AdvertiserGenerator(n_advertisers=50, seed=42)
    advertisers = adv_gen.generate()
    
    # Simulate day 15 (so most advertisers have signed up)
    sim = MarketplaceSimulator(seed=42)
    events = sim.simulate_day(advertisers, day=15, n_impressions=5000)
    
    print(f"Total impressions: {len(events)}")
    print(f"Total revenue: ${events['price_paid'].sum():.2f}")
    print(f"Total clicks: {events['clicked'].sum()}")
    print(f"Total conversions: {events['converted'].sum()}")
    print(f"\nCTR: {events['clicked'].mean()*100:.2f}%")
    print(f"CVR (of clicks): {events[events['clicked']]['converted'].mean()*100:.1f}%")
    
    print(f"\nTop 5 spenders:")
    top_spenders = events.groupby('advertiser_id')['price_paid'].sum().sort_values(ascending=False).head()
    print(top_spenders)