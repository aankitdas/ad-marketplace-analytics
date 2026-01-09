"""
Auction Simulator - Second-price auction mechanics
"""

import numpy as np
import pandas as pd


class AuctionSimulator:
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def run_auction(
        self, 
        advertisers: pd.DataFrame, 
        reserve_price: float = 0.10
    ) -> dict | None:
        """
        Run a second-price auction.
        
        Args:
            advertisers: DataFrame with 'advertiser_id', 'bid', 'quality_score'
            reserve_price: Minimum price to win
            
        Returns:
            Dict with winner info, or None if no valid bids
        """
        if len(advertisers) == 0:
            return None
        
        # Effective bid = bid * quality_score (ad rank)
        bids = advertisers['bid'].values
        quality = advertisers['quality_score'].values
        effective_bids = bids * quality
        
        # Filter out bids below reserve
        valid_mask = bids >= reserve_price
        if not valid_mask.any():
            return None
        
        # Find winner (highest effective bid)
        effective_bids[~valid_mask] = -1  # Exclude invalid
        winner_idx = np.argmax(effective_bids)
        
        # Second-price: winner pays second-highest bid (or reserve)
        sorted_bids = np.sort(bids[valid_mask])[::-1]
        if len(sorted_bids) > 1:
            second_price = max(sorted_bids[1], reserve_price)
        else:
            second_price = reserve_price
        
        price_paid = second_price + 0.01  # Pay $0.01 above second price
        
        winner = advertisers.iloc[winner_idx]
        
        return {
            'winner_id': winner['advertiser_id'],
            'bid': winner['bid'],
            'price_paid': round(price_paid, 4),
            'quality_score': winner['quality_score'],
            'num_participants': valid_mask.sum()
        }


# Test it
if __name__ == '__main__':
    # Create sample auction participants
    participants = pd.DataFrame({
        'advertiser_id': ['adv_001', 'adv_002', 'adv_003'],
        'bid': [0.50, 0.75, 0.60],
        'quality_score': [0.9, 0.7, 0.8]
    })
    
    print("Participants:")
    print(participants)
    print(f"\nEffective bids (bid * quality):")
    print(participants['bid'] * participants['quality_score'])
    
    auction = AuctionSimulator()
    result = auction.run_auction(participants, reserve_price=0.10)
    
    print(f"\nAuction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")