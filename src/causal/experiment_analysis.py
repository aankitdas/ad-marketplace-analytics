"""
Causal Inference - A/B Test Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats


class ExperimentAnalyzer:
    
    def __init__(self, events: pd.DataFrame):
        self.events = events
        self.experiment_data = events[events['group'].isin(['treatment', 'control'])]
    
    def summary(self) -> dict:
        """Basic experiment summary."""
        treatment = self.experiment_data[self.experiment_data['group'] == 'treatment']
        control = self.experiment_data[self.experiment_data['group'] == 'control']
        
        return {
            'treatment_impressions': len(treatment),
            'control_impressions': len(control),
            'treatment_revenue': treatment['price_paid'].sum(),
            'control_revenue': control['price_paid'].sum(),
            'treatment_avg_price': treatment['price_paid'].mean(),
            'control_avg_price': control['price_paid'].mean(),
            'treatment_ctr': treatment['clicked'].mean(),
            'control_ctr': control['clicked'].mean()
        }
    
    def ttest_revenue(self) -> dict:
        """
        T-test comparing average price paid between groups.
        
        Null hypothesis: No difference between treatment and control.
        """
        treatment = self.experiment_data[self.experiment_data['group'] == 'treatment']['price_paid']
        control = self.experiment_data[self.experiment_data['group'] == 'control']['price_paid']
        
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((treatment.std()**2 + control.std()**2) / 2)
        cohens_d = (treatment.mean() - control.mean()) / pooled_std
        
        # Confidence interval for difference
        diff = treatment.mean() - control.mean()
        se = np.sqrt(treatment.var()/len(treatment) + control.var()/len(control))
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_difference': diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'significant': p_value < 0.05
        }
    
    def ttest_ctr(self) -> dict:
        """T-test comparing click-through rates."""
        treatment = self.experiment_data[self.experiment_data['group'] == 'treatment']['clicked']
        control = self.experiment_data[self.experiment_data['group'] == 'control']['clicked']
        
        t_stat, p_value = stats.ttest_ind(treatment, control)
        
        diff = treatment.mean() - control.mean()
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'treatment_ctr': treatment.mean(),
            'control_ctr': control.mean(),
            'ctr_lift': diff / control.mean() * 100 if control.mean() > 0 else 0,
            'significant': p_value < 0.05
        }
    
    def difference_in_differences(self, daily_metrics: pd.DataFrame) -> dict:
        """
        Difference-in-Differences analysis.
        
        Compares pre/post changes between treatment and control.
        """
        # This requires daily data split by group
        # For now, we'll do a simplified version using event data
        
        pre_period = self.events[self.events['day'] < 30]
        post_period = self.experiment_data  # days 30-60
        
        # Pre-period: everyone is in 'none' group, so we use advertiser assignment
        # This is a simplified DiD - in production you'd track by group pre-assignment
        
        treatment_post = post_period[post_period['group'] == 'treatment']['price_paid'].mean()
        control_post = post_period[post_period['group'] == 'control']['price_paid'].mean()
        
        # Assuming pre-period means were equal (randomization)
        pre_mean = pre_period['price_paid'].mean()
        
        did_estimate = (treatment_post - pre_mean) - (control_post - pre_mean)
        
        return {
            'pre_period_mean': pre_mean,
            'treatment_post_mean': treatment_post,
            'control_post_mean': control_post,
            'did_estimate': did_estimate,
            'interpretation': 'Treatment effect on average price paid'
        }
    
    def full_report(self, daily_metrics: pd.DataFrame = None) -> None:
        """Print full experiment analysis report."""
        
        print("=" * 60)
        print("EXPERIMENT ANALYSIS REPORT")
        print("=" * 60)
        
        # Summary
        print("\n1. EXPERIMENT SUMMARY")
        print("-" * 40)
        summary = self.summary()
        print(f"Treatment impressions: {summary['treatment_impressions']:,}")
        print(f"Control impressions:   {summary['control_impressions']:,}")
        print(f"Treatment revenue:     ${summary['treatment_revenue']:,.2f}")
        print(f"Control revenue:       ${summary['control_revenue']:,.2f}")
        print(f"Revenue lift:          ${summary['treatment_revenue'] - summary['control_revenue']:,.2f} ({(summary['treatment_revenue']/summary['control_revenue']-1)*100:.1f}%)")
        
        # T-test on revenue
        print("\n2. STATISTICAL TEST: Revenue per Impression")
        print("-" * 40)
        ttest = self.ttest_revenue()
        print(f"Treatment avg price:   ${ttest['mean_difference'] + summary['control_avg_price']:.4f}")
        print(f"Control avg price:     ${summary['control_avg_price']:.4f}")
        print(f"Difference:            ${ttest['mean_difference']:.4f}")
        print(f"95% CI:                [${ttest['ci_95_lower']:.4f}, ${ttest['ci_95_upper']:.4f}]")
        print(f"T-statistic:           {ttest['t_statistic']:.2f}")
        print(f"P-value:               {ttest['p_value']:.4f}")
        print(f"Effect size (Cohen's d): {ttest['cohens_d']:.3f}")
        print(f"Significant at α=0.05: {'YES ✓' if ttest['significant'] else 'NO'}")
        
        # T-test on CTR
        print("\n3. STATISTICAL TEST: Click-Through Rate")
        print("-" * 40)
        ctr_test = self.ttest_ctr()
        print(f"Treatment CTR:         {ctr_test['treatment_ctr']*100:.2f}%")
        print(f"Control CTR:           {ctr_test['control_ctr']*100:.2f}%")
        print(f"CTR lift:              {ctr_test['ctr_lift']:.1f}%")
        print(f"P-value:               {ctr_test['p_value']:.4f}")
        print(f"Significant at α=0.05: {'YES ✓' if ctr_test['significant'] else 'NO'}")
        
        # DiD
        if daily_metrics is not None:
            print("\n4. DIFFERENCE-IN-DIFFERENCES")
            print("-" * 40)
            did = self.difference_in_differences(daily_metrics)
            print(f"Pre-period avg price:  ${did['pre_period_mean']:.4f}")
            print(f"Treatment post avg:    ${did['treatment_post_mean']:.4f}")
            print(f"Control post avg:      ${did['control_post_mean']:.4f}")
            print(f"DiD estimate:          ${did['did_estimate']:.4f}")
        
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        if ttest['significant']:
            print(f"The treatment (lower reserve price) had a SIGNIFICANT effect.")
            print(f"Average revenue per impression increased by ${ttest['mean_difference']:.4f}")
            print(f"This translates to ~${ttest['mean_difference'] * summary['treatment_impressions']:.2f} additional revenue")
            print(f"over the experiment period.")
        else:
            print("No statistically significant difference detected between groups.")


# Test it
if __name__ == '__main__':
    # Load saved data
    events = pd.read_parquet('data/events.parquet')
    daily = pd.read_parquet('data/daily_metrics.parquet')
    
    analyzer = ExperimentAnalyzer(events)
    analyzer.full_report(daily)