"""Experiment partition of the original supplementary run; no parameter retuning."""
SPECS = {
 'ridge_lambda_sensitivity': ('ridge', False),
 'student_t_df_sensitivity': ('df', False),
 'btc_exhaustive_lag_pairs': ('pairs', True),
 'btc_sampled_lag_triples': ('triples', True),
 'posterior_threshold_sensitivity': ('threshold', False),
 'transaction_cost_sensitivity': ('cost', False),
 'btc_raw_pnl_failure_analysis': ('failure', True),
 'btc_lag_selector_comparison': ('selectors', True),
}
def pools_for(name):
 family,_=SPECS[name]
 return {'sampled','all_pairs'} if family=='pairs' else ({'sampled','three_lags'} if family=='triples' else {'sampled'})
def methods_for(name):
 family,_=SPECS[name]
 return ('raw_pnl',) if family=='failure' else (('raw_pnl','raw_sharpe','wilcoxon') if family=='selectors' else ('raw_sharpe',))
def configs_for(name, configs):
 family,_=SPECS[name]
 allowed={'reference', 'failure_zero_cost' if family=='failure' else family}
 return [c for c in configs if c[0] in allowed]
