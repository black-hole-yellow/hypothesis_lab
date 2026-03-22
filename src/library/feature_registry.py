# src/library/feature_registry.py

from src.library.time_features import *
from src.library.macro_features import *
from src.library.structure_features import *

# Типы данных для строгой логики
BOOL = "boolean"      # 1 или 0 (сигналы)
FLOAT = "float"     # Ценовые уровни, ATR, вероятности
CAT = "categorical" # Режимы (0, 1, 2)

FEATURE_REGISTRY = {
    # --- SESSION / TIME (time_features.py) ---
    "Judas_Short": {"func": add_judas_swing_context, "type": BOOL},
    "Judas_Long": {"func": add_judas_swing_context, "type": BOOL},
    "Fix_Fade_Short": {"func": add_london_fix_fade_context, "type": BOOL},
    "Fix_Fade_Long": {"func": add_london_fix_fade_context, "type": BOOL},
    "NY_Sweep_Short": {"func": add_ny_news_sweep_context, "type": BOOL},
    "NY_Sweep_Long": {"func": add_ny_news_sweep_context, "type": BOOL},
    "NY_Cont_Short": {"func": add_ny_continuation_context, "type": BOOL},
    "NY_Cont_Long": {"func": add_ny_continuation_context, "type": BOOL},
    
    "Asian_Box_Long": {"func": add_asian_box_breakout_context, "type": BOOL},
    "Asian_Box_Short": {"func": add_asian_box_breakout_context, "type": BOOL},
    "Tokyo_Trap_Long": {"func": add_tokyo_trap_context, "type": BOOL},
    "Tokyo_Trap_Short": {"func": add_tokyo_trap_context, "type": BOOL},
    "Friday_Reversal_Long": {"func": add_friday_reversal_context, "type": BOOL},
    "Friday_Reversal_Short": {"func": add_friday_reversal_context, "type": BOOL},
    "Monday_Reversion_Long": {"func": add_monday_gap_reversion_context, "type": BOOL},
    "Monday_Reversion_Short": {"func": add_monday_gap_reversion_context, "type": BOOL},
    "Tuesday_Resumption_Long": {"func": add_turnaround_tuesday_context, "type": BOOL},
    "Tuesday_Resumption_Short": {"func": add_turnaround_tuesday_context, "type": BOOL},
    "Wed_Fakeout_Long": {"func": add_wednesday_fakeout_context, "type": BOOL},
    "Wed_Fakeout_Short": {"func": add_wednesday_fakeout_context, "type": BOOL},
    "Thursday_Trend_Long": {"func": add_thursday_expansion_context, "type": BOOL},
    "Thursday_Trend_Short": {"func": add_thursday_expansion_context, "type": BOOL},
    "LO_True_Trend_Long": {"func": add_london_true_trend_context, "type": BOOL},
    "LO_True_Trend_Short": {"func": add_london_true_trend_context, "type": BOOL},
    "Algo_Vol_Crush_Long": {"func": add_pure_algo_vol_crush_context, "type": BOOL},
    "Algo_Vol_Crush_Short": {"func": add_pure_algo_vol_crush_context, "type": BOOL},
    
    # Session Additions (Симметрия + Типы)
    "NY_Opened_In_Asia_Range": {"func": add_ny_sr_touch_context, "type": BOOL},
    "NY_Sweep_Asia_High": {"func": add_asian_sweep_context, "type": BOOL},
    "NY_Sweep_Asia_Low": {"func": add_asian_sweep_context, "type": BOOL},
    "First_LDN_PDL_Long": {"func": add_london_pdh_pdl_sweep_context, "type": BOOL},
    "First_LDN_PDH_Short": {"func": add_london_pdh_pdl_sweep_context, "type": BOOL},
    "LDN_Protected_AL_Long": {"func": add_asia_fvg_protection_context, "type": BOOL},
    "LDN_Protected_AH_Short": {"func": add_asia_fvg_protection_context, "type": BOOL},

    # --- MACRO & NEWS (macro_features.py) ---
    # Исправлена асимметрия: добавлены недостающие стороны
    "Geo_Shock_Long": {"func": add_geopolitical_shock_context, "type": BOOL}, # Добавлено
    "Geo_Shock_Short": {"func": add_geopolitical_shock_context, "type": BOOL},
    "Election_Vol_Crush_Long": {"func": add_election_volatility_context, "type": BOOL}, # Добавлено
    "Election_Vol_Crush_Short": {"func": add_election_volatility_context, "type": BOOL},
    "UK_Shock_Cont_Long": {"func": add_uk_political_shock_context, "type": BOOL},
    "UK_Shock_Cont_Short": {"func": add_uk_political_shock_context, "type": BOOL},
    "CPI_Momentum_Long": {"func": add_uk_cpi_momentum_context, "type": BOOL},
    "CPI_Momentum_Short": {"func": add_uk_cpi_momentum_context, "type": BOOL},
    "Macro_CPI_Div_Long": {"func": add_uk_us_cpi_divergence_context, "type": BOOL},
    "Macro_CPI_Div_Short": {"func": add_uk_us_cpi_divergence_context, "type": BOOL}, # Добавлено
    "NFP_Fade_Long": {"func": add_nfp_divergence_context, "type": BOOL},
    "NFP_Fade_Short": {"func": add_nfp_divergence_context, "type": BOOL},
    "NFP_Resumption_Long": {"func": add_nfp_divergence_context, "type": BOOL},
    "NFP_Resumption_Short": {"func": add_nfp_divergence_context, "type": BOOL},
    "CB_Divergence_Long": {"func": add_cb_divergence_state_context, "type": BOOL},
    "CB_Divergence_Short": {"func": add_cb_divergence_state_context, "type": BOOL},
    "BoE_Tone_Shift_Long": {"func": add_boe_tone_shift_proxy_context, "type": BOOL}, # Добавлено
    "BoE_Tone_Shift_Short": {"func": add_boe_tone_shift_proxy_context, "type": BOOL},
    "FOMC_Sell_News_Long": {"func": add_fomc_sell_the_news_context, "type": BOOL},
    "FOMC_Sell_News_Short": {"func": add_fomc_sell_the_news_context, "type": BOOL}, # Добавлено
    "Retail_Div_Long": {"func": add_retail_sales_divergence_context, "type": BOOL},
    "Retail_Div_Short": {"func": add_retail_sales_divergence_context, "type": BOOL}, # Добавлено
    "Unemp_Fakeout_Long": {"func": add_unemp_fakeout_context, "type": BOOL},
    "Unemp_Fakeout_Short": {"func": add_unemp_fakeout_context, "type": BOOL}, # Добавлено
    "Sovereign_Risk_Long": {"func": add_sovereign_risk_proxy_context, "type": BOOL}, # Добавлено
    "Sovereign_Risk_Short": {"func": add_sovereign_risk_proxy_context, "type": BOOL},

    # --- STRUCTURE & PA (structure_features.py) ---
    "Multi_TF_FVG": {"func": calculate_multi_tf_fvgs, "type": CAT},
    "HTF_Trend_Prob": {"func": add_htf_trend_probability, "type": FLOAT},
    "First_1W_Rej_Long": {"func": add_1w_level_rejection_context, "type": BOOL},
    "First_1W_Rej_Short": {"func": add_1w_level_rejection_context, "type": BOOL},
    "Swept_AL_Into_FVG": {"func": add_asia_fvg_protection_context, "type": BOOL},
    "Swept_AH_Into_FVG": {"func": add_asia_fvg_protection_context, "type": BOOL},
    "1h_Bullish_Flip": {"func": add_htf_trend_probability, "type": BOOL},
    "1h_Bearish_Flip": {"func": add_htf_trend_probability, "type": BOOL},
    
    # Специфические числовые значения
    "UA_Hour": {"func": None, "type": CAT}, # Генерируется движком
    "Optimal_Exit_Price": {"func": None, "type": FLOAT}
}