# src/library/feature_registry.py

# 1. Импорты из сессионных и временных фич
from src.library.time_features import (
    add_judas_swing_context,
    add_ny_news_sweep_context,
    add_london_fix_fade_context,
    add_ny_continuation_context,
    add_asian_sweep_context,
    add_ny_expansion_context,
    add_ny_sr_touch_context,
    add_asian_sr_alignment_context,
    add_london_counter_fractal_context,
    add_london_pdh_pdl_sweep_context,
    add_weekend_gap_context,
    add_friday_reversal_context,
    add_monday_gap_reversion_context,
    add_turnaround_tuesday_context,
    add_wednesday_fakeout_context,
    add_thursday_expansion_context,
    add_london_true_trend_context,
    add_asian_box_breakout_context,
    add_tokyo_trap_context,
    add_pure_algo_vol_crush_context
)

# 2. Импорты из макроэкономических фич
from src.library.macro_features import (
    add_geopolitical_shock_context,
    add_election_volatility_context,
    add_uk_political_shock_context,
    add_boe_hawkish_context,
    add_uk_cpi_momentum_context,
    add_sovereign_risk_proxy_context,
    add_boe_tone_shift_proxy_context,
    add_macro_shock_inside_bar_context,
    add_nfp_divergence_context,
    add_nfp_revision_trap_context,
    add_cpi_match_mean_reversion_context,
    add_cb_divergence_state_context,
    add_fomc_sell_the_news_context,
    add_uk_us_cpi_divergence_context,
    add_unemp_fakeout_context,
    add_retail_sales_divergence_context
)

# 3. Импорты из структурных фич
from src.library.structure_features import (
    calculate_multi_tf_fvgs,
    add_previous_boundaries,
    add_weekly_swing_context,
    add_1w_swing_context,
    add_1d_swing_context,
    add_htf_trend_probability,
    add_fvg_order_flow_context,
    add_fvg_sr_confluence_context,
    add_weekly_floor_context,
    add_1w_level_rejection_context,
    add_asia_fvg_protection_context
)

# Маппинг: "Название колонки в JSON" -> "Функция-обработчик"
# Примечание: Если функция создает несколько колонок (Long/Short), 
# мы маппим каждую колонку на одну и ту же функцию.
FEATURE_REGISTRY = {
    # --- TIME & SESSION FEATURES ---
    "Judas_Short": add_judas_swing_context,
    "Judas_Long": add_judas_swing_context,
    "NY_Sweep_Short": add_ny_news_sweep_context,
    "NY_Sweep_Long": add_ny_news_sweep_context,
    "Fix_Fade_Short": add_london_fix_fade_context,
    "Fix_Fade_Long": add_london_fix_fade_context,
    "NY_Cont_Short": add_ny_continuation_context,
    "NY_Cont_Long": add_ny_continuation_context,
    "Asian_Sweep_Long": add_asian_sweep_context,
    "Asian_Sweep_Short": add_asian_sweep_context,
    "NY_Expansion_Long": add_ny_expansion_context,
    "NY_Expansion_Short": add_ny_expansion_context,
    "NY_SR_Touch": add_ny_sr_touch_context,
    "Asian_SR_Alignment": add_asian_sr_alignment_context,
    "LO_Counter_Fractal": add_london_counter_fractal_context,
    "LO_PDH_PDL_Sweep": add_london_pdh_pdl_sweep_context,
    "Weekend_Gap": add_weekend_gap_context,
    "Friday_Reversal": add_friday_reversal_context,
    "Monday_Gap_Reversion": add_monday_gap_reversion_context,
    "Tuesday_Resumption": add_turnaround_tuesday_context,
    "Wed_Fakeout": add_wednesday_fakeout_context,
    "Thursday_Expansion": add_thursday_expansion_context,
    "LO_True_Trend": add_london_true_trend_context,
    "Asian_Box_Breakout": add_asian_box_breakout_context,
    "Tokyo_Trap": add_tokyo_trap_context,
    "Algo_Vol_Crush": add_pure_algo_vol_crush_context,

    # --- MACRO FEATURES ---
    "Geo_Shock": add_geopolitical_shock_context,
    "Election_Vol": add_election_volatility_context,
    "UK_Pol_Shock": add_uk_political_shock_context,
    "BoE_Hawkish": add_boe_hawkish_context,
    "UK_CPI_Momentum": add_uk_cpi_momentum_context,
    "Sovereign_Risk": add_sovereign_risk_proxy_context,
    "BoE_Tone_Shift": add_boe_tone_shift_proxy_context,
    "Macro_Inside_Bar": add_macro_shock_inside_bar_context,
    "NFP_Divergence": add_nfp_divergence_context,
    "NFP_Revision_Trap": add_nfp_revision_trap_context,
    "CPI_Match_Reversion": add_cpi_match_mean_reversion_context,
    "CB_Divergence": add_cb_divergence_state_context,
    "FOMC_Sell_News": add_fomc_sell_the_news_context,
    "UK_US_CPI_Div": add_uk_us_cpi_divergence_context,
    "Unemp_Fakeout": add_unemp_fakeout_context,
    "Retail_Sales_Div": add_retail_sales_divergence_context,

    # --- STRUCTURE FEATURES ---
    "Multi_TF_FVG": calculate_multi_tf_fvgs,
    "Prev_Boundaries": add_previous_boundaries,
    "Weekly_Swing": add_weekly_swing_context,
    "1W_Swing": add_1w_swing_context,
    "1D_Swing": add_1d_swing_context,
    "HTF_Trend_Prob": add_htf_trend_probability,
    "FVG_Order_Flow": add_fvg_order_flow_context,
    "FVG_SR_Confluence": add_fvg_sr_confluence_context,
    "Weekly_Floor": add_weekly_floor_context,
    "1W_Level_Rejection": add_1w_level_rejection_context,
    "Asia_FVG_Protection": add_asia_fvg_protection_context
}