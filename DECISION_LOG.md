## 2026-01-27T19:02:05+01:00 - Fixed Beta Sign Bug

**Context**: UltraFeedback MODPO showed no trade-off between helpfulness and honesty.

**Options**:
1. Dataset lacks conflict - REJECTED: computed 34% conflict rate
2. RM didn't learn - REJECTED: adapter weights non-zero with std ~0.02
3. Negative beta inverted objective - CONFIRMED

**Decision**: Changed `margin_beta = -script_args.beta` to `margin_beta = script_args.beta` in `scripts/modpo/ultrafeedback/modpo.py`

**Action**: 
- Quick sanity test showed margin rewards flip from +89 to -106 with positive beta
- Full retraining required to validate fix (24h per weight)

**Outcome**: Pending full retrain

**Confidence**: High - matches BeaverTails implementation which showed correct trade-off
