# Calibration-teacher budget audit

Generated from the locked v10 validation-only audit and the frozen raw-ridge follow-up. The empirical inverse uses `[1,m,mτ]→RGB`; raw ridge uses `[1,m,τ]→RGB`.

| K | affine forward | diagonal Malus | full Malus | matched empirical inverse | raw ridge |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.537702 | 0.902629 | 0.685145 | 0.891036 | 0.937562 |
| 80 | 0.654799 | 0.885884 | 0.897219 | 0.895091 | 0.949193 |
| 800 | 0.675140 | 0.887118 | 0.877234 | 0.904961 | -- |
| 8000 | 0.675833 | 0.887640 | 0.867489 | 0.905512 | -- |

## K80 direct R²

| teacher | pooled mean direct R² |
|---|---:|
| full Malus | 0.714323 |
| matched empirical inverse | 0.784455 |
