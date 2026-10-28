# Effective-emissivity tables (not tracked in git)

`L5_l5c.npz` and `Pallasite_grey.npz` are the Meteorites-TeraFlash effective emissivities e(f) used by the SO forecast
notebook (`notebooks/so_flora_forecast.ipynb`) and the ACT / ALMA comparison scripts. They are generated, not tracked
(`*.npz` is a Git LFS pattern in this repository and new LFS uploads are currently blocked), so regenerate them with the
exporter in the Meteorites-TeraFlash repository, from its directory:

```bash
OUT=/home/jorlo/dev/atm_scattering/data/emissivity
python export_emissivity.py --model-ids L5_l5c        --fmin 20 --fmax 800  --df 5 --out $OUT   # 20-800 GHz
python export_emissivity.py --model-ids Pallasite_grey --fmin 50 --fmax 1500 --df 5 --out $OUT   # 50-1490 GHz (defined above ~49 GHz)
```

Each file holds `f_ghz`, `e` (best fit), `e_draws` (300 posterior draws), `e_lo`, `e_hi`, `omega`, `g` and `fmin_valid`.
