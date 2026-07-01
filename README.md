# Antarctic Coastal Polynya Detection from SWOT & ICESat-2

Analysis code for:

> **Resolving sea ice production and retention at kilometre scales in Ross Sea coastal polynyas**  
> Lu Zhou, Andrew F. Thompson, Marte Hofsteenge, Sohey Nihashi, Shiming Xu, Michiel van den Broeke  
> *Science Advances* (submitted 2026)

---

## Overview

This repository contains the Python analysis code used to combine SWOT KaRIn swath observations with near-coincident ICESat-2 ATL07 profiles to resolve kilometre-scale sea ice structure in Ross Sea coastal polynyas. The approach retrieves coherent thin ice thickness snapshots and interprets them as a residual retained-ice state rather than an instantaneous freezing rate.

---

## Repository structure

```
.
├── data_processing/
│   ├── swot_classification.py        # KaRIn σ₀ de-striping and K-means surface-state classification
│   ├── icesat2_retrieval.py          # ATL07 loading, sea surface reference, freeboard & SIT
│   ├── colocation.py                 # SWOT ↔ IS-2 spatial/temporal co-location (|Δt| ≤ 2 h)
│   ├── amsr2_polynya_mask.py         # AMSR2/ASI SIC loading and SIC < 70 % polynya masking
│   ├── drift_pooling.py              # Polar Pathfinder drift-aware thickness pooling
│   ├── racmo_fluxes.py               # RACMO2.4p1 net heat flux → ice production equivalent
│   └── hssw_transformation.py        # Brine rejection → HSSW volume transformation rate
├── figures/
│   ├── plot_fig2_spatial_context.py  # Fig. 2: SWOT σ₀ + IS-2 height overview maps
│   ├── plot_fig3_classification.py   # Fig. 3: SWOT SSHA + surface-state classification maps
│   ├── plot_fig4_sit_retrieval.py    # Fig. 4: along-track sea surface reference and SIT
│   ├── plot_fig5_drift_collocation.py# Fig. 5: drift-aware backward-trajectory collocation
│   ├── plot_fig6_hssw_production.py  # Fig. 6: thickness time series, production rates, HSSW
│   ├── plot_racmo_antarctic_winds.py # RACMO near-surface wind speed (contourf) + vectors
│   └── figure1_v6.tex                # Fig. 1: TikZ conceptual cross-section schematic
├── utils/
│   └── vertical_reference.py        # SWOT–IS-2 mean sea surface alignment helpers
├── environment.yml                   # Conda environment specification
└── README.md
```

---

## Data requirements

| Dataset | Source |
|---------|--------|
| SWOT L3 KaRIn Low-Rate Unsmoothed v2.0.1 | [AVISO/DUACS](https://www.aviso.altimetry.fr) |
| ICESat-2 ATL07 Release 007 | [NSIDC](https://nsidc.org/data/ATL07) |
| AMSR2 ASI 3.125 km SIC | [Univ. Bremen](https://seaice.uni-bremen.de) |
| ESA CCI sea ice thickness | [ESA Climate Office](https://climate.esa.int/en/projects/sea-ice/) |
| AMSR-E/2 thin ice & production (Nihashi et al. 2024) | See paper citation |
| RACMO2.4p1 surface fluxes | IMAU, Utrecht University (on request) |
| NSIDC Polar Pathfinder sea ice motion v4 | [NSIDC-0116](https://nsidc.org/data/NSIDC-0116) |
| BedMachine Antarctica v4 | [NSIDC IRVBM4](https://nsidc.org/data/IRVBM4) |

---

## Dependencies

```bash
conda env create -f environment.yml
conda activate swot-polynya
```

Key packages: `numpy`, `scipy`, `matplotlib`, `cartopy`, `netCDF4`, `h5py`, `pyproj`, `scikit-learn`, `cmocean`, `xarray`

---

## Usage

```bash
# 1. Classify a SWOT scene
python data_processing/swot_classification.py --input SWOT_L3_LR_SSH_Expert_*.nc

# 2. Co-locate with ICESat-2 and retrieve SIT
python data_processing/icesat2_retrieval.py --atl07 ATL07_*.h5 --swot-class output/swot_classes.nc

# 3. Drift-aware pooling over a season
python data_processing/drift_pooling.py --season 2023-09 2023-10

# 4. Reproduce main figures
python figures/plot_fig6_hssw_production.py
```

---

## Citation

```
Zhou et al. (2025). Resolving sea ice production and retention at kilometre scales
in Ross Sea coastal polynyas. Science Advances.
```

---

## License

MIT License — see `LICENSE` for details.

## Contact

Lu Zhou · l.zhou@uu.nl · Institute for Marine and Atmospheric Research, Utrecht University
