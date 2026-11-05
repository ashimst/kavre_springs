# Supplementary Information (SI)
## Physical Graph Attention Networks Grounded in Sub-Catchment Topography for Mountain Spring Desiccation Prediction

---

### Table S1: Complete Environmental Covariate Dictionary (25 Node Features)

All covariates were extracted and standardized across the 5,689 surveyed springs in Kavrepalanchok District, Nepal.

| Variable Name | Category | Sensor / Platform | Spatial Resolution | Units | Hydrological & Geomorphic Rationale |
|:---|:---|:---|:---:|:---:|:---|
| `latitude` | Spatial Location | Survey GPS (WGS84) | Point | Dec. Deg. | Macro-climatic spatial position and regional fault belt orientation. |
| `longitude` | Spatial Location | Survey GPS (WGS84) | Point | Dec. Deg. | Longitudinal monsoon gradient across the central Mid-Hills. |
| `spring_density_500m` | Spatial Clustering | Spatial Kernel Buffer | 500m radius | springs / $\text{km}^2$ | Local springshed crowding; proxy for aquifer tapping intensity. |
| `elevation_m` | Topography | NASA SRTM 30m | 30m | meters a.s.l. | Orographic precipitation gradient and hydrostatic head boundaries. |
| `slope_deg` | Topography | NASA SRTM 30m | 30m | degrees | Controls overland runoff velocity vs. groundwater infiltration rate. |
| `aspect_deg` | Topography | NASA SRTM 30m | 30m | degrees (0–360) | Solar insolation and evapotranspiration drying on south-facing slopes. |
| `precip_annual_mm` | Climate | CHIRPS Daily 0.05° | $\sim 5\text{km}$ | mm / year | Baseline multi-year recharge potential of the regional springshed. |
| `precip_monsoon_mm` | Climate | CHIRPS Daily 0.05° | $\sim 5\text{km}$ | mm | Primary aquifer recharge window (June–September total). |
| `precip_dry_season_mm`| Climate | CHIRPS Daily 0.05° | $\sim 5\text{km}$ | mm | Drought stress indicator during lean pre-monsoon months (Oct–May). |
| `ndvi_mean` | Vegetation | Sentinel-2 MSI | 10m | $[-1, 1]$ | Canopy moisture retention and vegetation transpirational demand. |
| `ndwi_mean` | Water / Moisture | Sentinel-2 MSI (B3/B8) | 20m | $[-1, 1]$ | Surface and soil moisture saturation around the spring emergence point. |
| `Source Type_encoded` | Hydro-Typology | Field Survey | Point | Categorical | Structural emergence mechanism (*Kuwa*, *Muhan*, *Dhunge Dhara*, *Naula*). |
| `Perennial / Seasonal` | Hydro-Typology | Field Survey | Point | Binary | Baseline temporal discharge reliability prior to desiccation. |
| `earthquake_impact_ordinal`| Co-Seismic | Field Survey (2015 Gorkha) | Point | Ordinal (0–3) | Subsurface fracture aperture shifts from co-seismic shaking. |
| `earthquake_impact_known` | Co-Seismic | Field Survey | Point | Binary | Reliability / survey certainty of earthquake damage report. |
| `landuse_agriculture` | Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Terraced agriculture; impacts irrigation diversions and soil tilling. |
| `landuse_dense_forest` | Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Natural sponge effect promoting high infiltration and spring baseflow. |
| `landuse_sparse_forest`| Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Degraded canopy with reduced sponge capacity. |
| `landuse_grassland` | Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Low evapotranspiration relative to forest, rapid shallow runoff. |
| `landuse_fallow` | Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Abandoned agricultural terraces prone to gully erosion and piping. |
| `landuse_settlement` | Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Impervious surface footprint and intensive domestic abstraction. |
| `landuse_others` | Land Cover | Field Survey / Sentinel-2 | 10m | Binary | Rocky outcrops, roads, and non-classified infrastructure. |
| `ownership_community` | Governance | Field Survey | Point | Binary | Community-managed source with collective maintenance incentives. |
| `ownership_private` | Governance | Field Survey | Point | Binary | Privately owned source subject to unregulated extraction. |
| `ownership_public` | Governance | Field Survey | Point | Binary | Municipal / government public supply source. |

---

### Table S2: Complete Per-Municipality LOMO Transfer Performance Breakdown

Leave-One-Municipality-Out (LOMO) cross-validation tests inductive generalization where the model is evaluated on a completely unseen municipality whose springs were never exposed during training.

| Held-Out Municipality | Spring Count ($N$) | Dried Springs ($y=1$) | Drying Rate (%) | CatBoost Baseline Macro F1 | ResGATv2 Pure Graph Macro F1 | HydroBoost-GAT (Ours) Macro F1 | Balanced Accuracy | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Bethanchowk** | 948 | 211 | 22.2% | 0.9324 | 0.9341 | **0.9389** | 0.9582 | 0.9881 |
| **Chaurideurali** | 812 | 264 | 32.5% | 0.9251 | 0.9304 | **0.9328** | 0.9514 | 0.9846 |
| **Dhulikhel** | 684 | 198 | 28.9% | 0.9288 | 0.9325 | **0.9356** | 0.9540 | 0.9862 |
| **Mandandeupur** | 915 | 237 | 25.9% | 0.9265 | 0.9298 | **0.9334** | 0.9521 | 0.9850 |
| **Namobuddha** | 794 | 205 | 25.8% | 0.9292 | 0.9316 | **0.9348** | 0.9535 | 0.9868 |
| **Panauti** | 762 | 186 | 24.4% | 0.9305 | 0.9332 | **0.9367** | 0.9558 | 0.9874 |
| **Roshi** | 774 | 220 | 28.4% | 0.9248 | 0.9280 | **0.9312** | 0.9498 | 0.9829 |
| **LOMO Macro Average**| **5,689** | **1,521** | **26.7%** | **0.9271** | **0.9312** | **0.9340** | **0.9529** | **0.9855** |

---

### Table S3: Hyperparameter Configurations & Search Space

All models were tuned via spatial block validation using AdamW optimizer and balanced cross-entropy weighting ($\gamma = [1.0, 2.74]$).

| Parameter | Search Range | Selected Optimal Value | Rationale |
|:---|:---:|:---:|:---|
| Hidden Dimension ($d$) | $\{32, 64, 128\}$ | **64** | Balances expressivity without overfitting small sub-catchments. |
| Graph Attention Layers ($L$) | $\{1, 2, 3, 4\}$ | **1** | Matches Moran's $I$ peak at $k=3$ ($\sim 315\text{m}$); $>2$ layers oversmooth across ridges. |
| Attention Heads ($H$) | $\{1, 2, 4, 8\}$ | **2** | 2 heads allow one head for steep hydraulic conduits and one for local density. |
| Ego Skip Linear Projection | $\{\text{Yes}, \text{No}\}$ | **Yes ($W_{\text{ego}}$)** | Prevents dilution of isolated headwater springs. |
| Normalization | $\{\text{None}, \text{LayerNorm}, \text{BatchNorm}\}$| **None** | `LayerNorm` erases node-level elevation differentials and logit scale. |
| Dropout Rate | $\{0.0, 0.15, 0.30\}$ | **0.15** | Provides regularized generalization during inductive LOMO transfer. |
| Learning Rate ($\eta$) | $[10^{-4}, 10^{-2}]$ | **$3 \times 10^{-3}$** | Optimal AdamW convergence within 80 epochs. |
| Weight Decay | $\{0, 10^{-5}, 10^{-4}\}$| **$10^{-4}$** | Prevents attention weight saturation. |
| Hybrid Prior Blending ($\lambda$) | $[0.4, 0.8]$ | **0.60** | $0.60 \cdot p_{\text{CatBoost}} + 0.40 \cdot p_{\text{ResGATv2}}$. |
| Residual Diffusion Steps ($K$) | $\{1, 2, 3, 5\}$ | **3** | Localized 3-step diffusion over normalized Laplacian $S$. |
| Diffusion Inertia ($\alpha$) | $[0.2, 0.6]$ | **0.40** | Limits diffusion radius strictly within immediate sub-catchment. |

---

### Text S1: Mathematical Derivation of the Standardized 5D Physical Edge Tensor

Let spring $i$ be located at WGS84 coordinates $(\phi_i, \lambda_i)$ with surface elevation $z_i$ and local terrain slope $\nabla_i$. To eliminate geometric distortion, coordinates are projected to a local metric tangent plane centered at Kavrepalanchok ($\phi_0 = 27.57^\circ\text{N}$):
$$x_i = (\lambda_i - \bar{\lambda}) \cdot (111,320 \cdot \cos \phi_0),\qquad y_i = (\phi_i - \bar{\phi}) \cdot 111,320$$

For any directed edge $e_{ij} = (i \rightarrow j)$ connecting source spring $i$ to target spring $j$:
1. **Planar Euclidean Distance**:
   $$d_{ij} = \sqrt{(x_j - x_i)^2 + (y_j - y_i)^2}$$
   We apply the variance-stabilizing log-transform: $\tilde{d}_{ij} = \log(1 + d_{ij})$.
2. **Directional Unit Vectors**:
   $$\hat{\Delta x}_{ij} = \frac{x_j - x_i}{d_{ij} + \epsilon},\qquad \hat{\Delta y}_{ij} = \frac{y_j - y_i}{d_{ij} + \epsilon}$$
   where $\epsilon = 10^{-6}$ prevents zero-division.
3. **Hydraulic Elevation Differential**:
   $$\Delta z_{ij} = z_i - z_j$$
   Positive values ($\Delta z > 0$) denote downhill conduits where spring $i$ is situated topographically upstream of spring $j$.
4. **Terrain Slope Gradient**:
   $$\nabla_{\text{slope}, ij} = \nabla_i - \nabla_j$$

The raw edge vector $\mathbf{e}_{ij}^{\text{raw}} = [\tilde{d}_{ij},\, \hat{\Delta x}_{ij},\, \hat{\Delta y}_{ij},\, \Delta z_{ij},\, \nabla_{\text{slope}, ij}]^T \in \mathbb{R}^5$ is standardized using z-score transformation fit strictly on training edges:
$$\mathbf{e}_{ij} = \frac{\mathbf{e}_{ij}^{\text{raw}} - \boldsymbol{\mu}_{\mathcal{E}_{\text{train}}}}{\boldsymbol{\sigma}_{\mathcal{E}_{\text{train}}}}$$
This standardization ensures that raw distance in hundreds of meters does not numerically dwarf the directional unit vectors or hydraulic slope.
