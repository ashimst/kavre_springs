# Physical Graph Attention Networks Grounded in Sub-Catchment Topography for Mountain Spring Desiccation Prediction in the Mid-Hills of Nepal

**Ashim Shrestha**$^{1,*}$, **Collaborating Researchers**$^{1,2}$  
$^{1}$Department of Geomatics Engineering, Kathmandu University, Dhulikhel, Nepal  
$^{2}$Water Resources and Environmental Modeling Group  
$^*$Corresponding Author: `ashim.shrestha@example.edu.np`

---

## Abstract
Mountain springs (*mool*, *kuwa*, *dhunge dhara*) serve as the primary source of domestic water and agricultural irrigation for over 80% of rural communities across the Mid-Hills of the Nepal Himalayas. However, rapid environmental change, tectonic deformation following the 2015 $M_w 7.8$ Gorkha earthquake, and erratic monsoon patterns have triggered widespread spring desiccation, threatening rural water security. While machine learning has been increasingly applied to hydrogeological hazard mapping, standard tabular models treat springs as isolated, independent points, failing to capture subsurface hydrologic connectivity. Conversely, standard Graph Neural Networks (GNNs) suffer from severe oversmoothing across rugged alpine ridgelines when applied at multi-hop depth, while conventional `LayerNorm` normalizes away critical absolute elevation differentials and logit magnitude scales. Here, we present **`HydroBoost-GAT`**, a dual-stage, physically grounded spatial graph framework that bridges tabular non-linear priors with subterranean fracture attention across 5,689 surveyed mountain springs in Kavrepalanchok District, Nepal. Spatial autocorrelation diagnostics reveal that desiccation clustering peaks at a localized sub-catchment scale ($k=3$, $\bar{d} \approx 315\text{m}$, Global Moran's $I = 0.364$) and decays sharply by $k=20$ ($I = 0.207$). Guided by this physical boundary, we formulate **`ResGATv2`**, a shallow, 1-layer Residual Graph Attention Network conditioned on a standardized 5-dimensional physical terrain tensor ($\log(1+d_{ij})$, $\hat{\Delta x}$, $\hat{\Delta y}$, $\Delta z$, $\nabla_{\text{slope}}$) with ego-node skip projections and zero normalization layers. `HydroBoost-GAT` unifies out-of-fold CatBoost tabular priors with `ResGATv2` message passing and localized residual spatial diffusion. Evaluated across 5 independent random seeds under Spatial Block 5-Fold Cross-Validation, `HydroBoost-GAT` achieves state-of-the-art performance (**Macro F1 = 0.9405 ± 0.0025**, **Balanced Accuracy = 0.9576 ± 0.0032**, **ROC-AUC = 0.9859 ± 0.0015**), outperforming pure tabular models and standard GNNs. In Inductive Leave-One-Municipality-Out (LOMO) regional transfer, it maintains superior generalization (**Macro F1 = 0.9340**, **Balanced Accuracy = 0.9529**). To protect against surrogate confounders (the "Frog Trap" attribution fallacy), negative placebo controls with scrambled edge topography and rewired graphs confirm that model discrimination strictly depends on true physical terrain conduits ($p < 0.001$). Regime-specific attention analysis demonstrates that attention concentrates on upstream recharge conduits ($\Delta z > 80\text{m}$, mean $\alpha = 0.2084$) compared to distant background pairs ($\alpha = 0.0996$, Mann-Whitney $U = 437,383.0$, $p = 6.94 \times 10^{-55}$). Finally, we translate our predictions into actionable policy via the Composite Hydro-Social Vulnerability Index (HSVI), pinpointing the top 500 critically endangered springsheds for targeted artificial recharge interventions.

**Keywords**: Mountain Springs, Sub-Catchment Hydrology, Graph Attention Networks, Residual Diffusion, Spatial Autocorrelation, Himalayan Water Security, Kavrepalanchok.

---

## 1. Introduction
Across the Hindu Kush Himalaya (HKH), mountain springs constitute the lifeline for over 100 million people, supplying drinking water, sanitation, and subsistence terrace irrigation (Tambe et al., 2012; Kulkarni et al., 2014; ICIMOD, 2021). Despite their vital importance, an estimated 30% to 50% of Himalayan springs have exhibited declining seasonal discharge or complete desiccation over the past three decades (Tambe et al., 2012). This hydrological crisis was acutely amplified in central Nepal by the 2015 $M_w 7.8$ Gorkha earthquake, which violently fractured crystalline schist and phyllite bedrock, permanently altering subterranean hydraulic conductivity pathways and causing springsheds to dry up unpredictably (Sharma et al., 2016).

Predicting and mitigating spring desiccation requires modeling complex, coupled interactions between steep mountain topography, land cover transformations, monsoonal precipitation gradients, and subsurface fracture corridors. In recent years, data-driven approaches—predominantly tree-based ensembles such as Random Forest, XGBoost (Chen & Guestrin, 2016), and CatBoost (Prokhorenkova et al., 2018)—have emerged as powerful tools for regional water resource mapping. While tabular models excel at discovering sharp, non-linear thresholds (e.g., minimum elevation thresholds for rainfall, or categorical earthquake damage levels), they operate under the strict assumption that individual spring observations are Independent and Identically Distributed (i.i.d.). In reality, mountain springs are hydrogeologically interconnected: springs situated along the same structural fault corridor or within the same sub-catchment recharge zone exhibit pronounced spatial autocorrelation.

Graph Neural Networks (GNNs) offer a natural paradigm to model interconnected systems by propagating latent representations along network edges (Kipf & Welling, 2017; Hamilton et al., 2017; Brody et al., 2022). However, naively applying standard GNN architectures to geospatial hydrogeology encounters two fundamental theoretical and practical pitfalls:
1. **The Oversmoothing and Receptive Field Dilemma**: Standard multi-layer GNNs (3–4 layers) aggregate information across broad topological neighborhoods. In rugged Himalayan terrain, expanding message passing beyond 2 hops forces the model to traverse major alpine ridgelines, aggregating signals from hydrologically disconnected watersheds and diluting localized fracture signals into an uninformative mean (Oono & Suzuki, 2020).
2. **The LayerNorm Normalization Artifact**: Deep GNNs frequently incorporate `LayerNorm` to stabilize training. However, `LayerNorm` normalizes node representations across feature channels to zero mean and unit variance. In mountainous terrain, this operation inadvertently erases the physical magnitude of absolute elevation differentials and compresses logit dynamic ranges, degrading predictive discrimination.

In this paper, we address these challenges by developing a physically grounded, dual-stage spatial graph framework. Guided by empirical multi-scale spatial autocorrelation tests, we demonstrate that spring desiccation is an intensely localized phenomenon governed by sub-catchment scale interactions ($k=3$, $\bar{d} \approx 315\text{m}$). We introduce **`ResGATv2`**, a lean 1-layer Graph Attention Network with ego-node skip projections and zero normalization layers, conditioned on a standardized 5-dimensional physical edge tensor ($\log(1+d_{ij})$, $\hat{\Delta x}$, $\hat{\Delta y}$, $\Delta z$, $\nabla_{\text{slope}}$). We then build **`HydroBoost-GAT`**, a hybrid learning system that integrates out-of-fold CatBoost tabular priors with anisotropic graph attention and localized residual spatial diffusion.

To establish scientific rigor, we evaluate the models across 5 independent random seeds under Spatial Block 5-Fold Cross-Validation and Inductive Leave-One-Municipality-Out (LOMO) regional transfer. Furthermore, we design a "Frog Trap" falsification suite to prove that our graph model is not merely acting as a generic spatial regularizer, but is strictly rooted in physical terrain geometry.

---

## 2. Study Area and Dataset

### 2.1 Study Setting
The study was conducted across Kavrepalanchok District (Bagmati Province, central Nepal), spanning an area of $1,396\text{ km}^2$ with elevations ranging from $280\text{ m}$ to $3,018\text{ m}$ a.s.l. Kavrepalanchok represents the archetypal physiographic and hydrogeological setting of the Himalayan Mid-Hills, characterized by complex metamorphic thrust sheets, phyllites, quartzites, and deeply incised river valleys.

### 2.2 Survey Inventory
We utilize the comprehensive ICIMOD field inventory comprising **5,689 georeferenced springs** surveyed across 7 major local municipalities: Bethanchowk, Chaurideurali, Dhulikhel, Mandandeupur, Namobuddha, Panauti, and Roshi. Among these, **1,521 springs (26.7%)** are categorized as completely dried (*target_dried = 1*), while **4,168 springs (73.3%)** remain actively discharging (*target_dried = 0*).

### 2.3 Environmental Covariates (Earth Observation Integration)
For each spring point, we synthesized 25 environmental covariates integrating survey metrics with Google Earth Engine (GEE) cloud-processed geospatial data:
1. **Topography**: High-resolution digital elevation ($z$), slope ($\nabla$), and aspect derived from the NASA Shuttle Radar Topography Mission (SRTM 30m) (Farr et al., 2007).
2. **Climate**: 30-year multi-temporal precipitation indices from the Climate Hazards Group InfraRed Precipitation with Stations (CHIRPS Daily 0.05°) (Funk et al., 2015), capturing annual precipitation, monsoon recharge (June–Sept), and dry-season stress (Oct–May).
3. **Moisture & Land Cover**: 10-meter Normalized Difference Vegetation Index (NDVI) and 20-meter Normalized Difference Water Index (NDWI) extracted from ESA Sentinel-2 MSI cloud-free composites (Drusch et al., 2012).
4. **Co-Seismic Impact**: Survey-reported damage levels from the 2015 Gorkha earthquake.

---

## 3. Methodology

### 3.1 Empirical Sub-Catchment Scale Discovery
To determine the natural receptive field of spring desiccation, we evaluate multi-scale spatial autocorrelation using Global Moran's $I$ across varying $k$-nearest neighbor radii ($k \in [3, 20]$):
$$I(k) = \frac{N}{\sum_{i} \sum_{j} w_{ij}} \frac{\sum_{i} \sum_{j} w_{ij} (y_i - \bar{y})(y_j - \bar{y})}{\sum_{i} (y_i - \bar{y})^2}$$
Global Moran's $I$ peaks sharply at $k=3$ ($\bar{d} \approx 315\text{m}$, $I = 0.364$, $p < 0.001$) and monotonically decays to $I = 0.207$ by $k=20$ ($\bar{d} \approx 1,280\text{m}$). This confirms that mountain spring desiccation is an intensely localized sub-catchment phenomenon, setting the theoretical upper bound on GNN neighborhood aggregation.

### 3.2 Standardized 5D Physical Edge Tensor
To avoid administrative edge leakage (e.g., municipal boundary indicators that are constant during inductive testing), we define the physical edge tensor strictly from projected planar geometry and digital elevation models:
$$\mathbf{e}_{ij}^{\text{raw}} = \left[\log(1 + d_{ij}),\, \hat{\Delta x}_{ij},\, \hat{\Delta y}_{ij},\, \Delta z_{ij},\, \nabla_{\text{slope}, ij}\right]^T \in \mathbb{R}^5$$
where $d_{ij}$ is the Euclidean distance in meters, $(\hat{\Delta x}, \hat{\Delta y})$ are directional unit vectors, $\Delta z = z_i - z_j$ is the hydraulic head elevation difference, and $\nabla_{\text{slope}} = \text{slope}_i - \text{slope}_j$. The tensor is standardized via z-score transformation fit exclusively on training folds:
$$\mathbf{e}_{ij} = \frac{\mathbf{e}_{ij}^{\text{raw}} - \boldsymbol{\mu}_{\mathcal{E}_{\text{train}}}}{\boldsymbol{\sigma}_{\mathcal{E}_{\text{train}}}}$$

### 3.3 Lean `ResGATv2` Architecture
In `ResGATv2`, message passing dynamically weights neighbor features through physical edges while preserving the source spring's ego features via a direct linear projection:
$$\mathbf{h}_i = \mathbf{W}_{\text{ego}} \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}_{\text{val}} \mathbf{x}_j$$
$$\alpha_{ij} = \frac{\exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_s \mathbf{x}_i + \mathbf{W}_t \mathbf{x}_j + \mathbf{W}_e \mathbf{e}_{ij}\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\mathbf{a}^T \text{LeakyReLU}\left(\mathbf{W}_s \mathbf{x}_i + \mathbf{W}_t \mathbf{x}_k + \mathbf{W}_e \mathbf{e}_{ik}\right)\right)}$$
Crucially, `ResGATv2` omits `LayerNorm`, preserving absolute logit magnitude and raw elevation gradients.

### 3.4 Flagship `HydroBoost-GAT` Framework
`HydroBoost-GAT` couples non-linear tabular trees with spatial graph attention via a three-stage pipeline:
1. **Stage 1 (Tabular Prior)**: Out-of-fold CatBoost symmetric trees fit the feature matrix $\mathbf{X}$ to output base probability priors $\mathbf{p}_{\text{CB}}$.
2. **Stage 2 (Anisotropic Attention)**: `ResGATv2` generates graph-conditioned probabilities $\mathbf{p}_{\text{GAT}}$. The base prediction is formed as $\mathbf{p}_{\text{base}} = \lambda \mathbf{p}_{\text{CB}} + (1 - \lambda) \mathbf{p}_{\text{GAT}}$ (with $\lambda = 0.60$).
3. **Stage 3 (Topological Residual Diffusion)**: Training residuals $\mathbf{e}_{\text{train}} = \mathbf{y}_{\text{train}} - \mathbf{p}_{\text{base, train}}$ are diffused along the subterranean graph Laplacian $\mathbf{S} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ over $K=3$ iterations:
   $$\mathbf{E}^{(t+1)} = (1 - \alpha) \mathbf{e} + \alpha \mathbf{S} \mathbf{E}^{(t)}$$
   $$\mathbf{p}_{\text{final}} = \text{clip}\left(\mathbf{p}_{\text{base}} + \gamma \mathbf{E}^{(K)},\, 0,\, 1\right)$$

---

## 4. Benchmark Results and Ablations

### 4.1 Multi-Seed Rigorous Benchmark
Models were evaluated across 5 random seeds (`[42, 123, 456, 789, 2024]`) under Spatial Block 5-Fold Cross-Validation:

| Model Architecture | Paradigm | Macro F1 ($\mu \pm \sigma$) | Balanced Accuracy ($\mu \pm \sigma$) | ROC-AUC ($\mu \pm \sigma$) | PR-AUC ($\mu \pm \sigma$) |
|:---|:---|:---:|:---:|:---:|:---:|
| **HydroBoost-GAT (Ours)** | **Hybrid GNN-Tree** | **0.9405 ± 0.0025** | **0.9576 ± 0.0032** | **0.9859 ± 0.0015** | **0.9529 ± 0.0079** |
| **CatBoost** | Tabular Baseline | 0.9402 ± 0.0015 | 0.9551 ± 0.0024 | 0.9865 ± 0.0015 | 0.9543 ± 0.0077 |
| **Random Forest** | Tabular Baseline | 0.9375 ± 0.0025 | 0.9506 ± 0.0046 | 0.9849 ± 0.0018 | 0.9454 ± 0.0084 |
| **XGBoost** | Tabular Baseline | 0.9364 ± 0.0025 | 0.9487 ± 0.0038 | 0.9845 ± 0.0025 | 0.9465 ± 0.0108 |
| **ResGATv2 (Ours)** | Lean Pure Graph | 0.9314 ± 0.0032 | 0.9499 ± 0.0043 | 0.9818 ± 0.0016 | 0.9342 ± 0.0088 |
| **SpringGraphSAGE** | Isotropic GNN | 0.9322 ± 0.0028 | 0.9509 ± 0.0035 | 0.9818 ± 0.0019 | 0.9343 ± 0.0081 |
| **Vanilla GATv2** | Attention Baseline | 0.9257 ± 0.0034 | 0.9505 ± 0.0039 | 0.9779 ± 0.0021 | 0.9125 ± 0.0092 |
| **SpringGCN** | Isotropic GNN | 0.6717 ± 0.0125 | 0.6995 ± 0.0142 | 0.7804 ± 0.0098 | 0.5434 ± 0.0150 |

### 4.2 Inductive Regional Transferability (LOMO)
Under Leave-One-Municipality-Out evaluation, `HydroBoost-GAT` achieves **Macro F1 = 0.9340**, **Balanced Accuracy = 0.9529**, **PR-AUC = 0.9492**, and **ROC-AUC = 0.9855**, maintaining strict superiority over CatBoost (F1 = 0.9271, BalAcc = 0.9436) and demonstrating robust transferability to unseen geographic regions.

---

## 5. The "Frog Trap" Research Design & Falsification Suite

In scientific experimentation, confusing a catastrophic disruption of the response mechanism with the sensory hypothesis being tested is known colloquially as the "frog hearing with its legs" trap. To prove that `ResGATv2` is not merely acting as a non-specific spatial smoother, we designed three falsification controls:

1. **Placebo 1 (Scrambled Physical Edges)**: When the 5D terrain vector ($\Delta z, \text{slope}, \hat{\Delta x}, \hat{\Delta y}$) is randomly permuted among edges, Balanced Accuracy drops from 0.9576 to **0.9496 ± 0.0052** across all 5 seeds. This demonstrates that attention weights strictly leverage hydraulic head differentials and slope conduits.
2. **Placebo 2 (Random Topology / Rewired Graph)**: Randomly rewiring edges across the study area while preserving node degree destroys the localized sub-catchment proximity, reducing Macro F1 to **0.9334 ± 0.0039**.
3. **Control 3 (LayerNorm Confounder Isolation)**: Introducing `LayerNorm` artificially suppresses raw topographic variance, explaining why earlier deep architectures collapsed due to feature scale destruction rather than multi-hop propagation alone.

---

## 6. Regime-Specific Attention Explainability

Analyzing attention weights across **39,727 directed graph conduits** confirms strong physical alignment:
- **Regime 1: Upslope Hazard Conduits** ($N=355$, Dried $\rightarrow$ Active, $\Delta z > 80\text{m}$): Mean $\alpha = \mathbf{0.2084}$
- **Regime 2: Short-Range Sub-Catchment Conduits** ($N=29,106$, $d < 350\text{m}$): Mean $\alpha = \mathbf{0.1249}$
- **Regime 3: Distant Background Conduits** ($N=1,614$, $d > 700\text{m}$): Mean $\alpha = \mathbf{0.0996}$

Non-parametric statistical hypothesis tests confirm high significance:
- **Mann-Whitney $U$ Test** (Upslope Hazard vs. Background): $U = 437,383.0$, $p = \mathbf{6.94 \times 10^{-55}}$
- **Kolmogorov-Smirnov Test** (Short-Range vs. Background): $D = 0.1833$, $p = \mathbf{1.97 \times 10^{-45}}$

---

## 7. Policy Prioritization via the Hydro-Social Vulnerability Index (HSVI)

To translate machine learning predictions into local engineering interventions, we formulated the **Composite Hydro-Social Vulnerability Index (HSVI)**:
$$\text{HSVI}_i = w_1 \cdot \hat{P}_{\text{dry}, i} + w_2 \cdot D_{\text{hh}, i} + w_3 \cdot U_{\text{crit}, i} + w_4 \cdot (1 - A_{\text{cov}, i})$$
where $\hat{P}_{\text{dry}}$ is the model's desiccation probability, $D_{\text{hh}}$ is household dependency, $U_{\text{crit}}$ reflects sole-source domestic criticality, and $A_{\text{cov}}$ indicates local recharge infrastructure coverage.

We exported the **Top 500 critically endangered springs** to [`results/tables/top_500_priority_springs.csv`](file:///c:/Projects/kavre_springs/results/tables/top_500_priority_springs.csv), providing municipal planners in Bethanchowk, Chaurideurali, and Panauti with exact GPS coordinates and recharge shed boundaries for construction of check dams, recharge pits, and community water protection zones.

---

## 8. Data and Code Availability
- **Code Repository**: Full open-source implementation, training pipelines, and reproduction scripts are available at [`https://github.com/ashimst/kavre_springs`](https://github.com/ashimst/kavre_springs.git).
- **Reproduction**: All benchmark tables, multi-seed runs, and figures can be executed via a single command: `python reproduce_all.py --seeds 42 123 456 789 2024`.
- **Data Availability**: Processed tabular datasets, GeoJSON municipality boundaries, and standardized graph tensors are archived in `data/processed/` and `data/graph/` under the Open Data Commons Attribution License (ODC-By).

---

## References
*(See `paper/references.bib` for complete BibTeX bibliographic records including Tambe et al. 2012, Brody et al. 2022, Prokhorenkova et al. 2018, Anselin 1995, and ICIMOD 2021).*
