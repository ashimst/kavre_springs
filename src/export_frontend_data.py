import json
from pathlib import Path
import pandas as pd
from shapely.geometry import shape, mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "processed" / "master_dataset.parquet"
DISTRICT_GEOJSON = DATA_DIR / "processed" / "boundaries" / "kavre_district.geojson"
PALIKAS_GEOJSON = DATA_DIR / "processed" / "boundaries" / "kavre_palikas.geojson"
OUTPUT_JSON = PROJECT_ROOT / "frontend" / "src" / "data.json"

df = pd.read_parquet(PARQUET_PATH)

# 1. District GeoJSON
with open(DISTRICT_GEOJSON, encoding='utf-8') as f:
    dist_gj = json.load(f)
for feat in dist_gj['features']:
    s = shape(feat['geometry']).simplify(0.0005, preserve_topology=True)
    feat['geometry'] = mapping(s)

# 2. Municipality Stats & Palikas GeoJSON
muni_stats = df.groupby('municipality').agg(
    total=('target_dried', 'count'),
    dried=('target_dried', 'sum'),
    rate=('target_dried', 'mean'),
    elev=('elevation_m', 'mean'),
    road=('dist_to_road_m', 'mean')
).reset_index()

with open(PALIKAS_GEOJSON, encoding='utf-8') as f:
    palikas_gj = json.load(f)

for feat in palikas_gj['features']:
    name = feat['properties']['name']
    match = muni_stats[muni_stats['municipality'].str.lower().str.startswith(name[:4].lower())]
    if len(match) > 0:
        row = match.iloc[0]
        feat['properties']['surveyed'] = True
        feat['properties']['total_springs'] = int(row['total'])
        feat['properties']['dried_springs'] = int(row['dried'])
        feat['properties']['drying_rate'] = round(float(row['rate']) * 100, 1)
        feat['properties']['mean_elev'] = round(float(row['elev']), 0)
        feat['properties']['mean_road'] = round(float(row['road']), 0)
    else:
        feat['properties']['surveyed'] = False
        feat['properties']['total_springs'] = 0
        feat['properties']['dried_springs'] = 0
        feat['properties']['drying_rate'] = 0.0

    s = shape(feat['geometry']).simplify(0.0005, preserve_topology=True)
    feat['geometry'] = mapping(s)

# 3. Compact representation of all 5,689 springs
# [lat, lon, dried, elev, muni, type]
springs_list = []
for _, r in df.iterrows():
    springs_list.append([
        round(float(r['latitude']), 4),
        round(float(r['longitude']), 4),
        int(r['target_dried']),
        int(round(float(r['elevation_m']))),
        str(r['municipality']),
        str(r.get('source_type', 'spring'))
    ])

# 4. Detailed Chart Data Series
# A. Elevation Distribution
df['elev_bin'] = pd.cut(
    df['elevation_m'], 
    [0, 1000, 1300, 1600, 1900, 4000], 
    labels=['< 1,000m', '1,000–1,300m', '1,300–1,600m', '1,600–1,900m', '> 1,900m']
)
elev_grp = df.groupby('elev_bin', observed=False)['target_dried'].agg(
    total='count', dried='sum', rate=lambda x: round(float(x.mean()) * 100, 1)
).reset_index()
elev_dist = []
for _, r in elev_grp.iterrows():
    elev_dist.append({
        'range': str(r['elev_bin']),
        'total': int(r['total']),
        'dried': int(r['dried']),
        'active': int(r['total'] - r['dried']),
        'rate': float(r['rate'])
    })

# B. Road Distance Gradient
df['road_bin'] = pd.cut(
    df['dist_to_road_m'], 
    [-1, 50, 100, 250, 500, 20000], 
    labels=['< 50m', '50–100m', '100–250m', '250–500m', '> 500m']
)
road_grp = df.groupby('road_bin', observed=False)['target_dried'].agg(
    total='count', dried='sum', rate=lambda x: round(float(x.mean()) * 100, 1)
).reset_index()
road_dist = []
for _, r in road_grp.iterrows():
    road_dist.append({
        'distance': str(r['road_bin']),
        'total': int(r['total']),
        'dried': int(r['dried']),
        'active': int(r['total'] - r['dried']),
        'rate': float(r['rate'])
    })

# C. Land Use Resilience
lu_names = {
    'dense_forest': 'Dense Forest',
    'sparse_forest': 'Sparse Forest',
    'agriculture': 'Agricultural Terraces',
    'grassland': 'Grassland & Shrub',
    'settlement': 'Settlement & Urban Fringe',
    'fallow': 'Fallow Land',
    'others': 'Other Barren'
}
lu_grp = df.groupby('landuse')['target_dried'].agg(
    total='count', dried='sum', rate=lambda x: round(float(x.mean()) * 100, 1)
).reset_index()
lu_dist = []
for _, r in lu_grp.iterrows():
    lu_dist.append({
        'landuse': lu_names.get(r['landuse'], str(r['landuse']).title()),
        'total': int(r['total']),
        'dried': int(r['dried']),
        'rate': float(r['rate'])
    })
lu_dist.sort(key=lambda x: x['rate'])

# D. Aspect Solar Insolation
def aspect_to_quad(a):
    if (a >= 315) or (a < 45): return 'North (Shaded / Moist)'
    elif 45 <= a < 135: return 'East (Morning Sun)'
    elif 135 <= a < 225: return 'South (High Insolation / Deficit)'
    else: return 'West (Afternoon Thermal Deficit)'

df['aspect_quad'] = df['aspect_deg'].apply(aspect_to_quad)
asp_grp = df.groupby('aspect_quad')['target_dried'].agg(
    total='count', dried='sum', rate=lambda x: round(float(x.mean()) * 100, 1)
).reset_index()
aspect_dist = []
for _, r in asp_grp.iterrows():
    aspect_dist.append({
        'quadrant': str(r['aspect_quad']),
        'total': int(r['total']),
        'dried': int(r['dried']),
        'rate': float(r['rate'])
    })

# E. Recalled Community Desiccation Causes
recalled_causes = [
    {'cause': '2015 Gorkha Earthquake', 'count': 538, 'pct': 35.4, 'desc': 'Seismic fracture dilation & aquifer displacement'},
    {'cause': 'Road & Track Construction', 'count': 374, 'pct': 24.6, 'desc': 'Heavy bulldozer cuts shearing shallow fracture lines'},
    {'cause': 'Prolonged Drought & Rain Decline', 'count': 331, 'pct': 21.8, 'desc': 'Monsoon variability & winter precipitation deficits'},
    {'cause': 'Land Cover & Forest Degradation', 'count': 167, 'pct': 11.0, 'desc': 'Loss of canopy interception & infiltration capacity'},
    {'cause': 'Siltation & Unmanaged Upstream Diversion', 'count': 109, 'pct': 7.2, 'desc': 'Intake pipe tapping and sediment clogging'}
]

payload = {
    'totalSprings': int(len(df)),
    'totalDried': int(df['target_dried'].sum()),
    'dryingRate': round(float(df['target_dried'].mean()) * 100, 1),
    'district': dist_gj,
    'palikas': palikas_gj,
    'springs': springs_list,
    'charts': {
        'elevation': elev_dist,
        'road': road_dist,
        'landuse': lu_dist,
        'aspect': aspect_dist,
        'causes': recalled_causes
    },
    'icc': {
        'ward': 0.358,
        'catchment': 0.141,
        'ratio': '2.54×'
    },
    'typology': [
        {'id': 'T1', 'name': 'Urban Fringe Terraces', 'rate': 40.0, 'elev': '1,430m', 'hazard': 'High borehole and road density disruption', 'risk': 'High'},
        {'id': 'T2', 'name': 'Alluvial Valleys', 'rate': 29.5, 'elev': '1,030m', 'hazard': 'Agricultural irrigation drawdown', 'risk': 'Moderate'},
        {'id': 'T3', 'name': 'Mid-elevation Farmland', 'rate': 29.5, 'elev': '1,380m', 'hazard': 'Terraced cultivation water balance', 'risk': 'Moderate'},
        {'id': 'T4', 'name': 'Wetland Depressions', 'rate': 26.1, 'elev': '1,150m', 'hazard': 'Sediment accumulation in ponds', 'risk': 'Moderate'},
        {'id': 'T5', 'name': 'Steep Degraded South Slopes', 'rate': 36.9, 'elev': '1,490m', 'hazard': 'Severe climate water deficit & thin soils', 'risk': 'High'},
        {'id': 'T6', 'name': 'Shaded North Hillslopes', 'rate': 18.5, 'elev': '1,460m', 'hazard': 'Relatively moist and resilient', 'risk': 'Low'},
        {'id': 'T7', 'name': 'Forested Mid-Ridges', 'rate': 19.8, 'elev': '1,650m', 'hazard': 'Protected recharge canopy buffer', 'risk': 'Low'},
        {'id': 'T8', 'name': 'High-Altitude Sanctuaries', 'rate': 12.6, 'elev': '1,890m', 'hazard': 'Pristine mountain recharge zones', 'risk': 'Resilient'}
    ],
    'rules': [
        {'rule': 'Ridge Crest + High Water Deficit', 'consequent': 'Dried', 'confidence': '52.2%', 'lift': '1.95×', 'desc': 'High topographic position combined with south-facing moisture deficit doubles spring desiccation hazard.'},
        {'rule': 'Low Upstream Area + High TPI', 'consequent': 'Dried', 'confidence': '39.3%', 'lift': '1.47×', 'desc': 'Springs near ridgelines with limited upslope collection area desiccate rapidly during droughts.'},
        {'rule': 'Distance to Road < 50m + Public Ownership', 'consequent': 'Dried', 'confidence': '67.8%', 'lift': '2.54×', 'desc': 'Unpaved rural road cuts directly rupture shallow aquifer fracture lines and induce siltation.'},
        {'rule': 'Dense Forest + High TWI + Low Deficit', 'consequent': 'Active', 'confidence': '99.2%', 'lift': '1.35×', 'desc': 'Intact native canopy and natural concave depressions maintain near-perfect hydrological resilience.'}
    ],
    'timeline': [
        {'year': '2005', 'count': 18, 'isQuake': False},
        {'year': '2008', 'count': 116, 'isQuake': False},
        {'year': '2011', 'count': 50, 'isQuake': False},
        {'year': '2013', 'count': 139, 'isQuake': False},
        {'year': '2015', 'count': 538, 'isQuake': True},
        {'year': '2017', 'count': 48, 'isQuake': False},
        {'year': '2019', 'count': 39, 'isQuake': False},
        {'year': '2021', 'count': 43, 'isQuake': False},
        {'year': '2023', 'count': 29, 'isQuake': False}
    ],
    'credits': {
        'dataProvider': 'International Centre for Integrated Mountain Development (ICIMOD)',
        'dataDetails': 'Comprehensive field inventory of 5,689 springs conducted in collaboration with local palika governments and the Ministry of Energy, Water Resources and Irrigation, Nepal under Open Data principles.',
        'author': 'Ashim Shrestha',
        'organization': 'Noesis Frameworks',
        'orgUrl': 'https://noesisframeworks.com',
        'githubUrl': 'https://github.com/ashimst/kavre_springs',
        'userGithub': 'https://github.com/ashimst'
    }
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(payload, f)

print(f"Exported data.json to {OUTPUT_JSON} (size: {OUTPUT_JSON.stat().st_size / 1024:.1f} KB)")
