# Kavre Springs — GIS Spatial Data Mining Web Explorer

Interactive GIS explorer for 5,689 Himalayan springs across Kavrepalanchok District, Nepal. Built with **React 19**, **Vite**, and **Leaflet** using hardware-accelerated Canvas rendering.

Ready for zero-config global edge deployment on **Cloudflare Pages**.

---

## Key Features

1. **Native GIS Boundaries**:
   - Official Kavrepalanchok District outer boundary (`kavre_district.geojson`).
   - 13 Municipalities / Palika boundaries (`kavre_palikas.geojson`) with desiccation choropleth styling (Panauti, Bethanchowk, Namobuddha, Roshi, Dhulikhel, Temal, Panchkhal, etc.).
2. **High-Performance Canvas Spring Rendering**:
   - Renders all 5,689 GPS spring coordinates simultaneously at 60 FPS using Leaflet's `L.canvas()` renderer.
   - Color-coded: Muted Brick Red (`#c92a2a`) for Dried springs; Pine Forest Green (`#2b8a3e`) for Active springs.
   - Interactive popups showing elevation, municipality, source type, coordinates, and desiccation status.
3. **No API Key Tile Providers**:
   - Carto Positron (clean academic light style)
   - OpenStreetMap (reference roads and topo)
   - Esri World Imagery (high-resolution satellite terrain)
   - Esri World Topographic
4. **Data Mining Findings Drawer**:
   - **Hydro-Governance vs Hydrology**: Ward boundary variance ($ICC = 0.358$) vs catchment drainage basins ($ICC = 0.141$), demonstrating a 2.54× governance dominance over physical topography.
   - **Palika Desiccation Benchmarks**: Interactive ranking and click-to-zoom for surveyed municipalities.
   - **8 Landscape Typologies**: Micro-catchment clusters (T1–T8) with elevations, desiccation rates, and anthropogenic risk factors.
   - **Earthquake Desiccation Spike**: Interactive timeline highlighting the April 2015 Mw 7.8 Gorkha earthquake rupture (538 springs dried in 2015 alone).
   - **Apriori Association Rules**: Key feature co-occurrences (e.g. Road Distance < 50m + Public Ownership $\rightarrow$ Dried with Lift 2.54×).

---

## Deploying to Cloudflare Pages

### Option A: Via GitHub Integration (Continuous Deployment)
1. In the **Cloudflare Dashboard**, navigate to **Workers & Pages** $\rightarrow$ **Create Application** $\rightarrow$ **Pages** $\rightarrow$ **Connect to Git**.
2. Select your repository `kavre_springs`.
3. In the build settings:
   - **Framework preset**: `Vite`
   - **Root directory**: `frontend`
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
4. Click **Save and Deploy**. Cloudflare Pages will build the site and provide a free global edge URL (e.g., `https://kavre-springs.pages.dev`).

### Option B: Direct CLI Deployment (Wrangler)
```bash
cd frontend
npm run build
npx wrangler pages deploy dist --project-name kavre-springs
```

---

## Local Development

```bash
cd frontend
npm install
npm run dev
```

To build for production:
```bash
npm run build
npm run preview
```
