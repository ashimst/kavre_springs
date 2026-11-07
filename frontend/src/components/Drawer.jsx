import React, { useState } from 'react';
import { 
  Building2, 
  Map, 
  Layers, 
  Activity, 
  BarChart2, 
  BookOpen, 
  Globe, 
  ExternalLink,
  ShieldCheck,
  TrendingUp,
  Mountain,
  Compass,
  Trees,
  Truck
} from 'lucide-react';

const GithubIcon = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

export default function Drawer({
  isOpen,
  onClose,
  data,
  selectedPalika,
  onSelectPalika
}) {
  const [activeTab, setActiveTab] = useState('analytics');

  if (!isOpen) return null;

  const palikas = [
    { name: 'Panauti', rate: 9.4, count: 447, dried: 42, elev: 1747, road: 142 },
    { name: 'Bethanchowk', rate: 19.2, count: 1029, dried: 198, elev: 1770, road: 185 },
    { name: 'Namobuddha', rate: 20.1, count: 452, dried: 91, elev: 1475, road: 120 },
    { name: 'Roshi', rate: 21.8, count: 1708, dried: 373, elev: 1489, road: 168 },
    { name: 'Dhulikhel', rate: 28.8, count: 368, dried: 106, elev: 1327, road: 84 },
    { name: 'Temal', rate: 39.8, count: 679, dried: 270, elev: 1332, road: 154 },
    { name: 'Panchkhal', rate: 39.9, count: 1006, dried: 401, elev: 1074, road: 110 },
  ];

  const charts = data?.charts || {};

  return (
    <aside className="insights-drawer glass-panel" aria-label="Scientific Insights Panel">
      {/* Drawer Header & Tabs */}
      <div className="drawer-header">
        <div className="drawer-nav">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >
            <BarChart2 size={14} />
            <span>Charts &amp; Analytics</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'governance' ? 'active' : ''}`}
            onClick={() => setActiveTab('governance')}
          >
            <Building2 size={14} />
            <span>Governance (2.5&times;)</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'palikas' ? 'active' : ''}`}
            onClick={() => setActiveTab('palikas')}
          >
            <Map size={14} />
            <span>Palikas</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'typology' ? 'active' : ''}`}
            onClick={() => setActiveTab('typology')}
          >
            <Layers size={14} />
            <span>Typology (T1–T8)</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'credits' ? 'active' : ''}`}
            onClick={() => setActiveTab('credits')}
          >
            <BookOpen size={14} />
            <span>ICIMOD &amp; Credits</span>
          </button>
        </div>
        <button 
          type="button" 
          onClick={onClose} 
          className="drawer-close-btn"
          title="Minimize drawer"
        >
          &times;
        </button>
      </div>

      {/* Drawer Body Content */}
      <div className="drawer-content">

        {/* TAB 1: CHARTS & ANALYTICS */}
        {activeTab === 'analytics' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>Spatial Hydro-Geological Analytics</h4>
              <p>Key empirical distributions mined across 5,689 surveyed springs in Kavrepalanchok.</p>
            </div>

            {/* CHART 1: Elevation Gradient */}
            <div className="chart-card">
              <div className="chart-card-header">
                <div className="chart-title-group">
                  <Mountain size={14} className="chart-icon" />
                  <h5>Elevation Gradient &amp; Desiccation Rate</h5>
                </div>
                <span className="chart-badge">Orography</span>
              </div>
              <p className="chart-sub">
                Higher mountain zones (&gt;1,600m) maintain higher resilience (17% dried) compared to low-elevation valley fringes (&gt;34% dried).
              </p>
              <div className="custom-bar-chart">
                {(charts.elevation || []).map((b) => (
                  <div key={b.range} className="chart-bar-row">
                    <span className="bar-label tabular-nums">{b.range}</span>
                    <div className="bar-track-stacked">
                      <div 
                        className="bar-seg-active" 
                        style={{ width: `${(b.active / 2300) * 100}%` }} 
                        title={`Active: ${b.active}`}
                      />
                      <div 
                        className="bar-seg-dried" 
                        style={{ width: `${(b.dried / 2300) * 100}%` }} 
                        title={`Dried: ${b.dried}`}
                      />
                    </div>
                    <span className={`bar-rate-pill ${b.rate > 30 ? 'high' : b.rate > 20 ? 'med' : 'low'} tabular-nums`}>
                      {b.rate}%
                    </span>
                  </div>
                ))}
              </div>
              <div className="chart-legend-inline">
                <span className="legend-item"><span className="legend-chip active-chip"></span> Active Springs</span>
                <span className="legend-item"><span className="legend-chip dried-chip"></span> Dried Springs</span>
              </div>
            </div>

            {/* CHART 2: Road Distance Hazard Curve */}
            <div className="chart-card">
              <div className="chart-card-header">
                <div className="chart-title-group">
                  <Truck size={14} className="chart-icon" />
                  <h5>Distance to Motorable Road Hazard Gradient</h5>
                </div>
                <span className="chart-badge warning">Infrastructure</span>
              </div>
              <p className="chart-sub">
                Desiccation hazards drop sharply from <strong>30.1%</strong> within 50m of unpaved rural roads to <strong>18.3%</strong> at distances &gt;250m.
              </p>
              <div className="custom-bar-chart">
                {(charts.road || []).map((r) => (
                  <div key={r.distance} className="chart-bar-row">
                    <span className="bar-label tabular-nums">{r.distance}</span>
                    <div className="bar-track-simple">
                      <div 
                        className={`bar-simple-fill ${r.rate > 28 ? 'high' : r.rate > 22 ? 'med' : 'low'}`}
                        style={{ width: `${r.rate * 2.8}%` }}
                      />
                    </div>
                    <span className="bar-count-inline tabular-nums">
                      <strong>{r.rate}%</strong> <small>({r.total} springs)</small>
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* CHART 3: Land Use Resilience */}
            <div className="chart-card">
              <div className="chart-card-header">
                <div className="chart-title-group">
                  <Trees size={14} className="chart-icon" />
                  <h5>Land Cover Hydrological Resilience</h5>
                </div>
                <span className="chart-badge success">Ecology</span>
              </div>
              <p className="chart-sub">
                Dense forest canopy provides strong groundwater buffer (17.3% dried), while settlement fringes exceed 45% desiccation.
              </p>
              <div className="custom-bar-chart">
                {(charts.landuse || []).map((lu) => (
                  <div key={lu.landuse} className="chart-bar-row">
                    <span className="bar-label truncate" title={lu.landuse}>{lu.landuse}</span>
                    <div className="bar-track-simple">
                      <div 
                        className={`bar-simple-fill ${lu.rate > 35 ? 'high' : lu.rate > 25 ? 'med' : 'low'}`}
                        style={{ width: `${lu.rate * 2.1}%` }}
                      />
                    </div>
                    <span className="bar-count-inline tabular-nums">
                      <strong>{lu.rate}%</strong>
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* CHART 4: Aspect Solar Deficit */}
            <div className="chart-card">
              <div className="chart-card-header">
                <div className="chart-title-group">
                  <Compass size={14} className="chart-icon" />
                  <h5>Slope Aspect &amp; Solar Insolation Polarity</h5>
                </div>
                <span className="chart-badge">Microclimate</span>
              </div>
              <p className="chart-sub">
                South and West facing slopes experience severe afternoon vapor pressure deficit and evapotranspiration stress.
              </p>
              <div className="custom-bar-chart">
                {(charts.aspect || []).map((asp) => (
                  <div key={asp.quadrant} className="chart-bar-row">
                    <span className="bar-label truncate" title={asp.quadrant}>{asp.quadrant}</span>
                    <div className="bar-track-simple">
                      <div 
                        className={`bar-simple-fill ${asp.rate > 28 ? 'high' : 'med'}`}
                        style={{ width: `${asp.rate * 2.8}%` }}
                      />
                    </div>
                    <span className="bar-count-inline tabular-nums">
                      <strong>{asp.rate}%</strong>
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* CHART 5: Recalled Community Causes */}
            <div className="chart-card">
              <div className="chart-card-header">
                <div className="chart-title-group">
                  <Activity size={14} className="chart-icon" />
                  <h5>Community Recalled Desiccation Drivers</h5>
                </div>
                <span className="chart-badge">Oral History</span>
              </div>
              <div className="causes-list">
                {(charts.causes || []).map((c) => (
                  <div key={c.cause} className="cause-item">
                    <div className="cause-header">
                      <span className="cause-name">{c.cause}</span>
                      <span className="cause-pct tabular-nums">{c.pct}% ({c.count})</span>
                    </div>
                    <div className="cause-bar-track">
                      <div className="cause-bar-fill" style={{ width: `${c.pct * 2.5}%` }} />
                    </div>
                    <div className="cause-desc">{c.desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: GOVERNANCE VS CATCHMENT */}
        {activeTab === 'governance' && (
          <div className="tab-pane">
            <div className="insight-hero-card">
              <div className="hero-kpi-row">
                <div className="hero-kpi-box highlight">
                  <div className="hero-label">Ward Boundary ICC</div>
                  <div className="hero-val tabular-nums">0.358</div>
                  <div className="hero-sub">Administrative Units</div>
                </div>
                <div className="hero-vs">vs</div>
                <div className="hero-kpi-box">
                  <div className="hero-label">Catchment ICC</div>
                  <div className="hero-val tabular-nums">0.141</div>
                  <div className="hero-sub">Hydrological Basins</div>
                </div>
                <div className="hero-kpi-box badge-box">
                  <div className="hero-label">Governance Ratio</div>
                  <div className="hero-val tabular-nums">2.54&times;</div>
                  <div className="hero-sub">Ward Dominance</div>
                </div>
              </div>

              <h4 className="insight-heading">The Hydro-Governance Discovery</h4>
              <p className="insight-p">
                In classical hydrology, natural watersheds and ridge boundaries are assumed to govern spring discharge.
                Our spatial mixed-effects analysis demonstrates that <strong>local municipal wards explain 2.54 times more variance</strong> in spring drying than topographic drainage basins.
              </p>

              <div className="insight-grid">
                <div className="insight-card">
                  <div className="card-badge danger">Primary Hazard Mechanism</div>
                  <h5>Unpaved Rural Road Excavation</h5>
                  <p>
                    Local ward infrastructure budgets construct low-cost farm tracks without engineered culverts or hydrogeological surveys, shearing shallow fractured aquifers and diverting recharge away from downslope communities.
                  </p>
                </div>

                <div className="insight-card">
                  <div className="card-badge info">Decentralized Management</div>
                  <h5>Local Water Diversion Schemes</h5>
                  <p>
                    Upstream ward user committees frequently install concrete intake tanks that tap source heads directly, depriving downstream traditional <em>dhunge dharas</em> and community springs of baseflow.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: PALIKA BENCHMARKS */}
        {activeTab === 'palikas' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>Palika Desiccation Rates (7 Surveyed Municipalities)</h4>
              <p>Click any municipality to highlight and center the map on its boundary and spring clusters.</p>
            </div>

            <div className="palika-list">
              {palikas.map((p) => {
                const isSelected = selectedPalika && p.name.toLowerCase().startsWith(selectedPalika.slice(0, 4).toLowerCase());
                return (
                  <div 
                    key={p.name}
                    className={`palika-item ${isSelected ? 'selected' : ''}`}
                    onClick={() => onSelectPalika(p.name)}
                  >
                    <div className="palika-row-top">
                      <div className="palika-name-group">
                        <span className="palika-name">{p.name}</span>
                        <span className="palika-elev tabular-nums">{p.elev}m mean elev</span>
                      </div>
                      <div className="palika-rate-group">
                        <span className={`rate-pill ${p.rate > 30 ? 'high' : p.rate > 20 ? 'med' : 'low'} tabular-nums`}>
                          {p.rate}% Dried
                        </span>
                      </div>
                    </div>

                    <div className="palika-bar-track">
                      <div 
                        className={`palika-bar-fill ${p.rate > 30 ? 'high' : p.rate > 20 ? 'med' : 'low'}`} 
                        style={{ width: `${p.rate * 2.2}%` }}
                      />
                    </div>

                    <div className="palika-row-bottom">
                      <span><strong>{p.count.toLocaleString()}</strong> springs ({p.dried} dried)</span>
                      <span>Avg {p.road}m to motorable road</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* TAB 4: LANDSCAPE TYPOLOGY */}
        {activeTab === 'typology' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>8 Discovered Micro-Catchment Typologies</h4>
              <p>Derived via hierarchical spatial clustering over elevation, slope aspect, topographic wetness, and road proximity.</p>
            </div>

            <div className="typology-grid">
              {(data?.typology || []).map((t) => (
                <div key={t.id} className="typology-card">
                  <div className="typology-card-header">
                    <span className="typology-tag tabular-nums">{t.id}</span>
                    <span className={`typology-risk ${t.risk.toLowerCase()}`}>{t.risk} Risk</span>
                  </div>
                  <h5 className="typology-title">{t.name}</h5>
                  <div className="typology-stats">
                    <div>Rate: <strong className="tabular-nums">{t.rate}%</strong></div>
                    <div>Elev: <strong className="tabular-nums">{t.elev}</strong></div>
                  </div>
                  <div className="typology-desc">{t.hazard}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 5: ICIMOD & CREDITS */}
        {activeTab === 'credits' && (
          <div className="tab-pane">
            {/* ICIMOD Attribution Card */}
            <div className="credits-card primary">
              <div className="credits-header">
                <ShieldCheck size={18} className="credits-icon icimod" />
                <div>
                  <h4>Data Collection &amp; Field Survey</h4>
                  <h5>International Centre for Integrated Mountain Development (ICIMOD)</h5>
                </div>
              </div>
              <p className="credits-body">
                The baseline dataset of <strong>5,689 springs</strong> across Kavrepalanchok District was surveyed and published by <strong>ICIMOD</strong> in direct partnership with local municipal governments and the <em>Ministry of Energy, Water Resources and Irrigation, Government of Nepal</em> under Open Data principles.
              </p>
              <div className="credits-meta">
                <span>Coverage: 13 Municipalities (7 completely surveyed)</span>
                <span>Coordinates: WGS84 GPS Field Verification</span>
              </div>
            </div>

            {/* Author & Organization Attribution */}
            <div className="credits-card secondary">
              <div className="credits-header">
                <Globe size={18} className="credits-icon noesis" />
                <div>
                  <h4>Research &amp; Spatial Data Mining</h4>
                  <h5>Ashim Shrestha &bull; Noesis Frameworks</h5>
                </div>
              </div>
              <p className="credits-body">
                Spatial data mining, graph-based hydrological modeling, causal inference, and interactive GIS engineering developed at <strong>Noesis Frameworks</strong>.
              </p>
              
              <div className="links-group">
                <a 
                  href="https://noesisframeworks.com" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="credit-link-btn"
                >
                  <Globe size={14} />
                  <span>noesisframeworks.com</span>
                  <ExternalLink size={12} />
                </a>

                <a 
                  href="https://github.com/ashimst/kavre_springs" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="credit-link-btn"
                >
                  <GithubIcon size={14} />
                  <span>GitHub Repository</span>
                  <ExternalLink size={12} />
                </a>

                <a 
                  href="https://github.com/ashimst" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="credit-link-btn"
                >
                  <GithubIcon size={14} />
                  <span>Author GitHub Profile (@ashimst)</span>
                  <ExternalLink size={12} />
                </a>
              </div>
            </div>

            {/* Open Source / Cloudflare Edge Notice */}
            <div className="credits-footer-note">
              Designed for static edge deployment on <strong>Cloudflare Pages</strong> using React 19, Vite, and Leaflet hardware-accelerated Canvas.
            </div>
          </div>
        )}

      </div>
    </aside>
  );
}
