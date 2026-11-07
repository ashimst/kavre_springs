import React, { useState } from 'react';
import { 
  Building2, 
  Map, 
  Layers, 
  Activity, 
  ChevronDown, 
  ChevronUp, 
  AlertTriangle, 
  ShieldCheck, 
  Compass, 
  ArrowUpRight 
} from 'lucide-react';

export default function Drawer({
  isOpen,
  onClose,
  data,
  selectedPalika,
  onSelectPalika
}) {
  const [activeTab, setActiveTab] = useState('governance');

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

  return (
    <aside className="insights-drawer glass-panel" aria-label="Scientific Insights Panel">
      {/* Drawer Header & Tabs */}
      <div className="drawer-header">
        <div className="drawer-nav">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'governance' ? 'active' : ''}`}
            onClick={() => setActiveTab('governance')}
          >
            <Building2 size={15} />
            <span>Governance vs Catchment</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'palikas' ? 'active' : ''}`}
            onClick={() => setActiveTab('palikas')}
          >
            <Map size={15} />
            <span>Palika Benchmarks</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'typology' ? 'active' : ''}`}
            onClick={() => setActiveTab('typology')}
          >
            <Layers size={15} />
            <span>Landscape Typology</span>
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'earthquake' ? 'active' : ''}`}
            onClick={() => setActiveTab('earthquake')}
          >
            <Activity size={15} />
            <span>Earthquake &amp; Rules</span>
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
        {/* TAB 1: GOVERNANCE VS CATCHMENT */}
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

        {/* TAB 2: PALIKA BENCHMARKS */}
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

                    {/* Bar visualization */}
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

        {/* TAB 3: LANDSCAPE TYPOLOGY */}
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

        {/* TAB 4: EARTHQUAKE & RULES */}
        {activeTab === 'earthquake' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>2015 Gorkha Earthquake Desiccation Pulse</h4>
              <p>Historical recording year of dried springs reveals an unprecedented spike immediately following the April 2015 Mw 7.8 seismic rupture.</p>
            </div>

            <div className="timeline-container">
              <div className="timeline-bars">
                {(data?.timeline || []).map((item) => (
                  <div key={item.year} className={`timeline-col ${item.isQuake ? 'quake-year' : ''}`}>
                    <div className="bar-wrapper">
                      <span className="bar-val tabular-nums">{item.count}</span>
                      <div 
                        className={`bar-visual ${item.isQuake ? 'quake' : ''}`}
                        style={{ height: `${Math.min(100, Math.max(10, (item.count / 538) * 90))}px` }}
                      />
                    </div>
                    <span className="timeline-year tabular-nums">{item.year}</span>
                    {item.isQuake && <span className="quake-tag">Mw 7.8</span>}
                  </div>
                ))}
              </div>
            </div>

            <div className="tab-intro" style={{ marginTop: '20px' }}>
              <h4>Top Spatial Association Rules (Apriori Mining)</h4>
              <p>Discovered high-confidence feature co-occurrences explaining spring state transitions.</p>
            </div>

            <div className="rules-list">
              {(data?.rules || []).map((r, idx) => (
                <div key={idx} className="rule-card">
                  <div className="rule-header">
                    <span className="rule-antecedent">{r.rule}</span>
                    <span className="rule-arrow">&rarr;</span>
                    <span className={`rule-consequent ${r.consequent.toLowerCase()}`}>{r.consequent}</span>
                  </div>
                  <div className="rule-metrics">
                    <span>Confidence: <strong className="tabular-nums">{r.confidence}</strong></span>
                    <span>Lift: <strong className="tabular-nums">{r.lift}</strong></span>
                  </div>
                  <p className="rule-desc">{r.desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
