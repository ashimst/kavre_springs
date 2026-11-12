import React, { useMemo, useState } from 'react';
import {
  Building2,
  Map,
  ClipboardCheck,
  Activity,
  BarChart2,
  BookOpen,
  Globe,
  ExternalLink,
  ShieldCheck,
  Mountain,
  Compass,
  Trees,
  Truck,
  CalendarDays,
  AlertTriangle
} from 'lucide-react';

const GithubIcon = ({ size = 14 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

const rateClass = (rate, hi = 30, mid = 20) => (rate > hi ? 'high' : rate > mid ? 'med' : 'low');
const fmt = (v, d = 3) => (v === null || v === undefined ? '–' : Number(v).toFixed(d));
const signed = (v, d = 3) => (v < 0 ? `−${Math.abs(v).toFixed(d)}` : v.toFixed(d));

function RateChart({ rows, scale = 2.2, hi = 30, mid = 20 }) {
  return (
    <div className="custom-bar-chart">
      {rows.map((r) => (
        <div key={r.label} className="chart-bar-row">
          <span className="bar-label truncate tabular-nums" title={r.label}>{r.label}</span>
          <div className="bar-track-simple">
            <div className={`bar-simple-fill ${rateClass(r.rate, hi, mid)}`} style={{ width: `${Math.min(r.rate * scale, 100)}%` }} />
          </div>
          <span className="bar-count-inline tabular-nums" title={`${r.dried.toLocaleString()} of ${r.total.toLocaleString()} sources`}>
            <strong>{r.rate}%</strong>
          </span>
        </div>
      ))}
    </div>
  );
}

function ChartCard({ icon: Icon, title, badge, note, children }) {
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <div className="chart-title-group">
          <Icon size={14} className="chart-icon" />
          <h5>{title}</h5>
        </div>
        {badge && <span className="chart-badge">{badge}</span>}
      </div>
      {note && <p className="chart-sub">{note}</p>}
      {children}
    </div>
  );
}

export default function Drawer({ isOpen, onClose, data, selectedPalika, onSelectPalika }) {
  const [activeTab, setActiveTab] = useState('analytics');

  const palikas = useMemo(() => (data?.palikas?.features || [])
    .map((f) => f.properties)
    .filter((p) => p.surveyed)
    .sort((a, b) => a.drying_rate - b.drying_rate), [data]);

  if (!isOpen) return null;

  const charts = data?.charts || {};
  const f = data?.findings || {};
  const icc = f.icc || {};
  const credits = data?.credits || {};
  const years = charts.years || [];
  const maxYear = Math.max(1, ...years.map((y) => y.count));

  const tabs = [
    { id: 'analytics', icon: BarChart2, label: 'Descriptive charts' },
    { id: 'units', icon: Building2, label: 'Wards & catchments' },
    { id: 'palikas', icon: Map, label: 'Palikas' },
    { id: 'audit', icon: ClipboardCheck, label: 'Survey audit' },
    { id: 'credits', icon: BookOpen, label: 'Data & credits' }
  ];

  return (
    <aside className="insights-drawer glass-panel" aria-label="Findings panel">
      <div className="drawer-header">
        <div className="drawer-nav">
          {tabs.map(({ id, icon: Icon, label }) => (
            <button key={id} type="button" className={`tab-btn ${activeTab === id ? 'active' : ''}`} onClick={() => setActiveTab(id)}>
              <Icon size={14} />
              <span>{label}</span>
            </button>
          ))}
        </div>
        <button type="button" onClick={onClose} className="drawer-close-btn" title="Minimize drawer">&times;</button>
      </div>

      <div className="drawer-content">

        {/* TAB 1: DESCRIPTIVE CHARTS */}
        {activeTab === 'analytics' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>Share of mapped sources recorded as dried</h4>
              <p>
                {data.totalSources.toLocaleString()} springs and ponds ({data.springs.toLocaleString()} springs, {data.ponds.toLocaleString()} ponds);
                {' '}{data.dryingRate}% are recorded as dried. These are raw shares of mapped records, not adjusted for other
                variables and not estimates of district prevalence.
              </p>
            </div>

            <ChartCard icon={Mountain} title="By elevation" badge="Terrain">
              <RateChart rows={charts.elevation || []} />
            </ChartCard>

            <ChartCard icon={Truck} title="By distance to the nearest road" badge="Infrastructure"
              note="An association in the raw data; roads also follow settlements, valleys and lower ground.">
              <RateChart rows={charts.road || []} />
            </ChartCard>

            <ChartCard icon={Trees} title="By land use recorded in the survey" badge="Survey field"
              note="Land use was recorded by the survey teams and may partly reflect drying rather than precede it.">
              <RateChart rows={charts.landuse || []} scale={1.8} hi={35} mid={25} />
            </ChartCard>

            <ChartCard icon={Compass} title="By slope aspect" badge="Terrain">
              <RateChart rows={charts.aspect || []} />
            </ChartCard>

            <ChartCard icon={Activity} title="Recalled cause of drying" badge="Community recall"
              note={`Among the ${data.totalDried.toLocaleString()} dried sources. Causes are recalled by communities, often years later, and are not verified.`}>
              <div className="causes-list">
                {(charts.causes || []).map((c) => (
                  <div key={c.label} className="cause-item">
                    <div className="cause-header">
                      <span className="cause-name">{c.label}</span>
                      <span className="cause-pct tabular-nums">{c.pct}% ({c.count})</span>
                    </div>
                    <div className="cause-bar-track">
                      <div className="cause-bar-fill" style={{ width: `${Math.min(c.pct * 2.5, 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </ChartCard>

            <ChartCard icon={CalendarDays} title="Reported year of drying (treated as Bikram Sambat)" badge="Community recall"
              note="2072 BS spans April 2015 to April 2016. The field wording is ambiguous and visit dates are unavailable, so the year coding is a working interpretation.">
              <div className="year-chart" role="img" aria-label="Dried sources by reported year">
                {years.map((y) => (
                  <div key={y.year} className={`year-col ${y.year === 2072 ? 'quake' : ''}`} title={`${y.year} BS: ${y.count}`}>
                    <div className="year-bar" style={{ height: `${(y.count / maxYear) * 100}%` }} />
                  </div>
                ))}
              </div>
              <div className="year-axis tabular-nums"><span>{years[0]?.year} BS</span><span>red: 2072 BS</span><span>{years[years.length - 1]?.year} BS</span></div>
            </ChartCard>
          </div>
        )}

        {/* TAB 2: WARDS AND CATCHMENTS */}
        {activeTab === 'units' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>Does drying cluster within wards or within catchments?</h4>
              <p>
                Clustering is measured by the intraclass correlation (ICC): the share of variation in recorded drying that
                lies between units. It is shown on two scales. The observed (ANOVA) scale is the paper's main measure; the
                latent (logit) scale does not depend on how common drying is.
              </p>
            </div>

            <div className="chart-card">
              <h5 className="table-title">ICC of recorded drying by unit</h5>
              <table className="data-table tabular-nums">
                <thead><tr><th>Unit</th><th>Observed</th><th>Latent</th></tr></thead>
                <tbody>
                  {(icc.units || []).map((u) => (
                    <tr key={u.unit}>
                      <td>{u.unit}</td>
                      <td>{fmt(u.observed)}{u.observed_beats_grids && <span className="mark" title="Above all matched rotated grids"> ▲</span>}</td>
                      <td>{fmt(u.latent)}{u.latent_beats_grids && <span className="mark" title="Above all matched rotated grids"> ▲</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="table-note">▲ above all 1,000 rotated grids with the same number of cells. Smaller units have higher raw ICCs, so values are compared with their own grids, not with each other.</p>
            </div>

            <div className="chart-card">
              <h5 className="table-title">Wards against random reference partitions</h5>
              <table className="data-table tabular-nums">
                <thead><tr><th>Outcome</th><th>Reference (1,000 draws)</th><th>Reached by: observed</th><th>latent</th></tr></thead>
                <tbody>
                  {(icc.references || []).map((r) => (
                    <tr key={`${r.outcome}-${r.reference}`}>
                      <td>{r.outcome}</td>
                      <td>{r.reference}</td>
                      <td>{r.observed_exceed}</td>
                      <td className={r.latent_exceed > 0 ? 'warn-cell' : ''}>{r.latent_exceed}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="table-note">
                Number of random partitions whose ICC reached the ward ICC. Wards exceed contiguous random regions on both
                scales, and exceed every reference after covariate adjustment. Unadjusted wards do not exceed rotated grids on
                the latent scale.
              </p>
            </div>

            <div className="chart-card">
              <h5 className="table-title">Direct ward–catchment differences (ICC)</h5>
              <table className="data-table tabular-nums">
                <thead><tr><th>Contrast</th><th>Scale</th><th>Difference (95% interval)</th></tr></thead>
                <tbody>
                  {(icc.contrasts || []).map((c) => (
                    <tr key={`${c.contrast}-${c.scale}`}>
                      <td>{c.contrast}</td>
                      <td>{c.scale}</td>
                      <td>{signed(c.difference)} ({signed(c.low)} to {signed(c.high)})</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="table-note">The data do not rank wards above catchments: the one observed-scale difference that excludes zero disappears on the latent scale.</p>
            </div>

            <div className="insight-card caution-card">
              <div className="card-badge info"><AlertTriangle size={11} /> How to read this</div>
              <p>
                Ward clustering is descriptive. Simulations show these reference comparisons can over-reject under spatial
                autocorrelation alone, and the ward signal cannot be separated from survey team, visit timing or ward-level
                water and road schemes. It is not evidence that ward governance causes drying.
              </p>
            </div>

            {(f.withinWardDrivers || []).length > 0 && (
              <div className="chart-card">
                <h5 className="table-title">Terrain associations that remain within wards</h5>
                <table className="data-table tabular-nums">
                  <thead><tr><th>Covariate</th><th>Log-odds per SD (95% interval)</th></tr></thead>
                  <tbody>
                    {f.withinWardDrivers.map((d) => (
                      <tr key={d.term}><td>{d.term}</td><td>{d.beta.toFixed(2)} ({d.low.toFixed(2)} to {d.high.toFixed(2)})</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="table-note">From a ten-covariate model with ward fixed effects; exploratory and not adjusted for multiple comparisons.</p>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: PALIKAS */}
        {activeTab === 'palikas' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>Seven surveyed palikas</h4>
              <p>Share of mapped sources recorded as dried. Click a palika to centre the map on it.</p>
            </div>
            <div className="palika-list">
              {palikas.map((p) => {
                const isSelected = selectedPalika === p.name;
                return (
                  <div key={p.name} className={`palika-item ${isSelected ? 'selected' : ''}`} onClick={() => onSelectPalika(p.name)}>
                    <div className="palika-row-top">
                      <div className="palika-name-group">
                        <span className="palika-name">{p.name}</span>
                        <span className="palika-elev tabular-nums">{p.mean_elev.toLocaleString()} m mean elevation</span>
                      </div>
                      <div className="palika-rate-group">
                        <span className={`rate-pill ${rateClass(p.drying_rate)} tabular-nums`}>{p.drying_rate}% dried</span>
                      </div>
                    </div>
                    <div className="palika-bar-track">
                      <div className={`palika-bar-fill ${rateClass(p.drying_rate)}`} style={{ width: `${Math.min(p.drying_rate * 2.2, 100)}%` }} />
                    </div>
                    <div className="palika-row-bottom">
                      <span><strong>{p.total_sources.toLocaleString()}</strong> sources ({p.dried_sources} dried)</span>
                      <span>Mean {p.mean_road} m to a road</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* TAB 4: SURVEY AUDIT */}
        {activeTab === 'audit' && (
          <div className="tab-pane">
            <div className="tab-intro">
              <h4>What the survey design does to the data</h4>
              <p>Recorded condition can reflect how the inventory was collected as well as the landscape.</p>
            </div>

            <div className="insight-card">
              <div className="card-badge danger">Questionnaire skip logic</div>
              <p>
                {f.skipLogic?.outcomeDependent} of {f.skipLogic?.fields} raw survey fields have missingness that depends on
                whether the source was recorded as dried. Whether the earthquake question was answered alone predicts the label
                with ROC-AUC {fmt(f.skipLogic?.completionIndicatorAuc)}. These fields are excluded from every model.
              </p>
            </div>

            <div className="insight-card">
              <div className="card-badge info">Leak-free prediction</div>
              <p>
                With terrain, climate, hydrology and road covariates only, spatially blocked ROC-AUC is
                {' '}{fmt(f.prediction?.aucLow, 2)}–{fmt(f.prediction?.aucHigh, 2)}; ranking within a ward reaches about
                {' '}{fmt(f.prediction?.withinWardHigh, 2)}. Model scores may help design a validation sample, not select sites.
              </p>
            </div>

            <div className="insight-card">
              <div className="card-badge info">Recorded location vs boundaries</div>
              <p>
                {f.location?.insidePalika}% of sources lie inside their recorded palika. Against OpenStreetMap ward polygons,
                {' '}{f.location?.insideWard}% lie inside their recorded ward and {f.location?.within100m}% within 100 m of it;
                {' '}{f.location?.adjacentWard}% of the rest fall in a neighbouring ward. In {f.location?.wardsUnderHalf} wards fewer
                than half of the sources fall inside the polygon of the same name, a pattern more consistent with ward numbering
                that differs between the survey and the map than with GPS error. Turn on <em>Ward boundaries</em> on the map to see them.
              </p>
            </div>

            <div className="insight-card">
              <div className="card-badge info">Missing information</div>
              <p>
                The inventory has no independent frame of all local water sources, no enumerator identifiers and no visit dates,
                and the drying year and cause rest on recall. The share recorded as dried describes mapped records, not true
                vulnerability.
              </p>
            </div>
          </div>
        )}

        {/* TAB 5: DATA & CREDITS */}
        {activeTab === 'credits' && (
          <div className="tab-pane">
            <div className="credits-card primary">
              <div className="credits-header">
                <ShieldCheck size={18} className="credits-icon icimod" />
                <div>
                  <h4>Data</h4>
                  <h5>{credits.dataProvider}</h5>
                </div>
              </div>
              <p className="credits-body">
                <em>{credits.dataTitle}</em>. The inventory of {data.totalSources.toLocaleString()} springs and ponds was collected
                and published by ICIMOD; the authors of this analysis were not involved in its collection. Licence: {credits.dataLicence}.
              </p>
              <div className="links-group">
                <a href={credits.dataDoi} target="_blank" rel="noopener noreferrer" className="credit-link-btn">
                  <BookOpen size={14} /><span>Dataset DOI</span><ExternalLink size={12} />
                </a>
              </div>
              <div className="credits-meta">
                <span>Palika boundaries: MoFAGA via geoBoundaries (CC BY-SA 4.0)</span>
                <span>Ward boundaries: OpenStreetMap contributors (ODbL)</span>
              </div>
            </div>

            <div className="credits-card secondary">
              <div className="credits-header">
                <Globe size={18} className="credits-icon noesis" />
                <div>
                  <h4>Analysis</h4>
                  <h5>{(credits.authors || []).join(' & ')} &bull; {credits.organization}</h5>
                </div>
              </div>
              <p className="credits-body">
                A spatial audit of the inventory: survey-field leakage checks, spatially blocked prediction, and comparisons of
                wards and catchments against random reference partitions. Every number on this site is exported from the
                analysis pipeline in the repository.
              </p>
              <div className="links-group">
                <a href={credits.orgUrl} target="_blank" rel="noopener noreferrer" className="credit-link-btn">
                  <Globe size={14} /><span>noesisframeworks.com</span><ExternalLink size={12} />
                </a>
                <a href={credits.githubUrl} target="_blank" rel="noopener noreferrer" className="credit-link-btn">
                  <GithubIcon size={14} /><span>Code and data (GitHub)</span><ExternalLink size={12} />
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
