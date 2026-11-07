import React from 'react';
import { 
  Droplets, 
  Layers, 
  MapPin, 
  RefreshCw, 
  BarChart3, 
  Globe, 
  ExternalLink,
  Database
} from 'lucide-react';

const GithubIcon = ({ size = 13 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

export default function Header({
  data,
  selectedPalika,
  onSelectPalika,
  statusFilter,
  onStatusFilterChange,
  activeBasemap,
  onBasemapChange,
  onResetView,
  isDrawerOpen,
  onToggleDrawer,
  filteredCount,
  filteredDriedCount
}) {
  const dryingPct = filteredCount > 0 
    ? ((filteredDriedCount / filteredCount) * 100).toFixed(1) 
    : 0;

  const palikaNames = [
    'All Palikas',
    'Panauti',
    'Bethanchowk',
    'Namobuddha',
    'Roshi',
    'Dhulikhel',
    'Temal',
    'Panchkhal',
    'Banepa',
    'Mandandeupur',
    'Bhumlu',
    'Chaurideurali',
    'Khanikhola',
    'Mahabharat'
  ];

  return (
    <header className="header-bar glass-panel">
      {/* Brand & Stats */}
      <div className="header-left">
        <div className="brand-badge">
          <Droplets size={18} className="brand-icon" />
          <span className="brand-title">KAVRE SPRINGS</span>
          <span className="brand-tag">GIS MINING</span>
        </div>
        
        <div className="stats-pill">
          <span className="stat-item">
            <strong>{filteredCount.toLocaleString()}</strong> <span className="stat-label">Springs</span>
          </span>
          <span className="stat-divider">/</span>
          <span className="stat-item dried-stat">
            <strong>{filteredDriedCount.toLocaleString()}</strong> <span className="stat-label">Dried ({dryingPct}%)</span>
          </span>
          <span className="stat-divider">/</span>
          <span className="stat-item active-stat">
            <strong>{(filteredCount - filteredDriedCount).toLocaleString()}</strong> <span className="stat-label">Active</span>
          </span>
        </div>

        {/* Data Credit Pill */}
        <div className="credit-pill" title="5,689 GPS spring survey conducted by ICIMOD (International Centre for Integrated Mountain Development)">
          <Database size={12} className="credit-icon" />
          <span>Data: <strong>ICIMOD</strong></span>
        </div>
      </div>

      {/* Center Controls: Filter & Palika selection */}
      <div className="header-center">
        {/* Palika Dropdown */}
        <div className="control-group">
          <MapPin size={13} className="control-icon" />
          <select 
            value={selectedPalika} 
            onChange={(e) => onSelectPalika(e.target.value)}
            className="header-select"
            aria-label="Select Palika"
          >
            {palikaNames.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>

        {/* Status Filter buttons */}
        <div className="filter-button-group">
          <button 
            type="button"
            className={`filter-btn ${statusFilter === 'all' ? 'active' : ''}`}
            onClick={() => onStatusFilterChange('all')}
          >
            All
          </button>
          <button 
            type="button"
            className={`filter-btn dried ${statusFilter === 'dried' ? 'active' : ''}`}
            onClick={() => onStatusFilterChange('dried')}
          >
            <span className="dot dried-dot"></span> Dried
          </button>
          <button 
            type="button"
            className={`filter-btn active-btn ${statusFilter === 'active' ? 'active' : ''}`}
            onClick={() => onStatusFilterChange('active')}
          >
            <span className="dot active-dot"></span> Active
          </button>
        </div>

        {/* Basemap Switcher (Zero API keys) */}
        <div className="control-group">
          <Layers size={13} className="control-icon" />
          <select 
            value={activeBasemap} 
            onChange={(e) => onBasemapChange(e.target.value)}
            className="header-select"
            aria-label="Select Basemap"
          >
            <option value="topo">Topographic (Esri)</option>
            <option value="satellite">Satellite (Esri Imagery)</option>
            <option value="osm">OpenStreetMap</option>
            <option value="gray">Light Canvas (Esri)</option>
          </select>
        </div>
      </div>

      {/* Right Controls: External Links & Drawer Toggle */}
      <div className="header-right">
        {/* Noesis Frameworks link */}
        <a 
          href="https://noesisframeworks.com" 
          target="_blank" 
          rel="noopener noreferrer"
          className="header-link-btn"
          title="Visit Noesis Frameworks"
        >
          <Globe size={13} />
          <span>noesisframeworks.com</span>
          <ExternalLink size={10} className="ext-icon" />
        </a>

        {/* User GitHub profile / repo */}
        <a 
          href="https://github.com/ashimst/kavre_springs" 
          target="_blank" 
          rel="noopener noreferrer"
          className="header-link-btn github"
          title="View source on GitHub (ashimst)"
        >
          <GithubIcon size={13} />
          <span>GitHub</span>
        </a>

        <button 
          type="button" 
          onClick={onResetView}
          className="header-action-btn"
          title="Reset map view to entire district"
        >
          <RefreshCw size={13} />
        </button>

        <button 
          type="button" 
          onClick={onToggleDrawer}
          className={`header-action-btn drawer-toggle-btn ${isDrawerOpen ? 'drawer-active' : ''}`}
          title="Toggle scientific findings and data mining drawer"
        >
          <BarChart3 size={14} />
          <span>{isDrawerOpen ? 'Close Insights' : 'Analytics & Insights'}</span>
        </button>
      </div>
    </header>
  );
}
