import React from 'react';
import { Droplets, Filter, RefreshCw, BarChart3, Layers, MapPin } from 'lucide-react';

export default function Header({
  data,
  selectedPalika,
  onSelectPalika,
  statusFilter,
  onStatusFilterChange,
  activeBasemap,
  onBasemapChange,
  showPalikaBorders,
  onTogglePalikaBorders,
  onResetView,
  isDrawerOpen,
  onToggleDrawer,
  filteredCount,
  filteredDriedCount
}) {
  const dryingPct = filteredCount > 0 
    ? ((filteredDriedCount / filteredCount) * 100).toFixed(1) 
    : 0;

  // List unique municipalities from data
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
      {/* Brand & Title */}
      <div className="header-left">
        <div className="brand-badge">
          <Droplets size={18} className="brand-icon" />
          <span className="brand-title">KAVRE SPRINGS</span>
          <span className="brand-tag">GIS DATA MINING</span>
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
      </div>

      {/* Center Controls: Filter & Palika selection */}
      <div className="header-center">
        {/* Palika Dropdown */}
        <div className="control-group">
          <MapPin size={14} className="control-icon" />
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

        {/* Basemap Switcher */}
        <div className="control-group">
          <Layers size={14} className="control-icon" />
          <select 
            value={activeBasemap} 
            onChange={(e) => onBasemapChange(e.target.value)}
            className="header-select"
            aria-label="Select Basemap"
          >
            <option value="carto-light">Light (Carto)</option>
            <option value="osm">OpenStreetMap</option>
            <option value="satellite">Satellite (Esri Imagery)</option>
            <option value="topo">Topographic (Esri)</option>
          </select>
        </div>
      </div>

      {/* Right Controls: Reset & Toggle Drawer */}
      <div className="header-right">
        <button 
          type="button" 
          onClick={onResetView}
          className="header-action-btn"
          title="Reset map view to entire district"
        >
          <RefreshCw size={14} />
          <span>Reset View</span>
        </button>

        <button 
          type="button" 
          onClick={onToggleDrawer}
          className={`header-action-btn drawer-toggle-btn ${isDrawerOpen ? 'drawer-active' : ''}`}
          title="Toggle scientific findings and data mining drawer"
        >
          <BarChart3 size={15} />
          <span>{isDrawerOpen ? 'Close Insights' : 'Data Mining Insights'}</span>
        </button>
      </div>
    </header>
  );
}
