import React, { useState, useMemo } from 'react';
import Header from './components/Header';
import Map from './components/Map';
import Drawer from './components/Drawer';
import dataJson from './data.json';
import './App.css';

export default function App() {
  const [data] = useState(dataJson);
  const [selectedPalika, setSelectedPalika] = useState('All Palikas');
  const [statusFilter, setStatusFilter] = useState('all'); // 'all' | 'dried' | 'active'
  const [activeBasemap, setActiveBasemap] = useState('carto-light');
  const [isDrawerOpen, setIsDrawerOpen] = useState(true);
  const [hoveredPalika, setHoveredPalika] = useState(null);

  // Compute live filtered spring counts
  const { filteredCount, filteredDriedCount } = useMemo(() => {
    if (!data || !data.springs) return { filteredCount: 0, filteredDriedCount: 0 };

    const isAllPalikas = !selectedPalika || selectedPalika === 'All Palikas';
    let count = 0;
    let driedCount = 0;

    for (let i = 0; i < data.springs.length; i++) {
      const sp = data.springs[i];
      const dried = sp[2];
      const muni = sp[4];

      if (!isAllPalikas) {
        if (!muni.toLowerCase().startsWith(selectedPalika.slice(0, 4).toLowerCase())) {
          continue;
        }
      }

      if (statusFilter === 'dried' && dried === 0) continue;
      if (statusFilter === 'active' && dried === 1) continue;

      count++;
      if (dried === 1) driedCount++;
    }

    return { filteredCount: count, filteredDriedCount: driedCount };
  }, [data, selectedPalika, statusFilter]);

  const handleResetView = () => {
    setSelectedPalika('All Palikas');
    setStatusFilter('all');
    window.location.hash = '';
  };

  return (
    <div className="app-container">
      {/* Top Floating Header & Filter Controls */}
      <Header
        data={data}
        selectedPalika={selectedPalika}
        onSelectPalika={setSelectedPalika}
        statusFilter={statusFilter}
        onStatusFilterChange={setStatusFilter}
        activeBasemap={activeBasemap}
        onBasemapChange={setActiveBasemap}
        onResetView={handleResetView}
        isDrawerOpen={isDrawerOpen}
        onToggleDrawer={() => setIsDrawerOpen(!isDrawerOpen)}
        filteredCount={filteredCount}
        filteredDriedCount={filteredDriedCount}
      />

      {/* Main Interactive Map Viewport */}
      <main className="map-viewport">
        <Map
          data={data}
          selectedPalika={selectedPalika}
          onSelectPalika={setSelectedPalika}
          statusFilter={statusFilter}
          activeBasemap={activeBasemap}
          onHoverPalika={setHoveredPalika}
        />
      </main>

      {/* Collapsible Scientific Findings & Data Mining Drawer */}
      <Drawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        data={data}
        selectedPalika={selectedPalika}
        onSelectPalika={setSelectedPalika}
      />
    </div>
  );
}
