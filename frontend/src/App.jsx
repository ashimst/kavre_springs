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
  const [activeBasemap, setActiveBasemap] = useState('topo');
  const [showWards, setShowWards] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(true);

  // Live counts for the current palika and status filter. Each source is
  // [lat, lon, dried, elevation, palika, type, recorded ward].
  const { filteredCount, filteredDriedCount } = useMemo(() => {
    if (!data?.sources) return { filteredCount: 0, filteredDriedCount: 0 };
    const isAll = !selectedPalika || selectedPalika === 'All Palikas';
    let count = 0;
    let driedCount = 0;
    for (const s of data.sources) {
      if (!isAll && s[4] !== selectedPalika) continue;
      if (statusFilter === 'dried' && s[2] === 0) continue;
      if (statusFilter === 'active' && s[2] === 1) continue;
      count++;
      if (s[2] === 1) driedCount++;
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
      <Header
        selectedPalika={selectedPalika}
        onSelectPalika={setSelectedPalika}
        statusFilter={statusFilter}
        onStatusFilterChange={setStatusFilter}
        activeBasemap={activeBasemap}
        onBasemapChange={setActiveBasemap}
        showWards={showWards}
        onToggleWards={() => setShowWards(!showWards)}
        onResetView={handleResetView}
        isDrawerOpen={isDrawerOpen}
        onToggleDrawer={() => setIsDrawerOpen(!isDrawerOpen)}
        filteredCount={filteredCount}
        filteredDriedCount={filteredDriedCount}
      />

      <main className="map-viewport">
        <Map
          data={data}
          selectedPalika={selectedPalika}
          onSelectPalika={setSelectedPalika}
          statusFilter={statusFilter}
          activeBasemap={activeBasemap}
          showWards={showWards}
        />
      </main>

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
