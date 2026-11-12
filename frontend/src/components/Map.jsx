import React, { useEffect, useRef } from 'react';
import L from 'leaflet';

export default function Map({
  data,
  selectedPalika,
  onSelectPalika,
  statusFilter,
  activeBasemap,
  showWards
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const tileLayerRef = useRef(null);
  const districtLayerRef = useRef(null);
  const palikasLayerRef = useRef(null);
  const springsLayerRef = useRef(null);
  const wardsLayerRef = useRef(null);
  const canvasRendererRef = useRef(null);

  // Zero-API-Key Open GIS Basemaps
  const basemaps = {
    'topo': {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
      attrib: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ, TomTom, Intermap, iPC, USGS, FAO, NPS, NRCAN, GeoBase, Kadaster NL, Ordnance Survey, Esri Japan, METI, Esri China (Hong Kong), and the GIS User Community'
    },
    'osm': {
      url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      attrib: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    },
    'satellite': {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attrib: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    },
    'gray': {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      attrib: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
    }
  };

  // Color generator for Palikas
  const getPalikaStyle = (feature, isHovered = false) => {
    const p = feature.properties;
    if (!p.surveyed) {
      return {
        fillColor: '#f1f5f9',
        fillOpacity: isHovered ? 0.5 : 0.25,
        color: '#94a3b8',
        weight: isHovered ? 2 : 1,
        dashArray: '3, 4'
      };
    }

    const rate = p.drying_rate;
    let fillColor = '#d1fae5'; // < 15% (sage)
    let strokeColor = '#10b981';

    if (rate >= 35) {
      fillColor = '#fee2e2'; // >= 35% (brick rose)
      strokeColor = '#ef4444';
    } else if (rate >= 25) {
      fillColor = '#ffedd5'; // 25-35% (terracotta orange)
      strokeColor = '#f97316';
    } else if (rate >= 15) {
      fillColor = '#fef3c7'; // 15-25% (warm sand ochre)
      strokeColor = '#f59e0b';
    }

    const isSelected = selectedPalika === p.name;

    return {
      fillColor,
      fillOpacity: isHovered || isSelected ? 0.65 : 0.35,
      color: isSelected ? '#1e293b' : strokeColor,
      weight: isSelected ? 2.5 : (isHovered ? 2 : 1.2),
      dashArray: null
    };
  };

  // 1. Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [27.56, 85.60],
      zoom: 11,
      minZoom: 9,
      maxZoom: 18,
      zoomControl: false
    });

    // Custom zoom control in top-right
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Initial tile layer
    const conf = basemaps[activeBasemap] || basemaps['topo'];
    const tileLayer = L.tileLayer(conf.url, {
      attribution: conf.attrib,
      subdomains: conf.subdomains || 'abc',
      maxZoom: 19
    }).addTo(map);

    tileLayerRef.current = tileLayer;
    mapInstanceRef.current = map;
    canvasRendererRef.current = L.canvas({ padding: 0.5 });

    return () => {
      map.remove();
    };
  }, []);

  // 2. Handle Basemap Changes
  useEffect(() => {
    if (!mapInstanceRef.current || !tileLayerRef.current) return;

    const conf = basemaps[activeBasemap] || basemaps['topo'];
    mapInstanceRef.current.removeLayer(tileLayerRef.current);

    const newLayer = L.tileLayer(conf.url, {
      attribution: conf.attrib,
      subdomains: conf.subdomains || 'abc',
      maxZoom: 19
    }).addTo(mapInstanceRef.current);

    tileLayerRef.current = newLayer;
  }, [activeBasemap]);

  // 3. Render District & Palika GeoJSON Boundaries
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !data) return;

    // Remove previous boundary layers if they exist
    if (districtLayerRef.current) {
      map.removeLayer(districtLayerRef.current);
    }
    if (palikasLayerRef.current) {
      map.removeLayer(palikasLayerRef.current);
    }

    // A. District Boundary (Clean crisp outer boundary)
    if (data.district) {
      districtLayerRef.current = L.geoJSON(data.district, {
        style: {
          color: '#334155',
          weight: 2,
          opacity: 0.85,
          fill: false,
          dashArray: '5, 5'
        },
        interactive: false
      }).addTo(map);
    }

    // B. Palikas Choropleth Polygons
    if (data.palikas) {
      palikasLayerRef.current = L.geoJSON(data.palikas, {
        style: (feat) => getPalikaStyle(feat),
        onEachFeature: (feat, layer) => {
          const p = feat.properties;
          
          layer.on({
            mouseover: (e) => {
              const target = e.target;
              target.setStyle(getPalikaStyle(feat, true));
              target.bringToFront();
            },
            mouseout: (e) => {
              const target = e.target;
              target.setStyle(getPalikaStyle(feat, false));
            },
            click: () => {
              if (p.surveyed) {
                onSelectPalika(p.name);
                map.fitBounds(layer.getBounds(), { padding: [30, 30], maxZoom: 13 });
              } else {
                onSelectPalika(p.name);
              }
            }
          });

          // Minimal popup
          const statusHtml = p.surveyed 
            ? `<div style="font-weight:600; color:#0f172a; font-size:14px; margin-bottom:4px;">${p.name}</div>
               <div style="font-size:12px; color:#475569; display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                 <div><strong>Sources:</strong> ${p.total_sources.toLocaleString()}</div>
                 <div><strong>Recorded dried:</strong> <span style="color:${p.drying_rate > 30 ? '#dc2626' : '#2563eb'}">${p.drying_rate}%</span></div>
                 <div><strong>Mean Elev:</strong> ${p.mean_elev} m</div>
                 <div><strong>Road Dist:</strong> ${p.mean_road} m</div>
               </div>`
            : `<div style="font-weight:600; color:#64748b; font-size:14px;">${p.name}</div>
               <div style="font-size:12px; color:#94a3b8; margin-top:2px;">Not covered by the inventory</div>`;

          layer.bindTooltip(statusHtml, {
            sticky: true,
            className: 'palika-map-tooltip'
          });
        }
      }).addTo(map);
    }
  }, [data, selectedPalika]);

  // 4. Ward boundaries (OpenStreetMap), shaded by how many of each recorded ward's sources fall inside the polygon
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !data?.wards) return;
    if (wardsLayerRef.current) {
      map.removeLayer(wardsLayerRef.current);
      wardsLayerRef.current = null;
    }
    if (!showWards) return;

    const shade = (s) => (s === null ? '#e2e8f0' : s < 0.5 ? '#f59e0b' : s < 0.8 ? '#bfdbfe' : '#3b82f6');
    wardsLayerRef.current = L.geoJSON(data.wards, {
      style: (feat) => ({
        color: '#1e3a8a',
        weight: 0.8,
        fillColor: shade(feat.properties.share_inside),
        fillOpacity: 0.35
      }),
      onEachFeature: (feat, layer) => {
        const p = feat.properties;
        const share = p.share_inside === null ? '–' : `${Math.round(p.share_inside * 100)}%`;
        layer.bindTooltip(
          `<div style="font-weight:600; color:#0f172a; font-size:13px;">${p.palika} ward ${p.ward}</div>
           <div style="font-size:12px; color:#475569;">Sources recorded in this ward: ${p.n_recorded}<br/>
           Of those, inside this polygon: <strong>${share}</strong><br/>
           Recorded dried (by recorded ward): ${p.recorded_rate ?? '–'}%</div>`,
          { sticky: true, className: 'palika-map-tooltip' }
        );
      }
    }).addTo(map);
    if (springsLayerRef.current) springsLayerRef.current.bringToFront();
  }, [data, showWards]);

  // 5. Render the sources on the canvas renderer
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !data?.sources) return;

    if (springsLayerRef.current) {
      map.removeLayer(springsLayerRef.current);
    }

    const markers = [];
    const renderer = canvasRendererRef.current;

    const isAllPalikas = !selectedPalika || selectedPalika === 'All Palikas';

    for (const sp of data.sources) {
      const [lat, lon, dried, elev, muni, type, ward] = sp;

      if (!isAllPalikas && muni !== selectedPalika) continue;

      // Status Filter
      if (statusFilter === 'dried' && dried === 0) continue;
      if (statusFilter === 'active' && dried === 1) continue;

      const isDried = dried === 1;
      const marker = L.circleMarker([lat, lon], {
        renderer,
        radius: isDried ? 3.5 : 3.0,
        fillColor: isDried ? '#c92a2a' : '#2b8a3e',
        fillOpacity: isDried ? 0.85 : 0.8,
        color: isDried ? '#7f1d1d' : '#14532d',
        weight: 0.6
      });

      marker.bindPopup(`
        <div style="padding:10px; font-family:'Plus Jakarta Sans',sans-serif; min-width:180px;">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
            <span style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; color:${isDried ? '#b91c1c' : '#15803d'}; background:${isDried ? '#fef2f2' : '#f0fdf4'}; padding:2px 6px; border-radius:4px; border:1px solid ${isDried ? '#fecaca' : '#bbf7d0'};">
              ${isDried ? 'Recorded dried' : 'Recorded active'}
            </span>
            <span style="font-size:11px; color:#64748b; font-family:monospace;">${elev} m</span>
          </div>
          <div style="font-weight:600; font-size:13px; color:#0f172a; margin-bottom:4px;">${muni}, recorded ward ${ward}</div>
          <div style="font-size:12px; color:#475569; margin-bottom:2px;">Type: <span style="color:#0f172a;">${type}</span></div>
          <div style="font-size:11px; color:#94a3b8; font-family:monospace; margin-top:4px;">${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</div>
        </div>
      `, {
        closeButton: true,
        className: 'spring-detail-popup'
      });

      markers.push(marker);
    }

    const featureGroup = L.featureGroup(markers);
    featureGroup.addTo(map);
    springsLayerRef.current = featureGroup;

  }, [data, selectedPalika, statusFilter]);

  return (
    <div className="map-wrapper" style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={mapContainerRef} className="map-container" />

      {/* Cartographic Legend */}
      <div className="map-legend glass-panel">
        <div className="legend-title">Legend</div>

        <div className="legend-section">
          <div className="legend-subtitle">Recorded condition ({data.totalSources.toLocaleString()} springs and ponds)</div>
          <div className="legend-row">
            <span className="legend-symbol active-symbol"></span>
            <span className="legend-label">Active ({(100 - data.dryingRate).toFixed(1)}%)</span>
          </div>
          <div className="legend-row">
            <span className="legend-symbol dried-symbol"></span>
            <span className="legend-label">Dried ({data.dryingRate}%)</span>
          </div>
        </div>

        {showWards && (
          <div className="legend-section">
            <div className="legend-subtitle">Recorded ward sources inside polygon</div>
            <div className="choropleth-gradient">
              <div className="choro-step" style={{ background: '#fcd9a1' }}>&lt;50%</div>
              <div className="choro-step" style={{ background: '#dbeafe' }}>50–80%</div>
              <div className="choro-step" style={{ background: '#93c5fd' }}>&ge;80%</div>
            </div>
          </div>
        )}

        <div className="legend-section">
          <div className="legend-subtitle">Palika share recorded as dried</div>
          <div className="choropleth-gradient">
            <div className="choro-step" style={{ background: '#d1fae5' }}>&lt;15%</div>
            <div className="choro-step" style={{ background: '#fef3c7' }}>15–25%</div>
            <div className="choro-step" style={{ background: '#ffedd5' }}>25–35%</div>
            <div className="choro-step" style={{ background: '#fee2e2' }}>&gt;35%</div>
          </div>
          <div className="legend-row unsurveyed-row">
            <span className="legend-symbol unsurveyed-symbol"></span>
            <span className="legend-label">Not in the inventory (6 palikas)</span>
          </div>
        </div>

        <div className="legend-footer">
          <div>Data: <strong>ICIMOD</strong> (CC BY 4.0)</div>
          <div>Analysis: <a href="https://noesisframeworks.com" target="_blank" rel="noopener noreferrer" style={{ color: '#2563eb', textDecoration: 'none', fontWeight: 600 }}>Noesis Frameworks</a></div>
        </div>
      </div>
    </div>
  );
}

