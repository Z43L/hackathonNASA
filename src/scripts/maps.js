
import L from "leaflet";


// =====================
// 1) ESTADO Y CONSTANTES
// =====================

let map; // instancia Leaflet
let currentLayers = {}; // nombre -> { url, zoom, layer, active, ... }
let markers = [];
let isFullscreen = false;

const mapEl = document.getElementById('map');
const planet = (mapEl?.dataset?.planet || 'mars').toLowerCase();

// Cache de configuraciones de capas (por planeta) para evitar refetch
const layerCache = new Map();

// Ubicaciones predefinidas (para búsqueda textual)
const locations = {
  madrid: [40.4168, -3.7038],
  barcelona: [41.3851, 2.1734],
  sevilla: [37.3886, -5.9823],
  valencia: [39.4699, -0.3763],
  bilbao: [43.263, -2.934],
  zaragoza: [41.6488, -0.8891],
  'málaga': [36.7213, -4.4214],
  murcia: [37.9922, -1.1307],
  palma: [39.5696, 2.6502],
  'las palmas': [28.1235, -15.4363]
};

// Respuestas simuladas de IA
const aiResponses = {
  costa: { coords: [41.3851, 2.1734], name: 'Costa Mediterránea' },
  playa: { coords: [41.3851, 2.1734], name: 'Costa Mediterránea' },
  montaña: { coords: [42.6026, 0.6396], name: 'Pirineos' },
  pirineos: { coords: [42.6026, 0.6396], name: 'Pirineos' },
  desierto: { coords: [28.2916, -16.6291], name: 'Islas Canarias' },
  canarias: { coords: [28.2916, -16.6291], name: 'Islas Canarias' },
  bosque: { coords: [43.1828, -2.6735], name: 'Bosques del Norte' },
  verde: { coords: [43.1828, -2.6735], name: 'Bosques del Norte' },
  ciudad: { coords: [40.4168, -3.7038], name: 'Madrid' },
  urbano: { coords: [40.4168, -3.7038], name: 'Madrid' },
  'río': { coords: [41.3851, 2.1734], name: 'Delta del Ebro' },
  agua: { coords: [41.3851, 2.1734], name: 'Delta del Ebro' }
};

// =====================
// 2) CARGA DE CONFIGURACIONES
// =====================
async function loadLayerConfigs(planetName) {
  if (layerCache.has(planetName)) return layerCache.get(planetName);

  const url = `/${planetName}.json`; // desde public/
  const res = await fetch(url, { cache: 'no-cache' });

  if (!res.ok) {
    console.warn(`No se pudo cargar ${url}. Fallback a Mars.`);
    const fallback = await fetch('/mars.json');
    const data = await fallback.json();
    layerCache.set('mars', data);
    return data;
  }

  const data = await res.json();
  layerCache.set(planetName, data);
  return data;
}

// Carga inicial (top-level await permitido en módulos, deja tal cual)
const layerConfigs = await loadLayerConfigs(planet);

// =====================
// 3) UTILIDADES DOM / UI
// =====================
const $ = (id) => document.getElementById(id);

function showLoading(show) {
  const el = $('loadingIndicator');
  if (el) el.style.display = show ? 'flex' : 'none';
}

function showNotification(message, type = 'info') {
  // limpiar existentes
  document.querySelectorAll('.notification').forEach((n) => n.remove());

  const colors = {
    success: 'rgba(34, 197, 94, 0.9)',
    error: 'rgba(239, 68, 68, 0.9)',
    info: 'rgba(59, 130, 246, 0.9)',
    warning: 'rgba(245, 158, 11, 0.9)'
  };

  const notification = document.createElement('div');
  notification.className = 'notification';
  notification.textContent = message;
  notification.style.cssText = `
    position: fixed; top: 20px; right: 20px; z-index: 10000;
    background: ${colors[type] || colors.info}; color: #fff;
    padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 500;
    transform: translateX(100%); transition: transform .3s ease; max-width: 300px;
    box-shadow: 0 4px 12px rgba(0,0,0,.15);
  `;

  document.body.appendChild(notification);
  setTimeout(() => (notification.style.transform = 'translateX(0)'), 100);
  setTimeout(() => {
    notification.style.transform = 'translateX(100%)';
    setTimeout(() => notification.remove(), 300);
  }, 4000);
}

// Panel flotante (info + filtros)
function showInfoPanel(content) {
  const imageInfoEl = $('imageInfo');
  const infoPanelEl = $('infoPanel');
  if (!imageInfoEl || !infoPanelEl) return;

  if (!content) {
    // Panel de filtros por defecto
    content = `
<div class="filter-group" style="margin-bottom: 15px;">
  <label style="display:block;margin-bottom:5px;font-weight:bold;color:#fff;">
    <i class="fas fa-rainbow"></i> Hue: <span id="hue-value" style="color:var(--accent-cyan);background:rgba(0,188,212,.1);padding:2px 6px;border-radius:4px;">0°</span>
  </label>
  <input type="range" id="hue-slider" min="0" max="360" value="0" step="1" style="width:100%;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none;-webkit-appearance:none;">
</div>
<div class="filter-group" style="margin-bottom: 15px;">
  <label style="display:block;margin-bottom:5px;font-weight:bold;color:#fff;">
    <i class="fas fa-adjust"></i> Contrast: <span id="contrast-value" style="color:var(--accent-cyan);background:rgba(0,188,212,.1);padding:2px 6px;border-radius:4px;">100%</span>
  </label>
  <input type="range" id="contrast-slider" min="0" max="200" value="100" step="1" style="width:100%;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none;-webkit-appearance:none;">
</div>
<div class="filter-group" style="margin-bottom: 15px;">
  <label style="display:block;margin-bottom:5px;font-weight:bold;color:#fff;">
    <i class="fas fa-sun"></i> Brightness: <span id="brightness-value" style="color:var(--accent-cyan);background:rgba(0,188,212,.1);padding:2px 6px;border-radius:4px;">100%</span>
  </label>
  <input type="range" id="brightness-slider" min="0" max="200" value="100" step="1" style="width:100%;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none;-webkit-appearance:none;">
</div>
<div class="filter-group" style="margin-bottom: 15px;">
  <label style="display:block;margin-bottom:5px;font-weight:bold;color:#fff;">
    <i class="fas fa-tint"></i> Saturation: <span id="saturate-value" style="color:var(--accent-cyan);background:rgba(0,188,212,.1);padding:2px 6px;border-radius:4px;">100%</span>
  </label>
  <input type="range" id="saturate-slider" min="0" max="200" value="100" step="1" style="width:100%;height:6px;border-radius:3px;background:rgba(255,255,255,.1);outline:none;-webkit-appearance:none;">
</div>
<div style="display:flex;gap:10px;margin-top:20px;">
  <button onclick="resetFilters()" style="flex:1;padding:10px;border-radius:6px;background:linear-gradient(135deg,#ff6b6b,#ff5252);color:#fff;border:none;cursor:pointer;font-weight:600;transition:all .2s;">
    <i class="fas fa-undo"></i> Reset
  </button>
</div>`;
  }

  imageInfoEl.innerHTML = content;
  infoPanelEl.style.display = 'block';

  if (content.includes('hue-slider')) setTimeout(setupFilterListeners, 100);
}

function closeInfoPanel() {
  const infoPanelEl = $('infoPanel');
  if (infoPanelEl) infoPanelEl.style.display = 'none';
}

// =====================
// 4) PLUGIN: TILE COLOR FILTER
// =====================
function initColorFilterPlugin() {
  L.TileLayer.ColorFilter = L.TileLayer.extend({
    initialize: function (url, options) {
      L.TileLayer.prototype.initialize.call(this, url, options);
      this._filters = options.filter || [];
    },
    onAdd: function (map) {
      L.TileLayer.prototype.onAdd.call(this, map);
      this._applyContainerFilter();
    },
    updateFilter: function (newFilters) {
      this._filters = newFilters || [];
      this._applyContainerFilter();
    },
    _applyContainerFilter: function () {
      if (!this._container) return;
      const filterString = this._filters.join(' ');
      this._container.style.filter = filterString;
      const tiles = this._container.querySelectorAll('img');
      for (let i = 0; i < tiles.length; i++) tiles[i].style.filter = filterString;
      console.log('Filtros aplicados a', this.options.attribution, ':', filterString);
    }
  });

  L.tileLayer.colorFilter = function (url, options) {
    return new L.TileLayer.ColorFilter(url, options);
  };
}

// =====================
// 5) MAPA: INICIALIZACIÓN Y CAPAS
// =====================
function initMap() {
  map = L.map('map', {
    center: [0, 180],
    zoom: 1,
    zoomControl: false,
    preferCanvas: true,
    attributionControl: false
  });

  // Control de zoom personalizado
  L.control.zoom({ position: 'bottomright' }).addTo(map);

  // Inicializar capas con ColorFilter
  Object.keys(layerConfigs).forEach((key) => {
    const config = layerConfigs[key];

    config.layer = L.tileLayer.colorFilter(config.url, {
      attribution: config.attribution || '',
      minZoom: 1,
      maxZoom: 7, // max zoom global del mapa
      maxNativeZoom: config.zoom, // max zoom real de la capa
      noWrap: true,
      crossOrigin: true,
      filter: [
        'hue-rotate(0deg)',
        'contrast(100%)',
        'brightness(100%)',
        'saturate(100%)'
      ]
    });

    currentLayers[key] = config;
    if (config.active) config.layer.addTo(map);
  });

  setupMapEventListeners();
  updateActiveLayersCount();
  updateMapInfo();
  showNotification('Mapa satelital inicializado correctamente', 'success');
}

function toggleLayer(layerName) {
  const layerItem = document.querySelector(`[data-layer="${layerName}"]`);
  const layerConfig = currentLayers[layerName];
  if (!layerConfig || !layerItem) return;

  if (layerConfig.active) {
    map.removeLayer(layerConfig.layer);
  } else {
    layerConfig.layer.addTo(map);
  }
  layerConfig.active = !layerConfig.active;
  layerItem.classList.toggle('active', !!layerConfig.active);
  updateActiveLayersCount();
}

function setLayerOpacity(layerName, opacity) {
  const layerConfig = currentLayers[layerName];
  if (layerConfig?.active && layerConfig.layer) {
    layerConfig.layer.setOpacity(parseFloat(opacity));
  }
}

// =====================
// 6) INTERACCIÓN CON EL MAPA
// =====================
function setupMapEventListeners() {
  // Coordenadas bajo el ratón
  map.on('mousemove', (e) => {
    const { lat, lng } = e.latlng;
    const latEl = $('currentLat');
    const lngEl = $('currentLng');
    if (latEl) latEl.textContent = lat.toFixed(3);
    if (lngEl) lngEl.textContent = lng.toFixed(3);
  });

  // Zoom actual
  map.on('zoomend', () => {
    const zoomEl = $('currentZoom');
    if (zoomEl) zoomEl.textContent = map.getZoom();
  });

  // Info al mover
  map.on('moveend', updateMapInfo);

  // Click derecho: limpiar marcadores
  map.on('contextmenu', (e) => {
    e.originalEvent.preventDefault();
    clearMarkers();
  });

  // Ctrl + click: añadir marcador
  map.on('click', (e) => {
    if (e.originalEvent.ctrlKey) {
      addMarker(e.latlng, `Coordenadas: ${e.latlng.lat.toFixed(4)}, ${e.latlng.lng.toFixed(4)}`);
    }
  });
}

function addMarker(coords, popupText) {
  const marker = L.marker(coords).addTo(map).bindPopup(popupText).openPopup();
  markers.push(marker);
  return marker;
}

function clearMarkers() {
  if (!markers.length) return;
  markers.forEach((m) => map.removeLayer(m));
  markers = [];
  showNotification('Marcadores eliminados', 'info');
}

// =====================
// 7) BÚSQUEDAS (texto, coords, IA)
// =====================
function goToCoordinates() {
  const latInput = $('latInput');
  const lngInput = $('lngInput');
  if (!latInput || !lngInput) return;

  const lat = parseFloat(latInput.value);
  const lng = parseFloat(lngInput.value);

  if (Number.isNaN(lat) || Number.isNaN(lng)) {
    showNotification('Por favor, introduce coordenadas válidas', 'error');
    return;
  }
  if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
    showNotification('Coordenadas fuera de rango válido', 'error');
    return;
  }

  map.setView([lat, lng], 12);
  addMarker([lat, lng], `Coordenadas: ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
  showNotification(`Navegando a: ${lat.toFixed(4)}, ${lng.toFixed(4)}`);
}

function performSearch() {
  const searchInput = $('mainSearch');
  if (!searchInput) return;

  const searchTerm = searchInput.value.trim();
  if (!searchTerm) {
    showNotification('Introduce un término de búsqueda', 'error');
    return;
  }

  showLoading(true);
  setTimeout(() => {
    showLoading(false);
    const searchLower = searchTerm.toLowerCase();
    let found = false;

    // Coincidencia con ciudades
    for (const city of Object.keys(locations)) {
      if (searchLower.includes(city)) {
        const coords = locations[city];
        map.setView(coords, 10);
        const cityName = city.charAt(0).toUpperCase() + city.slice(1);
        addMarker(coords, cityName);
        showNotification(`Encontrado: ${cityName}`, 'success');
        found = true;
        break;
      }
    }

    // Intentar como coordenadas
    if (!found) {
      const coordMatch = searchTerm.match(/-?\d+\.?\d*,-?\s*-?\d+\.?\d*/);
      if (coordMatch) {
        const [lat, lng] = coordMatch[0].split(',').map((c) => parseFloat(c.trim()));
        if (!Number.isNaN(lat) && !Number.isNaN(lng)) {
          map.setView([lat, lng], 10);
          addMarker([lat, lng], `Coordenadas: ${lat}, ${lng}`);
          showNotification('Coordenadas encontradas', 'success');
          found = true;
        }
      }
    }

    if (!found) {
      showNotification('Ubicación no encontrada. Prueba con ciudades españolas o coordenadas', 'error');
    }
  }, Math.random() * 1000 + 500);
}

function performAISearch() {
  const aiInput = $('aiSearch');
  if (!aiInput) return;

  const aiQuery = aiInput.value.trim();
  if (!aiQuery) {
    showNotification('Introduce una consulta para la IA', 'error');
    return;
  }

  showLoading(true);
  setTimeout(() => {
    showLoading(false);
    const queryLower = aiQuery.toLowerCase();
    let found = false;

    for (const keyword of Object.keys(aiResponses)) {
      if (queryLower.includes(keyword)) {
        const { coords, name } = aiResponses[keyword];
        map.setView(coords, 8);
        addMarker(coords, `IA encontró: ${name}`);
        showNotification(`IA encontró: ${name}`, 'success');
        found = true;
        break;
      }
    }

    if (!found) {
      const suggestions = Object.keys(aiResponses).slice(0, 3).join(', ');
      showNotification(`IA: Intenta buscar "${suggestions}"...`, 'info');
    }
  }, Math.random() * 2000 + 1000);
}

// =====================
// 8) FILTROS DE COLOR
// =====================
function setupFilterListeners() {
  const hue = $('hue-slider');
  const contrast = $('contrast-slider');
  const brightness = $('brightness-slider');
  const saturate = $('saturate-slider');

  hue?.addEventListener('input', updateFilters);
  contrast?.addEventListener('input', updateFilters);
  brightness?.addEventListener('input', updateFilters);
  saturate?.addEventListener('input', updateFilters);

  [hue, contrast, brightness, saturate].forEach((s) => {
    if (s) s.style.cssText += '-webkit-appearance:none;appearance:none;';
  });
}

function updateFilters() {
  const hue = $('hue-slider')?.value || 0;
  const contrast = $('contrast-slider')?.value || 100;
  const brightness = $('brightness-slider')?.value || 100;
  const saturate = $('saturate-slider')?.value || 100;

  const hueValue = $('hue-value');
  const contrastValue = $('contrast-value');
  const brightnessValue = $('brightness-value');
  const saturateValue = $('saturate-value');

  if (hueValue) hueValue.textContent = hue + '°';
  if (contrastValue) contrastValue.textContent = contrast + '%';
  if (brightnessValue) brightnessValue.textContent = brightness + '%';
  if (saturateValue) saturateValue.textContent = saturate + '%';

  const newFilters = [
    `hue-rotate(${hue}deg)`,
    `contrast(${contrast}%)`,
    `brightness(${brightness}%)`,
    `saturate(${saturate}%)`
  ];

  Object.values(currentLayers).forEach((cfg) => {
    if (cfg.active && cfg.layer?.updateFilter) cfg.layer.updateFilter(newFilters);
  });
}

function resetFilters() {
  const hue = $('hue-slider');
  const contrast = $('contrast-slider');
  const brightness = $('brightness-slider');
  const saturate = $('saturate-slider');

  if (hue) hue.value = 0;
  if (contrast) contrast.value = 100;
  if (brightness) brightness.value = 100;
  if (saturate) saturate.value = 100;

  updateFilters();
  showNotification('🔄 Filtros restablecidos', 'info');
}

function applyPreset(preset) {
  const presets = {
    normal: [0, 100, 100, 100],
    sepia: [30, 120, 110, 80],
    vintage: [25, 130, 90, 70],
    mars: [15, 140, 120, 130],
    cyberpunk: [280, 150, 120, 150],
    infrared: [180, 160, 90, 200]
  };

  if (!presets[preset]) return;
  const [h, c, b, s] = presets[preset];

  const hue = $('hue-slider');
  const contrast = $('contrast-slider');
  const brightness = $('brightness-slider');
  const saturate = $('saturate-slider');

  if (hue) hue.value = h;
  if (contrast) contrast.value = c;
  if (brightness) brightness.value = b;
  if (saturate) saturate.value = s;

  updateFilters();

  const presetNames = {
    normal: 'Normal',
    sepia: 'Sepia',
    vintage: 'Vintage',
    mars: 'Marte',
    cyberpunk: 'Cyberpunk',
    infrared: 'Infrarrojo'
  };
  showNotification(`🎨 Preset aplicado: ${presetNames[preset]}`, 'success');
}

function openColorFilters() {
  showInfoPanel();
}

// =====================
// 9) PANELES DE AYUDA / INFO
// =====================
function showHelp() {
  const helpContent = `
    <h3>🛰️ NASA Satellite Explorer - Guía de Uso</h3>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">🗺️ Navegación</h4>
      <ul style="margin-left:20px;line-height:1.6;">
        <li>Arrastra para mover el mapa</li>
        <li>Rueda del ratón para zoom</li>
        <li>Ctrl + Click para añadir marcador</li>
        <li>Click derecho para limpiar marcadores</li>
      </ul>
    </div>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">🌍 Capas del Mapa</h4>
      <ul style="margin-left:20px;line-height:1.6;">
        <li>Click en una capa para activar/desactivar</li>
        <li>Usa los sliders para ajustar opacidad</li>
        <li>Combina múltiples capas para mejor análisis</li>
      </ul>
    </div>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">🎨 Filtros de Color</h4>
      <ul style="margin-left:20px;line-height:1.6;">
        <li>Usa el botón 🎨 para abrir los filtros</li>
        <li>Ajusta matiz, contraste, brillo y saturación</li>
        <li>Aplica presets rápidos para diferentes efectos</li>
        <li>Usa filtros aleatorios para explorar</li>
      </ul>
    </div>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">🔍 Búsquedas</h4>
      <ul style="margin-left:20px;line-height:1.6;">
        <li>Ciudades: Madrid, Barcelona, Sevilla...</li>
        <li>Coordenadas: 40.4168, -3.7038</li>
        <li>IA: "costa mediterránea", "montañas"</li>
      </ul>
    </div>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">⌨️ Atajos de Teclado</h4>
      <ul style="margin-left:20px;line-height:1.6;">
        <li>Ctrl + F: Enfocar búsqueda</li>
        <li>Ctrl + H: Mostrar ayuda</li>
        <li>Ctrl + C: Abrir filtros de color</li>
        <li>Escape: Cerrar paneles</li>
      </ul>
    </div>`;

  showInfoPanel(helpContent);
}

function showLayerInfo() {
  const activeCount = Object.values(currentLayers).filter((l) => l.active).length;
  const center = map.getCenter();

  const names = {
    marsViking: 'Mars Viking MDIM21',
    marsMOLA: 'Mars MOLA Blend',
    marsGlobal: 'Mars Global Surveyor'
  };

  const infoContent = `
    <h3>📊 Estado del Mapa</h3>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">🗂️ Capas Activas (${activeCount})</h4>
      <div style="margin-left:10px;">
        ${Object.keys(currentLayers)
          .map((layerName) => {
            const layer = currentLayers[layerName];
            const status = layer.active ? '🟢 Activa' : '🔴 Inactiva';
            return `<p><strong>${names[layerName] || layerName}:</strong> ${status}</p>`;
          })
          .join('')}
      </div>
    </div>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">📍 Posición Actual</h4>
      <div style="margin-left:10px;font-family:monospace;background:rgba(59,130,246,.1);padding:10px;border-radius:5px;">
        <p><strong>Centro:</strong> ${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}</p>
        <p><strong>Zoom:</strong> ${map.getZoom()}</p>
        <p><strong>Marcadores:</strong> ${markers.length}</p>
      </div>
    </div>
    <div style="margin:15px 0;">
      <h4 style="color:#3b82f6;margin-bottom:8px;">ℹ️ Información Técnica</h4>
      <div style="margin-left:10px;font-size:12px;color:#94a3b8;">
        <p>Sistema de coordenadas: WGS84</p>
        <p>Proyección: Web Mercator (EPSG:3857)</p>
        <p>Última actualización: ${new Date().toLocaleString()}</p>
      </div>
    </div>`;

  showInfoPanel(infoContent);
}

// =====================
// 10) OTRAS HERRAMIENTAS / ACCIONES
// =====================
function toggleFullscreen() {
  if (!isFullscreen) {
    const element = document.documentElement;
    element.requestFullscreen?.() || element.webkitRequestFullscreen?.() || element.msRequestFullscreen?.();
    isFullscreen = true;
    showNotification('Modo pantalla completa activado');
  } else {
    document.exitFullscreen?.() || document.webkitExitFullscreen?.() || document.msExitFullscreen?.();
    isFullscreen = false;
    showNotification('Modo pantalla completa desactivado');
  }
}

function toggleMeasurement() {
  showNotification('🔧 Herramienta de medición: En desarrollo. Próximamente permitirá medir distancias y áreas.', 'info');
}

function exportView() {
  showNotification('📥 Exportando vista actual...', 'info');
  showLoading(true);

  setTimeout(() => {
    showLoading(false);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 300;
    canvas.height = 200;

    const gradient = ctx.createLinearGradient(0, 0, 300, 200);
    gradient.addColorStop(0, '#0c1220');
    gradient.addColorStop(1, '#1a2332');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 300, 200);

    ctx.fillStyle = '#3b82f6';
    ctx.font = '16px Arial';
    ctx.fillText('NASA Satellite Explorer', 10, 30);
    ctx.font = '12px Arial';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(`Centro: ${map.getCenter().lat.toFixed(2)}, ${map.getCenter().lng.toFixed(2)}`, 10, 50);
    ctx.fillText(`Zoom: ${map.getZoom()}`, 10, 70);
    ctx.fillText(`Fecha: ${new Date().toLocaleDateString()}`, 10, 90);

    const link = document.createElement('a');
    link.download = `satellite-view-${Date.now()}.png`;
    link.href = canvas.toDataURL();
    link.click();

    showNotification('✅ Vista exportada correctamente', 'success');
  }, 2000);
}

function toggle3D() {
  showNotification('🎮 Vista 3D: Requiere datos de elevación adicionales y WebGL. Funcionalidad en desarrollo.', 'info');
}

function updateActiveLayersCount() {
  const count = Object.values(currentLayers).filter((l) => l.active).length;
  const activeLayersEl = $('activeLayers');
  if (activeLayersEl) activeLayersEl.textContent = count;
}

function updateMapInfo() {
  const center = map.getCenter();
  $('currentLat') && ($('currentLat').textContent = center.lat.toFixed(3));
  $('currentLng') && ($('currentLng').textContent = center.lng.toFixed(3));
  $('currentZoom') && ($('currentZoom').textContent = map.getZoom());
}

function updateTimeDisplay(value) {
  const day = parseInt(value, 10) + 1;
  const el = $('timeDisplay');
  if (el) el.textContent = `Día del año: ${day}`;

  if (day === 1) showNotification('🗓️ Mostrando datos del 1 de enero', 'info');
  else if (day === 365) showNotification('🗓️ Mostrando datos del 31 de diciembre', 'info');
}

// =====================
// 11) LISTENERS GLOBALES / TECLADO
// =====================
function setupEventListeners() {
  // Botones principales
  $('searchBtn')?.addEventListener('click', performSearch);
  $('fullscreenBtn')?.addEventListener('click', toggleFullscreen);
  $('helpBtn')?.addEventListener('click', showHelp);
  $('goToCoords')?.addEventListener('click', goToCoordinates);
  $('aiSearchBtn')?.addEventListener('click', performAISearch);
  $('closePanelBtn')?.addEventListener('click', closeInfoPanel);

  // Overlay
  $('measureBtn')?.addEventListener('click', toggleMeasurement);
  $('exportBtn')?.addEventListener('click', exportView);
  $('infoBtn')?.addEventListener('click', showLayerInfo);
  $('view3dBtn')?.addEventListener('click', toggle3D);
  $('colorFiltersBtn')?.addEventListener('click', openColorFilters);

  // Delegación: click capa
  document.addEventListener('click', (e) => {
    const layerItem = e.target.closest('.layer-item');
    if (layerItem && !e.target.classList.contains('opacity-slider')) {
      const layerName = layerItem.dataset.layer;
      if (layerName) toggleLayer(layerName);
    }
  });

  // Sliders de opacidad
  document.addEventListener('input', (e) => {
    if (e.target.classList.contains('opacity-slider')) {
      const layerName = e.target.closest('.layer-item')?.dataset.layer;
      if (layerName) setLayerOpacity(layerName, e.target.value);
    }
  });

  // Slider temporal
  $('timeSlider')?.addEventListener('input', (e) => updateTimeDisplay(e.target.value));

  // Enter en búsqueda
  $('mainSearch')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      performSearch();
    }
  });

  // Enter en coords
  ['latInput', 'lngInput'].forEach((id) => {
    $(id)?.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        Coordinates();
      }
    });
  });

  // Atajos
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey) {
      switch (e.key.toLowerCase()) {
        case 'f':
          e.preventDefault();
          $('mainSearch')?.focus();
          break;
        case 'h':
          e.preventDefault();
          showHelp();
          break;
        case 'c':
          e.preventDefault();
          openColorFilters();
          break;
      }
    }
    if (e.key === 'Escape') closeInfoPanel();
  });

  // Pantalla completa
  document.addEventListener('fullscreenchange', () => {
    isFullscreen = !!(document.fullscreenElement || document.webkitFullscreenElement || document.msFullscreenElement);
    if (map) setTimeout(() => map.invalidateSize(), 200);
  });

  // Conectividad
  window.addEventListener('online', () => showNotification('🌐 Conexión restablecida', 'success'));
  window.addEventListener('offline', () => showNotification('📡 Sin conexión. Funcionalidad limitada.', 'warning'));
}

// =====================
// 12) INICIALIZACIÓN APP
// =====================
function initApp() {
  if (typeof L === 'undefined') {
    setTimeout(initApp, 100);
    return;
  }
  try {
    initColorFilterPlugin();
    initMap();
    setupEventListeners();

    // Fecha actual
    const today = new Date();
    const dateInput = $('dateInput');
    if (dateInput) dateInput.value = today.toISOString().split('T')[0];

    // Año en footer
    const lastUpdateEl = $('lastUpdate');
    if (lastUpdateEl) lastUpdateEl.textContent = today.getFullYear();

    console.log('🛰️ NASA Satellite Explorer con filtros de color inicializado correctamente');
  } catch (error) {
    console.error('Error inicializando la aplicación:', error);
    showNotification('❌ Error al inicializar el mapa. Recarga la página.', 'error');
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// =====================
// 13) API GLOBAL (compatibilidad con HTML onclick)
// =====================
window.toggleLayer = toggleLayer;
window.setLayerOpacity = setLayerOpacity;
window.goToCoordinates = goToCoordinates;
window.performSearch = performSearch;
window.performAISearch = performAISearch;
window.showHelp = showHelp;
window.toggleFullscreen = toggleFullscreen;
window.updateTimeDisplay = updateTimeDisplay;
window.exportView = exportView;
window.showLayerInfo = showLayerInfo;
window.toggle3D = toggle3D;
window.closeInfoPanel = closeInfoPanel;
window.toggleMeasurement = toggleMeasurement;
window.showInfoPanel = showInfoPanel;
window.openColorFilters = openColorFilters;
window.resetFilters = resetFilters;
window.applyPreset = applyPreset;
