import { lazy, Suspense, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Clapperboard,
  Compass,
  Database,
  Film,
  Menu,
  Microscope,
  Palette,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api } from "../services/api";
import { Loading } from "../components/States";

const AnalyticsPage = lazy(() => import("../pages/AnalyticsPage").then((module) => ({ default: module.AnalyticsPage })));
const DatasetPage = lazy(() => import("../pages/DatasetPage").then((module) => ({ default: module.DatasetPage })));
const MethodologyPage = lazy(() => import("../pages/MethodologyPage").then((module) => ({ default: module.MethodologyPage })));
const ModelPage = lazy(() => import("../pages/ModelPage").then((module) => ({ default: module.ModelPage })));
const MoviesPage = lazy(() => import("../pages/MoviesPage").then((module) => ({ default: module.MoviesPage })));
const OverviewPage = lazy(() => import("../pages/OverviewPage").then((module) => ({ default: module.OverviewPage })));
const PredictPage = lazy(() => import("../pages/PredictPage").then((module) => ({ default: module.PredictPage })));
const ResearcherPage = lazy(() => import("../pages/ResearcherPage").then((module) => ({ default: module.ResearcherPage })));

const navigation = [
  { to: "/", label: "Overview", icon: Compass, end: true },
  { to: "/dataset", label: "Dataset", icon: Database },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/predict", label: "Prediction studio", icon: SlidersHorizontal },
  { to: "/researcher", label: "Researcher", icon: Sparkles },
  { to: "/models", label: "Model lab", icon: Microscope },
  { to: "/movies", label: "Movie explorer", icon: Film },
  { to: "/methodology", label: "Methodology", icon: Activity },
];

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [theme, setTheme] = useState<"diary" | "original">(() =>
    window.localStorage.getItem("cinescope-theme") === "original" ? "original" : "diary",
  );
  const [apiState, setApiState] = useState<"checking" | "ready" | "offline">("checking");
  const [datasetRows, setDatasetRows] = useState<number | null>(null);
  const location = useLocation();

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    window.localStorage.setItem("cinescope-theme", theme);
  }, [theme]);

  useEffect(() => {
    api.health().then(() => setApiState("ready")).catch(() => setApiState("offline"));
    api.overview().then((summary) => setDatasetRows(summary.raw_rows)).catch(() => setDatasetRows(null));
  }, []);

  return (
    <div className="app-shell" data-theme={theme}>
      <button
        className="mobile-menu icon-button"
        aria-label={menuOpen ? "Close navigation" : "Open navigation"}
        onClick={() => setMenuOpen(!menuOpen)}
      >
        {menuOpen ? <X size={20} /> : <Menu size={20} />}
      </button>
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <NavLink to="/" className="brand">
          <span className="brand-mark"><Clapperboard size={20} /></span>
          <span>CINE<span className="brand-light">SCOPE</span><small>FILM INTELLIGENCE</small></span>
        </NavLink>
        <div className="nav-caption">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              <Icon size={17} strokeWidth={1.7} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="dataset-badge"><span className="badge-dot" /><span>IMDB MOVIE DATASET<br /><small>{datasetRows === null ? "SOURCE CSV" : `${datasetRows.toLocaleString()} RAW RECORDS`}</small></span></div>
          <span className="sidebar-version">CAPSTONE · 2026</span>
        </div>
      </aside>
      {menuOpen && <button aria-label="Close menu overlay" className="menu-scrim" onClick={() => setMenuOpen(false)} />}
      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">CINESCOPE <span>/</span> {navigation.find((item) => item.to === location.pathname)?.label.toUpperCase() ?? "WORKSPACE"}</div>
          <div className="topbar-actions">
            <button
              className="theme-toggle"
              type="button"
              aria-label={theme === "diary" ? "Switch to original CineScope design" : "Switch to film diary design"}
              onClick={() => setTheme((current) => current === "diary" ? "original" : "diary")}
            >
              <Palette size={15} />
              <span>{theme === "diary" ? "Original look" : "Film diary look"}</span>
            </button>
            <div className={`connection-state ${apiState}`}><span />{apiState === "ready" ? "DATA API CONNECTED" : apiState === "offline" ? "API OFFLINE" : "CONNECTING"}</div>
          </div>
        </header>
        <div className="page-content">
          <Suspense fallback={<Loading />}>
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/dataset" element={<DatasetPage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/researcher" element={<ResearcherPage />} />
              <Route path="/models" element={<ModelPage />} />
              <Route path="/movies" element={<MoviesPage />} />
              <Route path="/methodology" element={<MethodologyPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  );
}
