import { Search, ChevronLeft, ChevronRight, Eye, X, SlidersHorizontal, Download } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { DatasetColumn, DatasetResponse } from "../types";
import { ErrorState, Loading, PageHeader, Panel } from "../components/States";

export function DatasetPage() {
  const [columns, setColumns] = useState<DatasetColumn[]>([]);
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [summary, setSummary] = useState<{ raw_rows: number; raw_columns: number; filename: string } | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState("movie_title");
  const [filters, setFilters] = useState({ genre: "", country: "", language: "", content_rating: "", color: "", min_score: "", max_score: "", min_year: "", max_year: "" });
  const [selected, setSelected] = useState("");
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [showColumns, setShowColumns] = useState(false);
  const [visibleColumns, setVisibleColumns] = useState(["movie_title", "director_name", "genres", "country", "language", "title_year", "budget", "imdb_score"]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const loadSchema = () => Promise.all([api.schema(), api.overview()]).then(([result, overview]) => {
    setColumns(result.columns);
    setSummary({ raw_rows: overview.raw_rows, raw_columns: overview.raw_columns, filename: overview.filename });
  }).catch((reason: Error) => setError(reason.message));
  const loadRows = () => {
    setLoading(true);
    api.rows({ page, page_size: 20, search, sort_by: sortBy, ...filters }).then(setData).catch((reason: Error) => setError(reason.message)).finally(() => setLoading(false));
  };
  useEffect(() => { void loadSchema(); }, []);
  useEffect(() => { const timer = window.setTimeout(loadRows, 220); return () => window.clearTimeout(timer); }, [page, search, sortBy, filters]);
  useEffect(() => { if (selected) api.columnStats(selected).then(setStats).catch((reason: Error) => setError(reason.message)); else setStats(null); }, [selected]);
  if (error) return <ErrorState error={error} retry={() => { setError(""); void loadSchema(); loadRows(); }} />;
  return <>
    <PageHeader eyebrow="SOURCE RECORDS / 02" title="The dataset, in detail." description="Browse the real CSV, inspect its fields, and see data-quality issues without moving the full dataset into the browser." />
    <div className="dataset-overview-strip"><div><span>FILE</span><b>{summary?.filename ?? "Loading…"}</b></div><div><span>RAW ROWS</span><b>{summary?.raw_rows.toLocaleString() ?? "Loading…"}</b></div><div><span>COLUMNS</span><b>{summary?.raw_columns ?? "Loading…"}</b></div><div><span>RAW/CLEAN DISTINCTION</span><b>Explicit</b></div></div>
    <Panel title="Movie records" eyebrow="RAW DATA / PAGINATED">
      <div className="table-controls"><label className="search-field"><Search size={16} /><input value={search} onChange={(event) => { setSearch(event.target.value); setPage(1); }} placeholder="Search title, director, genre, country…" /></label><label className="select-field">Sort by <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>{["movie_title", "imdb_score", "title_year", "budget"].map((column) => <option key={column} value={column}>{column.replaceAll("_", " ")}</option>)}</select></label><label className="select-field"><Eye size={14} /> Column info <select value={selected} onChange={(event) => setSelected(event.target.value)}><option value="">Choose field</option>{columns.map((column) => <option key={column.name}>{column.name}</option>)}</select></label><button className="button button-quiet" onClick={() => setShowColumns(!showColumns)}><SlidersHorizontal size={14} /> Columns</button><button className="button button-quiet" onClick={() => void api.downloadDataset().catch((reason: Error) => setError(reason.message))}><Download size={14} /> Export CSV</button></div>
      <div className="filter-grid">{(["genre", "country", "language", "content_rating", "color", "min_year", "max_year", "min_score", "max_score"] as const).map((field) => { const yearField = field.endsWith("_year"); const scoreField = field.endsWith("_score"); return <label key={field}>{field.replaceAll("_", " ")}<input type={yearField || scoreField ? "number" : "search"} min={yearField ? 1800 : scoreField ? 0 : undefined} max={yearField ? 2200 : scoreField ? 10 : undefined} step={scoreField ? 0.1 : undefined} value={filters[field]} placeholder="Any" onChange={(event) => { setFilters((current) => ({ ...current, [field]: event.target.value })); setPage(1); }} /></label>; })}</div>
      {showColumns && <div className="column-toggles">{columns.map((column) => <label key={column.name}><input type="checkbox" checked={visibleColumns.includes(column.name)} onChange={(event) => setVisibleColumns((current) => event.target.checked ? [...current, column.name] : current.filter((name) => name !== column.name))} />{column.name.replaceAll("_", " ")}</label>)}</div>}
      {stats && <div className="column-detail"><div><strong>{String(stats.column)}</strong><span>{String(stats.dtype)} · {String(stats.unique)} unique · {String(stats.missing)} missing</span></div>{"mean" in stats && <span>Mean {Number(stats.mean).toLocaleString(undefined, { maximumFractionDigits: 2 })} · median {Number(stats.median).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>}{"top_values" in stats && <span>Common values: {(stats.top_values as Array<{value: string}>).slice(0, 5).map((item) => item.value).join(", ")}</span>}<button aria-label="Clear selected column" className="icon-button" onClick={() => setSelected("")}><X size={15} /></button></div>}
      {loading && !data ? <Loading /> : <div className="table-scroll"><table><thead><tr>{visibleColumns.map((column) => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr></thead><tbody>{data?.items.map((row) => <tr key={String(row.row_id)}>{visibleColumns.map((column) => <td key={column}>{row[column] === null || row[column] === undefined ? <span className="null-value">—</span> : column === "budget" ? `$${Number(row[column]).toLocaleString()}` : String(row[column])}</td>)}</tr>)}</tbody></table>{loading && <div className="table-loading">Updating results…</div>}{data?.items.length === 0 && <div className="empty-state">No records match this search.</div>}</div>}
      <div className="pagination"><span>{data ? `${data.total.toLocaleString()} records · page ${data.page} of ${Math.max(1, data.pages)}` : "Loading records"}</span><div><button className="icon-button" aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}><ChevronLeft size={17} /></button><button className="icon-button" aria-label="Next page" disabled={!data || page >= data.pages} onClick={() => setPage((value) => value + 1)}><ChevronRight size={17} /></button></div></div>
    </Panel>
    <Panel title="Column profile" eyebrow="DATA QUALITY"><div className="quality-grid">{columns.slice(0, 14).map((column) => <div className="quality-row" key={column.name}><div><b>{column.name}</b><small>{column.dtype} · {column.unique.toLocaleString()} unique</small></div><div className="quality-meter"><span style={{ width: `${column.missing_percent}%` }} /></div><small>{column.missing_percent}% missing</small></div>)}</div><p className="fine-print">Missing-value bars show the percentage of raw records missing each field. Feature imputations are applied after the train/test split.</p></Panel>
  </>;
}
