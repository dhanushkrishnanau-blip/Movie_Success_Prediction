import { ArrowDownRight, ArrowRight, ArrowUpRight, Database, Film, Layers3, TriangleAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import type { Overview } from "../types";
import { ErrorState, Loading, Panel, PageHeader } from "../components/States";

const colors: Record<string, string> = { Flop: "#c86c56", Average: "#c8a05b", Hit: "#829779" };

export function OverviewPage() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  const load = () => api.overview().then(setData).catch((reason: Error) => setError(reason.message));
  useEffect(() => { void load(); }, []);
  if (error) return <ErrorState error={error} retry={() => { setError(""); void load(); }} />;
  if (!data) return <Loading />;
  const rf = data.model_comparison.find((model) => model.model === "Random Forest");
  const flopRecall = data.model_experiment?.flop_recall_percent;
  const stats = [
    { label: "Movies in source", value: data.raw_rows.toLocaleString(), detail: `${data.raw_columns} recorded fields`, icon: Film },
    { label: "Clean modeling rows", value: data.clean_rows.toLocaleString(), detail: "After notebook cleaning", icon: Database },
    { label: "IMDb categories", value: "03", detail: `${data.unique_genres} distinct genres`, icon: Layers3 },
    { label: "Missing source values", value: data.missing_values.toLocaleString(), detail: `${data.duplicate_rows} duplicate rows`, icon: TriangleAlert },
  ];
  return <>
    <PageHeader eyebrow="MOVIE INTELLIGENCE / 01" title="Stories, measured." description="A data-led view of movie attributes and IMDb-score-derived success categories." />
    <section className="hero-panel">
      <div className="hero-copy"><div className="eyebrow">THE CINESCOPE BRIEFING</div><h2>Look beyond the<br /><em>opening frame.</em></h2><p>Explore the IMDb movie metadata collection, examine the model behind the categories, and test a prediction using the notebook’s Random Forest pipeline.</p><div className="hero-actions"><Link className="button button-primary" to="/predict">Open prediction studio <ArrowRight size={16} /></Link><Link className="text-link" to="/dataset">Explore the dataset <ArrowRight size={15} /></Link></div></div>
      <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="hero-reel"><Film size={58} strokeWidth={1} /><span>IMDB<br />ARCHIVE</span></div><div className="hero-index">DATASET<br /><b>01 — {data.raw_rows.toLocaleString()}</b></div></div>
      <div className="hero-foot"><span>ANALYSIS WINDOW</span><b>IMDb movie metadata · {data.raw_columns} fields</b><span>MODEL TARGET</span><b>IMDb score category</b></div>
    </section>
    <div className="stats-grid">{stats.map(({ label, value, detail, icon: Icon }) => <article className="stat-card" key={label}><div className="stat-top"><span>{label}</span><Icon size={16} /></div><strong>{value}</strong><small>{detail}</small></article>)}</div>
    <div className="overview-grid">
      <Panel title="Success categories" eyebrow="CLEANED DATA / N = 4,998" action={<Link className="panel-link" to="/analytics">Full analysis <ArrowRight size={14} /></Link>}>
        <div className="chart-wrap chart-medium"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.categories} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}><CartesianGrid stroke="#2b2c28" vertical={false} /><XAxis dataKey="category" axisLine={false} tickLine={false} tick={{ fill: "#a7a69d", fontSize: 12 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: "#777970", fontSize: 11 }} /><Tooltip contentStyle={{ background: "#1d1e1a", border: "1px solid #3a3b34", borderRadius: 4 }} /><Bar dataKey="count" radius={[3, 3, 0, 0]}>{data.categories.map((row) => <Cell key={row.category} fill={colors[row.category]} />)}</Bar></BarChart></ResponsiveContainer></div>
        <div className="legend-row">{data.categories.map((category) => <div key={category.category}><span className="legend-swatch" style={{ backgroundColor: colors[category.category] }} />{category.category}<b>{category.percent}%</b></div>)}</div>
      </Panel>
      <Panel title="The selected model" eyebrow="DEPLOYED MODEL / STRATIFIED 80:20">
        <div className="model-highlight"><div className="model-seal"><span>RF</span><small>MODEL</small></div><div><h3>Random Forest · reduced inputs</h3><span className="subtle">200 trees · class-balanced · 10 movie attributes</span></div></div>
        <div className="metric-lines"><div><span>Accuracy</span><b>{data.model_experiment?.accuracy.toFixed(1) ?? rf?.accuracy.toFixed(1)}%</b></div><div><span>Macro F1</span><b>{data.model_experiment?.macro_f1.toFixed(2) ?? rf?.macro_f1.toFixed(2)}%</b></div><div><span>Balanced accuracy</span><b>{data.model_experiment?.balanced_accuracy.toFixed(2) ?? rf?.balanced_accuracy.toFixed(2)}%</b></div></div>
        <div className="notice notice-warn"><TriangleAlert size={16} /><p>{flopRecall === undefined ? <>The notebook baseline Flop recall is <strong>0%</strong>.</> : <>Flop recall for the deployed model is <strong>{flopRecall.toFixed(1)}%</strong>.</>} The test set contains only 8 Flop titles; overall accuracy obscures this weakness.</p></div>
        <Link className="button button-outline full-width" to="/models">Inspect model evaluation <ArrowRight size={15} /></Link>
      </Panel>
    </div>
    <div className="lower-grid"><Panel title="Data quality, in context" eyebrow="SOURCE / RAW CSV"><p className="body-copy">The raw dataset has {data.missing_values.toLocaleString()} missing cells and {data.duplicate_rows} duplicate records. The modeling cohort removes duplicates and rows without an IMDb score; missing feature values are imputed inside the training pipeline.</p><Link className="text-link" to="/dataset">Review columns and missingness <ArrowRight size={15} /></Link></Panel><Panel title="An important distinction" eyebrow="READ THE TARGET CAREFULLY"><p className="body-copy">“Success” is a category created from an IMDb score—not a measurement of ticket sales, revenue, or commercial performance.</p><div className="micro-metrics"><span><ArrowDownRight size={15} /> [0, 3) <small>Flop</small></span><span><ArrowUpRight size={15} /> [3, 6) <small>Average</small></span><span><ArrowUpRight size={15} /> [6, 10) <small>Hit</small></span></div></Panel></div>
  </>;
}
