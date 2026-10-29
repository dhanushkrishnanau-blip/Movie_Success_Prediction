import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api";
import type { ConfusionMatrix, ModelReport, ModelResult } from "../types";
import { ErrorState, Loading, PageHeader, Panel } from "../components/States";

export function ModelPage() {
  const [comparison, setComparison] = useState<{ source: string; split: string; items: ModelResult[]; deployed_experiment?: ModelResult & {
    baseline_metrics: Record<string, number>;
    baseline_source: string;
    metric_change_percentage_points: Record<string, number>;
    flop_recall_percent: number;
  } | null } | null>(null);
  const [matrix, setMatrix] = useState<ConfusionMatrix | null>(null);
  const [report, setReport] = useState<ModelReport | null>(null);
  const [importance, setImportance] = useState<Array<{ feature: string; importance: number }> | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api.modelComparison(), api.confusionMatrix(), api.classificationReport(), api.featureImportance()])
      .then(([models, confusion, classification, features]) => {
        setComparison(models); setMatrix(confusion); setReport(classification);
        setImportance(features.items);
      }).catch((reason: Error) => {
        setError(reason.message);
        api.modelComparison().then(setComparison).catch(() => undefined);
      });
  }, []);
  if (error && !comparison) return <ErrorState error={error} />;
  if (!comparison) return <Loading />;
  const rf = comparison.items.find((model) => model.model === "Random Forest");
  const selectedModel = comparison.deployed_experiment ?? rf;
  const displayedModels = comparison.items
    .filter((model) => model.model !== "Random Forest")
    .concat(comparison.deployed_experiment ?? rf ?? []);
  const flopRecall = report?.report.Flop ? report.report.Flop.recall * 100 : null;
  const chartData = [...comparison.items].sort((a, b) => b.macro_f1 - a.macro_f1);
  return <>
    <PageHeader eyebrow="EVALUATION WORKSPACE / 05" title="Read the model honestly." description="The deployed Random Forest is the selected model. Other algorithms are shown for reference using the saved notebook results." />
    {error && <div className="notice notice-warn"><p>Some model details are unavailable: {error}</p></div>}
    <div className="metric-ribbon"><div><span>{comparison.deployed_experiment ? "DEPLOYED RF ACCURACY" : "NOTEBOOK RF ACCURACY"}</span><b>{selectedModel?.accuracy.toFixed(2)}%</b></div><div><span>MACRO F1</span><b>{selectedModel?.macro_f1.toFixed(2)}%</b></div><div><span>BALANCED ACCURACY</span><b>{selectedModel?.balanced_accuracy.toFixed(2)}%</b></div><div><span>FLOP CLASS RECALL</span><b className="warning-text">{flopRecall === null ? "—" : `${flopRecall.toFixed(1)}%`}</b></div></div>
    <Panel title="Algorithm comparison" eyebrow="SAVED NOTEBOOK HOLDOUT RESULTS" action={<span className="table-unit">METRICS IN %</span>}>
      <div className="table-scroll"><table><thead><tr><th>Algorithm</th><th>Accuracy</th><th>Macro precision</th><th>Balanced accuracy</th><th>Macro F1</th></tr></thead><tbody>{displayedModels.map((item) => {
        const deployed = item.model === "Random Forest · reduced inputs";
        return <tr key={item.model} className={deployed ? "highlight-row" : ""}><td>{deployed ? "Random Forest" : item.model}{deployed && <span className="selected-tag">DEPLOYED</span>}</td><td>{item.accuracy.toFixed(deployed ? 2 : 1)}</td><td>{item.macro_precision.toFixed(2)}</td><td>{item.balanced_accuracy.toFixed(2)}</td><td>{item.macro_f1.toFixed(2)}</td></tr>;
      })}</tbody></table></div>
      <p className="fine-print">{comparison.source} {comparison.split} The deployed Random Forest uses the reduced 10-field input set; other algorithms retain their saved notebook results.</p>
    </Panel>
    <div className="analytics-grid">
      <Panel title="Macro F1 by model" eyebrow="PER-CLASS BALANCE"><div className="chart-wrap chart-large"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} layout="vertical" margin={{ left: 26, right: 14, top: 6, bottom: 0 }}><CartesianGrid stroke="#2b2c28" horizontal={false} /><XAxis type="number" domain={[0, 60]} axisLine={false} tickLine={false} tick={{ fill: "#777970", fontSize: 10 }} /><YAxis type="category" dataKey="model" width={160} axisLine={false} tickLine={false} tick={{ fill: "#a7a69d", fontSize: 10 }} /><Tooltip contentStyle={{ background: "#1d1e1a", border: "1px solid #3a3b34" }} /><Bar dataKey="macro_f1" fill="#77866c" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div></Panel>
      <Panel title="Reduced-input model class report" eyebrow="REPRODUCED HOLDOUT DETAIL">{report && <><div className="table-scroll"><table><thead><tr><th>True class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{report.labels.map((label) => { const row = report.report[label]; return <tr key={label} className={label === "Flop" ? "warning-row" : ""}><td>{label}</td><td>{(row.precision * 100).toFixed(1)}%</td><td>{(row.recall * 100).toFixed(1)}%</td><td>{(row["f1-score"] * 100).toFixed(1)}%</td><td>{row.support}</td></tr>; })}</tbody></table></div><p className="fine-print">{report.source}</p></>}</Panel>
      <Panel title="Reduced-input model confusion matrix" eyebrow="TRUE CLASS × PREDICTED CLASS" className="span-two">{matrix && <><div className="table-scroll matrix-table"><table><thead><tr><th>Actual ↓ / Predicted →</th>{matrix.labels.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>{matrix.matrix.map((row, index) => <tr key={matrix.labels[index]}><th>{matrix.labels[index]}</th>{row.map((count, column) => <td key={column} style={{ backgroundColor: `rgba(130,151,121,${Math.min(0.7, count / 500 + 0.06)})` }}><strong>{count}</strong></td>)}</tr>)}</tbody></table></div><p className="fine-print">{matrix.source}. Diagonal cells are correct classifications; off-diagonal cells are errors.</p></>}</Panel>
      <Panel title="Feature importance" eyebrow="FITTED RANDOM FOREST"><div className="chart-wrap chart-large"><ResponsiveContainer width="100%" height="100%"><BarChart data={importance ?? []} layout="vertical" margin={{ left: 65, right: 14, top: 6, bottom: 0 }}><CartesianGrid stroke="#2b2c28" horizontal={false} /><XAxis type="number" axisLine={false} tickLine={false} tick={{ fill: "#777970", fontSize: 10 }} /><YAxis dataKey="feature" type="category" width={170} axisLine={false} tickLine={false} tick={{ fill: "#a7a69d", fontSize: 9 }} /><Tooltip contentStyle={{ background: "#1d1e1a", border: "1px solid #3a3b34" }} /><Bar dataKey="importance" fill="#c8a05b" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div><p className="fine-print">Feature importance reflects model reliance, not causal influence.</p></Panel>
    </div>
    <div className="notice notice-warn"><strong>Why accuracy can mislead:</strong><p>In the cleaned cohort only 41 of 4,998 titles are Flops. The stratified test split includes 8 Flops; zero recall means the model misses every one. Macro F1 and balanced accuracy weight classes equally, so they expose the poor minority-class result. Categories still describe IMDb score, not revenue.</p></div>
  </>;
}
