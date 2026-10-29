import { useEffect, useState, type FormEvent } from "react";
import { ArrowRight, RotateCcw, TriangleAlert } from "lucide-react";
import { api } from "../services/api";
import { PageHeader, Panel } from "../components/States";

const numericFields = [
  ["duration", "Movie duration (minutes)", "110"],
  ["budget", "Budget in USD", "25000000"],
  ["title_year", "Release year", "2020"],
] as const;
const categoricalFields = [
  ["genres", "Genre (separate with |)", "Drama|Thriller"],
  ["director_name", "Director name", "Director's name"],
  ["actor_1_name", "Lead actor name", "Lead cast member"],
  ["actor_2_name", "Second actor name", "Second cast member"],
  ["actor_3_name", "Third actor name", "Third cast member"],
  ["language", "Language", "English"],
  ["country", "Country", "USA"],
] as const;

function NameField({
  field,
  label,
  placeholder,
  value,
  onChange,
}: {
  field: string;
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionState, setSuggestionState] = useState("");

  useEffect(() => {
    const query = value.trim();
    if (query.length < 2) {
      setSuggestions([]);
      setSuggestionState("");
      return;
    }

    let active = true;
    setSuggestionState("Searching names…");
    const timeout = window.setTimeout(() => {
      api.nameSuggestions(query)
        .then(({ items }) => {
          if (!active) return;
          setSuggestions(items);
          setSuggestionState(items.length ? "" : "No match found; you can still enter any name.");
        })
        .catch(() => {
          if (!active) return;
          setSuggestions([]);
          setSuggestionState("Suggestions unavailable; you can still enter any name.");
        });
    }, 200);

    return () => {
      active = false;
      window.clearTimeout(timeout);
    };
  }, [value]);

  const listId = `suggestions-${field}`;
  const statusId = `${listId}-status`;
  return <label className="form-field" htmlFor={field}>
    {label}
    <input
      id={field}
      list={listId}
      value={value}
      placeholder={placeholder}
      autoComplete="off"
      aria-describedby={statusId}
      onChange={(event) => onChange(event.target.value)}
    />
    <datalist id={listId}>
      {suggestions.map((name) => <option key={name} value={name} />)}
    </datalist>
    <small className="suggestion-status" id={statusId} aria-live="polite">
      {suggestionState || "Type 2+ characters for name suggestions."}
    </small>
  </label>;
}

export function PredictPage() {
  const [form, setForm] = useState<Record<string, string>>({});
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.predict>> | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [flopRecall, setFlopRecall] = useState<number | null>(null);
  const [evaluationError, setEvaluationError] = useState("");

  useEffect(() => {
    api.classificationReport()
      .then(({ report }) => setFlopRecall(report.Flop.recall * 100))
      .catch((reason: Error) => setEvaluationError(reason.message));
  }, []);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(""); setResult(null); setSubmitting(true);
    const features: Record<string, string | number | null> = {};
    for (const [field] of numericFields) features[field] = form[field] === "" || form[field] === undefined ? null : Number(form[field]);
    for (const [field] of categoricalFields) features[field] = form[field]?.trim() || null;
    try { setResult(await api.predict(features)); } catch (reason) { setError(reason instanceof Error ? reason.message : "Prediction failed."); }
    finally { setSubmitting(false); }
  };
  const reset = () => { setForm({}); setResult(null); setError(""); };
  return <>
    <PageHeader eyebrow="INTERACTIVE INFERENCE / 04" title="Build a film profile." description="Use director and cast names, genre, runtime, budget, year, language, and country. Facebook-like counts and color are not required." />
    <div className="predict-layout">
      <Panel title="Movie attributes" eyebrow="MODEL INPUT / REDUCED FEATURE SET">
        <form onSubmit={submit}>
          <div className="form-grid">{categoricalFields.map(([key, label, placeholder]) => key.endsWith("_name")
            ? <NameField
              key={key}
              field={key}
              label={label}
              placeholder={placeholder}
              value={form[key] ?? ""}
              onChange={(value) => setForm((values) => ({ ...values, [key]: value }))}
            />
            : <label className="form-field" key={key}>{label}<input value={form[key] ?? ""} placeholder={placeholder} onChange={(event) => setForm((values) => ({ ...values, [key]: event.target.value }))} /></label>)}
            {numericFields.map(([key, label, placeholder]) => <label className="form-field" key={key}>{label}<input type="number" min="0" step="any" value={form[key] ?? ""} placeholder={placeholder} onChange={(event) => setForm((values) => ({ ...values, [key]: event.target.value }))} /></label>)}
          </div>
          <div className="form-actions"><button className="button button-primary" disabled={submitting} type="submit">{submitting ? "Evaluating…" : "Run prediction"} <ArrowRight size={15} /></button><button className="button button-quiet" type="button" onClick={reset}><RotateCcw size={14} /> Reset fields</button></div>
        </form>
      </Panel>
      <aside className="prediction-aside">
        <Panel title="Prediction result" eyebrow="MODEL OUTPUT">
          {result ? <div className={`prediction-result result-${result.predicted_category.toLowerCase()}`}><span>ESTIMATED IMDb CATEGORY</span><strong>{result.predicted_category}</strong><small>{result.model_name}</small>{result.probabilities && <div className="probability-list">{Object.entries(result.probabilities).map(([label, probability]) => <div key={label}><span>{label}</span><div><i style={{ width: `${probability * 100}%` }} /></div><b>{(probability * 100).toFixed(1)}%</b></div>)}</div>}<p>{result.note}</p></div> : <div className="result-empty"><div className="empty-rings"><ArrowRight size={23} /></div><p>Your model output will appear here after you submit a profile.</p></div>}
          {error && <div className="notice notice-error"><TriangleAlert size={16} /><p>{error}</p></div>}
        </Panel>
        <div className="notice notice-warn" role="status">
          <TriangleAlert size={16} />
          {flopRecall === null
            ? <p>{evaluationError ? `Flop-detection evaluation is unavailable: ${evaluationError}` : "Loading Flop-detection evaluation…"}</p>
            : <p><strong>Flop detection limitation:</strong> The model identified {flopRecall.toFixed(1)}% of Flop movies in its evaluation. A Flop prediction—or another category—may be unreliable, especially for Flop titles.</p>}
        </div>
        <div className="notice notice-warn"><TriangleAlert size={16} /><p>This classifier predicts a category derived from IMDb score—not box-office performance. Review the deployed model’s class-level recall in Model Lab; do not treat this prediction as a guarantee.</p></div>
        <div className="schema-note"><b>MODEL INPUTS</b><span>10 fields · no Facebook-like counts or color · IMDb score excluded</span><span>Unseen director and actor names may provide limited signal</span><span>Missing numeric values are median-imputed; missing categories use the most common training value</span></div>
      </aside>
    </div>
  </>;
}
