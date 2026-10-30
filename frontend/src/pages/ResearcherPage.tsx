import { useEffect, useState, type FormEvent } from "react";
import {
  Bot,
  LoaderCircle,
  Send,
  Sparkles,
  Trash2,
  TriangleAlert,
} from "lucide-react";
import { api } from "../services/api";
import { PageHeader, Panel } from "../components/States";
import type {
  GenrePerformanceEvidence,
  PredictionExplanation,
  ResearchChatMessage,
  ResearchChatMode,
} from "../types";

const fields = [
  { key: "director_name", label: "Director", type: "text", note: "Use only if announced before the prediction date." },
  { key: "actor_1_name", label: "Lead actor", type: "text", note: "Use an announced cast member." },
  { key: "actor_2_name", label: "Second actor", type: "text", note: "Use an announced cast member." },
  { key: "actor_3_name", label: "Third actor", type: "text", note: "Use an announced cast member." },
  { key: "genres", label: "Genres", type: "text", note: "Genre labels can be assigned retrospectively." },
  { key: "duration", label: "Duration (minutes)", type: "number", note: "Final runtime may not be known before release." },
  { key: "budget", label: "Budget (USD)", type: "number", note: "Budget may be estimated or unavailable." },
  { key: "title_year", label: "Planned release year", type: "number", note: "Release plans may change." },
  { key: "language", label: "Language", type: "text", note: "Confirm the relevant language is known." },
  { key: "country", label: "Country", type: "text", note: "Confirm the country definition and timing." },
] as const;

interface ChatEntry extends ResearchChatMessage {
  evidence?: Record<string, unknown> | null;
  notice?: string | null;
}

const emptyChats: Record<ResearchChatMode, ChatEntry[]> = {
  general: [],
  dataset: [],
};

function formatValue(value: string | number | null): string {
  return value === null ? "Missing (pipeline imputation used)" : String(value);
}

function isGenrePerformanceEvidence(
  evidence: Record<string, unknown>,
): evidence is Record<string, unknown> & GenrePerformanceEvidence {
  return evidence.operation === "genre_performance"
    && evidence.supported === true
    && Array.isArray(evidence.statistics)
    && evidence.statistics.every((item) =>
      typeof item === "object"
      && item !== null
      && "genre" in item
      && typeof item.genre === "string"
      && "movies" in item
      && typeof item.movies === "number"
      && "mean_imdb_score" in item
      && typeof item.mean_imdb_score === "number"
      && "median_imdb_score" in item
      && typeof item.median_imdb_score === "number")
    && typeof evidence.minimum_movies_per_genre === "number"
    && typeof evidence.excluded_genres_below_minimum === "number"
    && typeof evidence.eligible_genres === "number"
    && typeof evidence.requested_count === "number"
    && (typeof evidence.message === "string" || evidence.message === null)
    && typeof evidence.source === "string";
}

function NameField({
  field,
  label,
  value,
  onChange,
}: {
  field: string;
  label: string;
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

export function ResearcherPage() {
  const [tab, setTab] = useState<"explain" | "chat">("explain");
  const [features, setFeatures] = useState<Record<string, string>>({});
  const [explanation, setExplanation] = useState<PredictionExplanation | null>(null);
  const [explanationError, setExplanationError] = useState("");
  const [explaining, setExplaining] = useState(false);
  const [mode, setMode] = useState<ResearchChatMode>("general");
  const [chats, setChats] = useState(emptyChats);
  const [question, setQuestion] = useState("");
  const [chatError, setChatError] = useState("");
  const [sending, setSending] = useState(false);

  const submitExplanation = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setExplanation(null);
    setExplanationError("");
    setExplaining(true);
    const submitted: Record<string, string | number | null> = {};
    for (const field of fields) {
      const value = features[field.key]?.trim() ?? "";
      submitted[field.key] = value
        ? field.type === "number" ? Number(value) : value
        : null;
    }
    try {
      setExplanation(await api.explainPrediction(submitted));
    } catch (reason) {
      setExplanationError(reason instanceof Error ? reason.message : "Could not explain this prediction.");
    } finally {
      setExplaining(false);
    }
  };

  const sendQuestion = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const text = question.trim();
    if (!text || sending) return;
    const previous = chats[mode];
    setQuestion("");
    setChatError("");
    setChats((current) => ({
      ...current,
      [mode]: [...current[mode], { role: "user", content: text }],
    }));
    setSending(true);
    try {
      const history = previous.slice(-12).map((message) => ({
        role: message.role,
        content: message.content.slice(0, 1200),
      }));
      const response = await api.researcherChat(mode, text, history);
      setChats((current) => ({
        ...current,
        [mode]: [
          ...current[mode],
          {
            role: "assistant",
            content: response.answer,
            evidence: response.evidence,
            notice: response.notice,
          },
        ],
      }));
    } catch (reason) {
      setChatError(reason instanceof Error ? reason.message : "The assistant could not answer.");
    } finally {
      setSending(false);
    }
  };

  return <>
    <PageHeader
      eyebrow="AI RESEARCH WORKSPACE"
      title="Researcher"
      description="Explore the existing Random Forest and the movie dataset. Model predictions and Python-calculated statistics stay separate from Gemini's written explanations."
    />
    <div className="research-tabs" role="tablist" aria-label="Research tools">
      <button
        id="research-explain-tab"
        role="tab"
        aria-selected={tab === "explain"}
        aria-controls="research-explain-panel"
        className={`button ${tab === "explain" ? "button-primary" : "button-quiet"}`}
        onClick={() => setTab("explain")}
        type="button"
      >
        <Sparkles size={15} /> AI Prediction Explainer
      </button>
      <button
        id="research-chat-tab"
        role="tab"
        aria-selected={tab === "chat"}
        aria-controls="research-chat-panel"
        className={`button ${tab === "chat" ? "button-primary" : "button-quiet"}`}
        onClick={() => setTab("chat")}
        type="button"
      >
        <Bot size={15} /> AI Movie Research Assistant
      </button>
    </div>

    {tab === "explain" ? <div id="research-explain-panel" role="tabpanel" aria-labelledby="research-explain-tab">
      <div className="research-layout">
        <Panel title="Pre-release movie details" eyebrow="EXISTING RANDOM FOREST INPUTS">
          <p className="body-copy">The saved pipeline makes the category prediction. Blank fields are imputed by that pipeline; only enter details that would be known at your intended prediction date.</p>
          <form onSubmit={submitExplanation}>
            <div className="form-grid">
              {fields.map((field) => field.key.endsWith("_name")
                ? <NameField
                  key={field.key}
                  field={field.key}
                  label={field.label}
                  value={features[field.key] ?? ""}
                  onChange={(value) => setFeatures((current) => ({
                    ...current,
                    [field.key]: value,
                  }))}
                />
                : <label className="form-field" key={field.key}>
                  {field.label}
                  <input
                    type={field.type}
                    min={field.key === "title_year" ? 1800 : field.type === "number" ? 0 : undefined}
                    max={field.key === "title_year" ? 2200 : undefined}
                    step={field.type === "number" ? "any" : undefined}
                    value={features[field.key] ?? ""}
                    onChange={(event) => setFeatures((current) => ({
                      ...current,
                      [field.key]: event.target.value,
                    }))}
                  />
                  <small className="suggestion-status">{field.note}</small>
                </label>)}
            </div>
            <div className="form-actions">
              <button className="button button-primary" disabled={explaining} type="submit">
                {explaining ? "Running model…" : "Predict and explain"} <Sparkles size={15} />
              </button>
            </div>
          </form>
          {explanationError && <div className="notice notice-error" role="alert"><TriangleAlert size={16} /><p>{explanationError}</p></div>}
        </Panel>

        <Panel title="Prediction and evidence" eyebrow="MODEL OUTPUT + VERIFIED DATA">
          {explaining && <div className="state-box" role="status"><LoaderCircle className="spin" size={18} />Running the Random Forest and calculating comparisons…</div>}
          {!explanation && !explaining && <div className="result-empty"><p>Submit details to see the saved model's category, local sensitivity, and dataset comparison.</p></div>}
          {explanation && <>
            <div className={`prediction-result result-${explanation.prediction.predicted_category.toLowerCase()}`}>
              <span>RANDOM FOREST PREDICTION · IMDB-SCORE CATEGORY</span>
              <strong>{explanation.prediction.predicted_category}</strong>
              <small>{explanation.prediction.model_name}</small>
              {explanation.prediction.probabilities && <div className="probability-list">
                {Object.entries(explanation.prediction.probabilities).map(([label, probability]) => <div key={label}>
                  <span>{label}</span>
                  <div><i style={{ width: `${probability * 100}%` }} /></div>
                  <b>{(probability * 100).toFixed(1)}%</b>
                </div>)}
              </div>}
              <p>{explanation.prediction.note}</p>
            </div>

            <div className="research-subsection">
              <h3>Model sensitivity</h3>
              <p className="fine-print">{explanation.explanation_method}</p>
              <div className="table-scroll">
                <table className="research-table">
                  <thead><tr><th>Input</th><th>Submitted</th><th>Typical reference</th><th>Change in predicted-category probability</th></tr></thead>
                  <tbody>{explanation.feature_effects.map((effect) => <tr key={effect.feature}>
                    <th>{effect.feature}</th>
                    <td>{formatValue(effect.input_value)}</td>
                    <td>{formatValue(effect.reference_value)}</td>
                    <td>{effect.predicted_probability_change_percentage_points > 0 ? "+" : ""}{effect.predicted_probability_change_percentage_points.toFixed(2)} pp</td>
                  </tr>)}</tbody>
                </table>
              </div>
            </div>

            <div className="research-subsection">
              <h3>Historical dataset comparison</h3>
              <p className="fine-print">Python-calculated from the project dataset; historical association is not a cause or a promise about a new movie.</p>
              <p>{explanation.historical_comparison.matched_movies.toLocaleString()} matching scored movies · {explanation.historical_comparison.criteria.join(", ")}</p>
              {explanation.historical_comparison.available
                ? <div className="research-stats">
                  <span>Mean IMDb score <b>{explanation.historical_comparison.mean_imdb_score?.toFixed(2)}</b></span>
                  <span>Median IMDb score <b>{explanation.historical_comparison.median_imdb_score?.toFixed(2)}</b></span>
                  {Object.entries(explanation.historical_comparison.category_percentages ?? {}).map(([category, percent]) =>
                    <span key={category}>{category} <b>{percent.toFixed(1)}%</b></span>)}
                </div>
                : <p>{explanation.historical_comparison.message}</p>}
            </div>

            <div className="notice notice-warn"><TriangleAlert size={16} /><div>
              <p><strong>Pre-release availability needs review.</strong> The dataset does not record when these fields became known. Verify that every value was available at your prediction date.</p>
              <ul>{explanation.feature_availability.map((item) => <li key={item.feature}><b>{item.feature}:</b> {item.note}</li>)}</ul>
            </div></div>
            {explanation.gemini_explanation
              ? <div className="research-ai-answer"><h3>Gemini explanation</h3><p>{explanation.gemini_explanation}</p><small>Written from the model and Python evidence above. It does not set the category or calculate the displayed statistics.</small></div>
              : <div className="notice notice-warn"><TriangleAlert size={16} /><p>{explanation.gemini_notice ?? "Gemini explanation is unavailable; the Random Forest result and calculated evidence are still available."}</p></div>}
          </>}
        </Panel>
      </div>
    </div> : <div id="research-chat-panel" role="tabpanel" aria-labelledby="research-chat-tab">
      <Panel
        title="Ask about movies or your data"
        eyebrow="GEMINI-POWERED CHAT"
        action={<button className="button button-quiet" disabled={sending} type="button" onClick={() => {
          setChats((current) => ({ ...current, [mode]: [] }));
          setChatError("");
        }}><Trash2 size={14} /> Clear conversation</button>}
      >
        <div className="research-mode" role="group" aria-label="Assistant mode">
          <button type="button" className={`button ${mode === "general" ? "button-primary" : "button-quiet"}`} aria-pressed={mode === "general"} onClick={() => { setMode("general"); setChatError(""); }}>
            General Movie Assistant
          </button>
          <button type="button" className={`button ${mode === "dataset" ? "button-primary" : "button-quiet"}`} aria-pressed={mode === "dataset"} onClick={() => { setMode("dataset"); setChatError(""); }}>
            Dataset Researcher
          </button>
        </div>
        <p className="fine-print">{mode === "general"
          ? "General movie and filmmaking answers use Gemini's knowledge and are not live-verified."
          : "Dataset answers use fixed, safe Pandas analyses. Gemini explains the calculated results; it cannot run code or invent statistics."}</p>
        <div className="research-chat" aria-live="polite">
          {chats[mode].length === 0 && <div className="result-empty"><p>{mode === "general"
            ? "Ask about a film, actor, director, genre, filmmaking, or recommendations."
            : "Try: “Which genres have the highest average IMDb score?” or “Summarize movie budgets.”"}</p></div>}
          {chats[mode].map((entry, index) => <article className={`chat-message chat-${entry.role}`} key={`${mode}-${index}`}>
            <b>{entry.role === "user" ? "You" : "Research Assistant"}</b>
            <p>{entry.content}</p>
            {entry.notice && <small>{entry.notice}</small>}
            {entry.evidence && <details>
              <summary>Show Python-calculated evidence</summary>
              {isGenrePerformanceEvidence(entry.evidence)
                ? <>
                  <p className="fine-print">{entry.evidence.source}</p>
                  {entry.evidence.message && <p className="fine-print">
                    {entry.evidence.message}
                  </p>}
                  {entry.evidence.excluded_genres_below_minimum > 0
                    && <p className="fine-print">
                      {entry.evidence.excluded_genres_below_minimum} genres with fewer than{" "}
                      {entry.evidence.minimum_movies_per_genre} scored movies were excluded.
                    </p>}
                  <div className="table-scroll">
                    <table className="research-table">
                      <thead><tr>
                        <th>Rank</th>
                        <th>Genre</th>
                        <th>Average IMDb score</th>
                        <th>Movies</th>
                      </tr></thead>
                      <tbody>{entry.evidence.statistics.map((item) => <tr key={item.genre}>
                        <td>{item.rank}</td>
                        <th>{item.genre}</th>
                        <td>{item.mean_imdb_score.toFixed(2)}</td>
                        <td>{item.movies.toLocaleString()}</td>
                      </tr>)}</tbody>
                    </table>
                  </div>
                </>
                : <pre>{JSON.stringify(entry.evidence, null, 2)}</pre>}
            </details>}
          </article>)}
          {sending && <div className="state-box" role="status"><LoaderCircle className="spin" size={18} />Gemini is preparing a response…</div>}
        </div>
        {chatError && <div className="notice notice-error" role="alert"><TriangleAlert size={16} /><p>{chatError}</p></div>}
        <form className="research-chat-form" onSubmit={sendQuestion}>
          <label className="sr-only" htmlFor="research-question">Your question</label>
          <textarea id="research-question" maxLength={1200} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question…" rows={3} />
          <button className="button button-primary" disabled={sending || !question.trim()} type="submit">
            {sending ? "Sending…" : "Send"} <Send size={14} />
          </button>
        </form>
      </Panel>
    </div>}
  </>;
}
