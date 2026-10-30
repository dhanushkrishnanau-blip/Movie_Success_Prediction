import type {
  CategoryAnalytics,
  ConfusionMatrix,
  DatasetColumn,
  DatasetResponse,
  ModelReport,
  ModelResult,
  MovieRecord,
  Overview,
  PredictionExplanation,
  ResearchChatMessage,
  ResearchChatMode,
  ResearchChatResponse,
  ScoreDistribution,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new Error("Cannot reach CineScope API. Start the FastAPI backend and try again.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

const query = (params: Record<string, string | number | undefined>) =>
  new URLSearchParams(
    Object.entries(params).flatMap(([key, value]) =>
      value === undefined || value === "" ? [] : [[key, String(value)]],
    ),
  ).toString();

export const api = {
  health: () => request<{ status: string; dataset: boolean; model: boolean }>("/health"),
  overview: () => request<Overview>("/overview"),
  schema: () => request<{ columns: DatasetColumn[] }>("/dataset/schema"),
  nameSuggestions: (value: string) =>
    request<{ items: string[] }>(`/dataset/name-suggestions?${query({ query: value })}`),
  rows: (params: Record<string, string | number | undefined>) =>
    request<DatasetResponse>(`/dataset/rows?${query(params)}`),
  movies: (params: Record<string, string | number | undefined>) =>
    request<DatasetResponse>(`/movies?${query(params)}`),
  movie: (id: number) => request<MovieRecord>(`/movies/${id}`),
  scoreDistribution: () => request<ScoreDistribution>("/analytics/score-distribution"),
  categories: () => request<CategoryAnalytics>("/analytics/categories"),
  genres: () => request<{ items: Array<{ genre: string; movies: number; mean_score: number }> }>(
    "/analytics/genres",
  ),
  countries: () => request<{ items: Array<{ country: string; movies: number }> }>(
    "/analytics/countries",
  ),
  relationships: () =>
    request<{
      budget_vs_score: Array<{ budget: number; imdb_score: number }>;
      budget_sample_size: number;
      budget_non_missing: number;
      year_trends: Array<{ year: number; mean_score: number; movies: number }>;
    }>("/analytics/relationships"),
  correlation: () =>
    request<{ columns: string[]; values: Array<Array<number | null>> }>(
      "/analytics/correlation",
    ),
  columnStats: (column: string) =>
    request<Record<string, unknown>>(`/dataset/column-stats/${encodeURIComponent(column)}`),
  modelComparison: () =>
    request<{ source: string; split: string; items: ModelResult[] }>("/models/comparison"),
  confusionMatrix: () => request<ConfusionMatrix>("/models/confusion-matrix"),
  classificationReport: () => request<ModelReport>("/models/classification-report"),
  featureImportance: () =>
    request<{ available: boolean; items: Array<{ feature: string; importance: number }> }>(
      "/models/feature-importance",
    ),
  predict: (features: Record<string, string | number | null>) =>
    request<{
      predicted_category: string;
      model_name: string;
      probabilities: Record<string, number> | null;
      note: string;
    }>("/predict", { method: "POST", body: JSON.stringify(features) }),
  explainPrediction: (features: Record<string, string | number | null>) =>
    request<PredictionExplanation>("/researcher/explain", {
      method: "POST",
      body: JSON.stringify(features),
    }),
  researcherChat: (
    mode: ResearchChatMode,
    question: string,
    history: ResearchChatMessage[],
  ) =>
    request<ResearchChatResponse>("/researcher/chat", {
      method: "POST",
      body: JSON.stringify({ mode, question, history }),
    }),
  downloadDataset: async () => {
    let response: Response;
    try {
      response = await fetch(`${API_BASE}/dataset/export`);
    } catch {
      throw new Error("Cannot reach CineScope API. Start the FastAPI backend and try again.");
    }
    if (!response.ok) throw new Error("The dataset export could not be downloaded.");
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "movie_metadata.csv";
    link.click();
    URL.revokeObjectURL(url);
  },
};
