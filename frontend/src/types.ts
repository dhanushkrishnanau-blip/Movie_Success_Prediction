export interface Overview {
  filename: string;
  raw_rows: number;
  raw_columns: number;
  clean_rows: number;
  numeric_columns: number;
  categorical_columns: number;
  missing_values: number;
  duplicate_rows: number;
  unique_genres: number;
  categories: Array<{ category: string; count: number; percent: number }>;
  model_comparison: Array<ModelResult>;
  model_experiment?: (ModelResult & { flop_recall_percent: number }) | null;
}

export interface ModelResult {
  model: string;
  accuracy: number;
  macro_precision: number;
  balanced_accuracy: number;
  macro_f1: number;
}

export interface DatasetColumn {
  name: string;
  dtype: string;
  missing: number;
  missing_percent: number;
  unique: number;
}

export interface DatasetResponse {
  items: Array<Record<string, string | number | null>>;
  page: number;
  page_size: number;
  total: number;
  pages: number;
}

export interface CategoryAnalytics {
  boundaries: Array<Record<string, string | number>>;
  items: Array<{ category: string; count: number; percent: number }>;
  score_summary: Array<{ category: string; count: number; mean_score: number | null }>;
}

export interface ScoreDistribution {
  count: number;
  mean: number;
  median: number;
  bins: Array<{ start: number; end: number; count: number }>;
}

export interface ModelReport {
  labels: string[];
  report: Record<string, { precision: number; recall: number; "f1-score": number; support: number }>;
  source: string;
}

export interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
  source: string;
}

export interface MovieRecord {
  [key: string]: unknown;
  row_id: number;
  movie_title: string;
  related?: Array<{
    row_id: number;
    movie_title: string | null;
    imdb_score: number | null;
  }>;
}

export interface FeatureEffect {
  feature: string;
  input_value: string | number | null;
  reference_value: string | number | null;
  predicted_probability_change_percentage_points: number;
}

export interface HistoricalComparison {
  available: boolean;
  matched_movies: number;
  criteria: string[];
  message?: string;
  mean_imdb_score?: number;
  median_imdb_score?: number;
  category_percentages?: Record<string, number>;
  interpretation_note?: string;
}

export interface PredictionExplanation {
  prediction: {
    predicted_category: string;
    model_name: string;
    probabilities: Record<string, number> | null;
    note: string;
  };
  feature_effects: FeatureEffect[];
  historical_comparison: HistoricalComparison;
  feature_availability: Array<{
    feature: string;
    status: string;
    note: string;
  }>;
  explanation_method: string;
  gemini_explanation: string | null;
  gemini_notice: string | null;
}

export type ResearchChatMode = "general" | "dataset";

export interface ResearchChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ResearchChatResponse {
  mode: ResearchChatMode;
  answer: string;
  evidence: Record<string, unknown> | null;
  notice: string | null;
}
