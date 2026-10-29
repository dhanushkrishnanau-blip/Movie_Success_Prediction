import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../app/App";
import { DatasetPage } from "./DatasetPage";
import { ModelPage } from "./ModelPage";
import { OverviewPage } from "./OverviewPage";
import { PredictPage } from "./PredictPage";

vi.mock("../services/api", () => ({
  api: {
    health: vi.fn().mockResolvedValue({ status: "ok", dataset: true, model: true }),
    nameSuggestions: vi.fn().mockResolvedValue({ items: ["Christopher Nolan"] }),
    classificationReport: vi.fn().mockResolvedValue({
      labels: ["Flop", "Average", "Hit"],
      report: {
        Flop: { precision: 0, recall: 0, "f1-score": 0, support: 8 },
        Average: { precision: 0.55, recall: 0.54, "f1-score": 0.54, support: 280 },
        Hit: { precision: 0.82, recall: 0.84, "f1-score": 0.83, support: 712 },
      },
    }),
    modelComparison: vi.fn().mockResolvedValue({
      source: "Saved notebook holdout",
      split: "80/20 split",
      items: [{ model: "Random Forest", accuracy: 76.2, macro_precision: 47.37, balanced_accuracy: 48.68, macro_f1: 47.94 }],
      deployed_experiment: {
        model: "Random Forest · reduced inputs",
        accuracy: 74.6,
        macro_precision: 45.69,
        balanced_accuracy: 45.83,
        macro_f1: 45.76,
        baseline_metrics: { accuracy_percent: 76.4, macro_precision_percent: 47.26, balanced_accuracy_percent: 47.54, macro_f1_percent: 47.4 },
        baseline_source: "Previous deployed model on the same holdout.",
        metric_change_percentage_points: { accuracy_percent: -1.8, macro_precision_percent: -1.57, balanced_accuracy_percent: -1.71, macro_f1_percent: -1.64 },
        flop_recall_percent: 0,
      },
    }),
    confusionMatrix: vi.fn().mockResolvedValue({
      labels: ["Flop", "Average", "Hit"],
      matrix: [[0, 6, 2], [0, 151, 129], [0, 117, 595]],
      source: "Test holdout",
    }),
    featureImportance: vi.fn().mockResolvedValue({ available: true, items: [] }),
    schema: vi.fn().mockResolvedValue({
      columns: [
        { name: "movie_title", dtype: "object", missing: 0, missing_percent: 0, unique: 1 },
        { name: "director_name", dtype: "object", missing: 0, missing_percent: 0, unique: 1 },
      ],
    }),
    rows: vi.fn().mockResolvedValue({
      items: [{ row_id: 0, movie_title: "Example", director_name: "Example Director" }],
      page: 1, page_size: 20, total: 1, pages: 1,
    }),
    overview: vi.fn().mockResolvedValue({
      filename: "movie_metadata.csv", raw_rows: 5043, raw_columns: 28,
      clean_rows: 4998, numeric_columns: 13, categorical_columns: 15,
      missing_values: 2700, duplicate_rows: 45, unique_genres: 26,
      categories: [
        { category: "Flop", count: 41, percent: 0.82 },
        { category: "Average", count: 1401, percent: 28.03 },
        { category: "Hit", count: 3556, percent: 71.15 },
      ],
      model_comparison: [{ model: "Random Forest", accuracy: 76.2, macro_f1: 47.94, balanced_accuracy: 48.68, macro_precision: 47.37 }],
    }),
  },
}));

afterEach(cleanup);
beforeEach(() => window.localStorage.removeItem("cinescope-theme"));

describe("CineScope overview", () => {
  it("renders the real-data overview", async () => {
    render(<MemoryRouter><OverviewPage /></MemoryRouter>);
    expect(await screen.findByText("Stories, measured.")).toBeInTheDocument();
    expect(screen.getByText("Random Forest · reduced inputs")).toBeInTheDocument();
  });

  it("renders navigation", () => {
    render(<MemoryRouter><App /></MemoryRouter>);
    expect(screen.getByText("Dataset")).toBeInTheDocument();
  });

  it("can switch to the original look and back", async () => {
    const user = userEvent.setup();
    const { container } = render(<MemoryRouter><App /></MemoryRouter>);
    const shell = container.querySelector(".app-shell");
    expect(shell).toHaveAttribute("data-theme", "diary");
    await user.click(screen.getByRole("button", { name: "Switch to original CineScope design" }));
    expect(shell).toHaveAttribute("data-theme", "original");
    expect(window.localStorage.getItem("cinescope-theme")).toBe("original");
    await user.click(screen.getByRole("button", { name: "Switch to film diary design" }));
    expect(shell).toHaveAttribute("data-theme", "diary");
  });

  it("shows the updated reduced-input Random Forest accuracy in the main comparison", async () => {
    render(<MemoryRouter><ModelPage /></MemoryRouter>);
    expect(await screen.findByText("DEPLOYED RF ACCURACY")).toBeInTheDocument();
    expect(screen.getByText("74.60%")).toBeInTheDocument();
    expect(screen.getByText("DEPLOYED")).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /Random Forest DEPLOYED 74\.60 45\.69 45\.83 45\.76/ })).toBeInTheDocument();
    expect(screen.queryByText("Previous deployed model (20 fields)")).not.toBeInTheDocument();
  });

  it("shows director name in the dataset table by default", async () => {
    render(<MemoryRouter><DatasetPage /></MemoryRouter>);
    expect(await screen.findByRole("columnheader", { name: "director name" })).toBeInTheDocument();
    expect(await screen.findByText("Example Director")).toBeInTheDocument();
  });

  it("shows only the reduced model input fields", () => {
    render(<MemoryRouter><PredictPage /></MemoryRouter>);
    expect(screen.getByLabelText(/Director name/)).toBeInTheDocument();
    expect(screen.getByLabelText("Movie duration (minutes)")).toBeInTheDocument();
    expect(screen.getByLabelText("Genre (separate with |)")).toBeInTheDocument();
    expect(screen.queryByLabelText("Director Facebook likes")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Color or black-and-white")).not.toBeInTheDocument();
  });

  it("shows the evaluated Flop recall beside prediction", async () => {
    render(<MemoryRouter><PredictPage /></MemoryRouter>);
    expect(await screen.findByText(/identified 0\.0% of Flop movies/)).toBeInTheDocument();
  });

  it("suggests full director names while typing a partial name", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><PredictPage /></MemoryRouter>);
    await user.type(screen.getByLabelText(/Director name/), "nol");
    await waitFor(() => {
      expect(
        document.querySelector('#suggestions-director_name option[value="Christopher Nolan"]'),
      ).not.toBeNull();
    });
  });
});
