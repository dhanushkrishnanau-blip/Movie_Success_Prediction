import { ArrowRight, Database, Microscope, Scissors, Split, Target } from "lucide-react";
import { PageHeader, Panel } from "../components/States";

const stages = [
  ["01", "Dataset", "5,043 rows · 28 columns", Database],
  ["02", "Cleaning", "Remove duplicates; require IMDb score", Scissors],
  ["03", "EDA", "Inspect scores, missingness, and categories", Microscope],
  ["04", "Target", "IMDb score → Flop / Average / Hit", Target],
  ["05", "Preprocess", "Impute, scale numeric, one-hot encode", Split],
  ["06", "Split", "Stratified 80:20 · random_state 42", Split],
  ["07", "Train", "Compare models; fit class-balanced RF", Microscope],
  ["08", "Evaluate", "Accuracy, macro F1, balance, confusion matrix", Target],
];

export function MethodologyPage() {
  return <>
    <PageHeader eyebrow="CAPSTONE NOTES / 07" title="From metadata to model." description="A transparent walkthrough of the notebook’s data preparation, target definition, model evaluation, and practical limitations." />
    <Panel title="Problem statement" eyebrow="WHY THIS ANALYSIS"><p className="body-copy large-copy">Can movie metadata help classify a title into an IMDb-score band? CineScope documents the dataset and evaluates that question without presenting IMDb rating as a proxy for box-office revenue.</p></Panel>
    <Panel title="The workflow" eyebrow="NOTEBOOK METHOD"><div className="workflow">{stages.map(([number, title, detail, Icon], index) => <div className="workflow-stage" key={String(number)}><div className="workflow-icon"><Icon size={18} /></div><span>{String(number)}</span><b>{String(title)}</b><small>{String(detail)}</small>{index < stages.length - 1 && <ArrowRight className="workflow-arrow" size={15} />}</div>)}</div></Panel>
    <div className="method-grid">
      <Panel title="Target labels" eyebrow="IMDB_SCORE"><div className="boundary-list"><div><b>Flop</b><span>0 ≤ score &lt; 3</span></div><div><b>Average</b><span>3 ≤ score &lt; 6</span></div><div><b>Hit</b><span>6 ≤ score &lt; 10</span></div></div><p className="fine-print">The notebook uses bins [0,3,6,10] with the right edge excluded. Any score of exactly 10 is outside the defined bins.</p></Panel>
      <Panel title="Data and features" eyebrow="MODELING POPULATION"><p className="body-copy">The notebook removes duplicates and records without an IMDb score, yielding 4,998 examples. It holds out 20% using stratification. The deployed model uses 10 inputs: director name, runtime, genre, three actor names, budget, release year, language, and country.</p><p className="body-copy">Facebook-like counts, color, IMDb score, movie title, plot keywords, voter/review counts, gross, and IMDb URL are excluded. Person names are learned as categories only when they appear at least twice in training. The random split can put the same person in train and test, so evaluation may overstate performance on entirely new people; names may also be unavailable before release.</p></Panel>
      <Panel title="Preprocessing and models" eyebrow="PIPELINE"><p className="body-copy">Numeric columns are median-imputed and standardized. Categorical and person-name columns use most-frequent imputation and one-hot encoding; rare and unseen values are handled by the fitted encoders.</p><p className="body-copy">The Model Lab compares this reduced-input model with the previous deployed model on the same stratified holdout. The saved notebook comparison remains separate and unchanged.</p></Panel>
      <Panel title="Limits and next steps" eyebrow="INTERPRET WITH CARE"><ul className="method-list"><li>Only 41 of 4,998 cleaned rows are Flops; the test split contains 8.</li><li>Check the deployed model’s class report in Model Lab; the original notebook model had 0% Flop recall.</li><li>Accuracy is dominated by common classes; macro metrics surface minority-class performance.</li><li>Metadata includes fields that may be post-release or unavailable at prediction time.</li><li>Future work: improve minority-class coverage, use time-aware validation, and assess probability calibration and external validity.</li></ul></Panel>
    </div>
  </>;
}
