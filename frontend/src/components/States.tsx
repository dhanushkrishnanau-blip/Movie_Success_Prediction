import { AlertCircle, LoaderCircle } from "lucide-react";
import type { ReactNode } from "react";

export function Loading({ label = "Loading verified data…" }: { label?: string }) {
  return <div className="state-box"><LoaderCircle className="spin" size={18} />{label}</div>;
}

export function ErrorState({ error, retry }: { error: string; retry?: () => void }) {
  return (
    <div className="error-box" role="alert">
      <AlertCircle size={19} /><div><strong>Unable to load this data</strong><p>{error}</p></div>
      {retry && <button className="button button-quiet" onClick={retry}>Try again</button>}
    </div>
  );
}

export function Panel({ title, eyebrow, action, children, className = "" }: {
  title: string; eyebrow?: string; action?: ReactNode; children: ReactNode; className?: string;
}) {
  return <section className={`panel ${className}`}><div className="panel-heading"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{action}</div>{children}</section>;
}

export function PageHeader({ eyebrow, title, description, children }: {
  eyebrow: string; title: string; description: string; children?: ReactNode;
}) {
  return <div className="page-header"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{children}</div>;
}
