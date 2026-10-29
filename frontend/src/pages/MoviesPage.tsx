import { useEffect, useState } from "react";
import { Search, ChevronLeft, ChevronRight, X } from "lucide-react";
import { api } from "../services/api";
import type { DatasetResponse, MovieRecord } from "../types";
import { ErrorState, Loading, PageHeader } from "../components/States";

export function MoviesPage() {
  const [data, setData] = useState<DatasetResponse | null>(null);
  const [search, setSearch] = useState("");
  const [genre, setGenre] = useState("");
  const [country, setCountry] = useState("");
  const [language, setLanguage] = useState("");
  const [contentRating, setContentRating] = useState("");
  const [minimumScore, setMinimumScore] = useState("");
  const [minimumYear, setMinimumYear] = useState("");
  const [sort, setSort] = useState("imdb_score");
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<MovieRecord | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(() => {
      api.movies({ page, page_size: 24, search, genre, country, language, content_rating: contentRating, min_score: minimumScore, min_year: minimumYear, sort_by: sort, direction: sort === "movie_title" ? "asc" : "desc" })
        .then(setData).catch((reason: Error) => setError(reason.message));
    }, 180);
    return () => window.clearTimeout(timer);
  }, [page, search, genre, country, language, contentRating, minimumScore, minimumYear, sort]);
  if (error) return <ErrorState error={error} />;
  return <>
    <PageHeader eyebrow="MOVIE ARCHIVE / 06" title="Browse the collection." description="Search and sort real movie records. Open a title to inspect its available metadata; missing posters and synopses are never invented." />
    <div className="movie-controls"><label className="search-field"><Search size={16} /><input placeholder="Search movie title…" value={search} onChange={(event) => { setSearch(event.target.value); setPage(1); }} /></label><input aria-label="Filter by genre" placeholder="Genre contains…" value={genre} onChange={(event) => { setGenre(event.target.value); setPage(1); }} /><input aria-label="Filter by country" placeholder="Country contains…" value={country} onChange={(event) => { setCountry(event.target.value); setPage(1); }} /><input aria-label="Filter by language" placeholder="Language contains…" value={language} onChange={(event) => { setLanguage(event.target.value); setPage(1); }} /><input aria-label="Filter by content rating" placeholder="Rating contains…" value={contentRating} onChange={(event) => { setContentRating(event.target.value); setPage(1); }} /><input aria-label="Minimum IMDb score" type="number" min="0" max="10" step=".1" placeholder="Minimum score" value={minimumScore} onChange={(event) => { setMinimumScore(event.target.value); setPage(1); }} /><input aria-label="Minimum release year" type="number" min="1800" max="2200" placeholder="Minimum year" value={minimumYear} onChange={(event) => { setMinimumYear(event.target.value); setPage(1); }} /><select aria-label="Sort movies" value={sort} onChange={(event) => setSort(event.target.value)}><option value="imdb_score">IMDb score</option><option value="budget">Budget</option><option value="title_year">Release year</option><option value="movie_title">Title</option></select></div>
    <div className="movie-grid">{!data ? <Loading /> : data.items.length === 0 ? <div className="empty-state">No movies match your filters.</div> : data.items.map((movie) => <button className="movie-card" key={String(movie.row_id)} onClick={() => api.movie(Number(movie.row_id)).then(setSelected).catch((reason: Error) => setError(reason.message))}><div className="movie-poster"><span className="poster-kicker">{String(movie.title_year ?? "YEAR UNKNOWN")}</span><span className="poster-title">{String(movie.movie_title ?? "").replace(/\s+\(\d{4}\)\s*$/, "") || "Untitled film"}</span><span className="poster-rule" /><span className="poster-genre">{String(movie.genres ?? "GENRE NOT LISTED").replaceAll("|", " · ")}</span><strong>{String(movie.imdb_score ?? "—")}</strong></div><div className="movie-card-meta"><b>{String(movie.movie_title ?? "Untitled film")}</b><span>{String(movie.country ?? "Country not listed")} · {String(movie.content_rating ?? "Unrated")}</span></div></button>)}</div>
    <div className="pagination"><span>{data ? `${data.total.toLocaleString()} titles · page ${data.page} of ${Math.max(data.pages, 1)}` : "Loading titles"}</span><div><button className="icon-button" aria-label="Previous page" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}><ChevronLeft size={17} /></button><button className="icon-button" aria-label="Next page" disabled={!data || page >= data.pages} onClick={() => setPage((value) => value + 1)}><ChevronRight size={17} /></button></div></div>
    {selected && <div className="dialog-scrim" role="presentation" onClick={() => setSelected(null)}><section className="movie-dialog" role="dialog" aria-modal="true" aria-label="Movie details" onClick={(event) => event.stopPropagation()}><button className="dialog-close icon-button" aria-label="Close details" onClick={() => setSelected(null)}><X size={18} /></button><div className="eyebrow">ARCHIVE RECORD · #{selected.row_id}</div><h2>{selected.movie_title}</h2><div className="detail-score">{String(selected.imdb_score ?? "—")} <small>IMDb score</small></div><dl className="detail-list">{["genres", "title_year", "duration", "country", "language", "content_rating", "budget", "director_name", "movie_facebook_likes"].map((key) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{selected[key] === null || selected[key] === undefined ? "Not recorded" : String(selected[key])}</dd></div>)}</dl><p className="fine-print">No synopsis or poster is present in the source dataset.</p>{selected.related?.length ? <div><h3>Related metadata matches</h3><p className="fine-print">{selected.related.map((item) => item.movie_title).join(" · ")}</p></div> : null}</section></div>}
  </>;
}
