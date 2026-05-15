import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.figure import Figure


LINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
    "#e6550d",
    "#9ecae1",
    "#a1d99b",
    "#bcbddc",
    "#fdae6b",
]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _term_rows(report: dict[str, Any]) -> list[tuple[str, int, int]]:
    terms = report["totals"]["terms"]
    rows = [
        (name, int(stats["songs_mentioning"]), int(stats["mentions"]))
        for name, stats in terms.items()
    ]
    return sorted(rows, key=lambda row: row[2], reverse=True)


def plot_album_term_heatmap(report: dict[str, Any], out_dir: Path) -> Path:
    term_names = [name for name, _, mentions in _term_rows(report) if mentions > 0]
    albums = report["by_album"]
    album_names = [album["album"] for album in albums]
    matrix = [
        [int(album["terms"][term]["mentions"]) for term in term_names]
        for album in albums
    ]

    fig, ax = plt.subplots(
        figsize=(max(10, len(term_names) * 0.45), max(8, len(album_names) * 0.28))
    )
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(term_names)))
    ax.set_xticklabels(term_names, rotation=45, ha="right")
    ax.set_yticks(range(len(album_names)))
    ax.set_yticklabels(album_names)
    ax.set_title("Term mentions by album")
    fig.colorbar(image, ax=ax, label="mentions")
    return _save(fig, out_dir / "album_term_mentions_heatmap.png")


def _parse_release_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


ReleaseRow = tuple[datetime, str, int, dict[str, int], dict[str, int]]


def _terms_from_song(song: dict[str, Any]) -> dict[str, int]:
    terms = song.get("terms")
    if not isinstance(terms, dict):
        return {}
    return {str(term): int(count) for term, count in terms.items()}


def _release_rows(
    by_album_dir: Path,
) -> tuple[list[str], list[ReleaseRow]]:
    term_names: list[str] = []
    rows: list[ReleaseRow] = []

    for report_path in sorted(by_album_dir.glob("*.json")):
        report = _load_json(report_path)
        songs = report.get("songs")
        if not isinstance(songs, list) or not songs:
            continue

        album_name = str(report.get("album") or report_path.stem)
        if album_name == "None":
            for song in songs:
                if not isinstance(song, dict):
                    continue
                release_date = _parse_release_date(song.get("release_date"))
                if release_date is None:
                    continue
                term_counts = _terms_from_song(song)
                for term in term_counts:
                    if term not in term_names:
                        term_names.append(term)
                song_counts = {
                    term: 1 for term, count in term_counts.items() if count > 0
                }
                title = str(song.get("title") or "Untitled single")
                rows.append((release_date, title, 1, term_counts, song_counts))
            continue

        dates: list[datetime] = []
        term_counts: dict[str, int] = {}
        term_song_counts: dict[str, int] = {}
        for song in songs:
            if not isinstance(song, dict):
                continue
            release_date = _parse_release_date(song.get("release_date"))
            if release_date is not None:
                dates.append(release_date)

            for term_name, count in _terms_from_song(song).items():
                if term_name not in term_names:
                    term_names.append(term_name)
                term_counts[term_name] = term_counts.get(term_name, 0) + count
                if count > 0:
                    term_song_counts[term_name] = (
                        term_song_counts.get(term_name, 0) + 1
                    )

        if dates:
            rows.append((min(dates), album_name, len(songs), term_counts, term_song_counts))

    return term_names, sorted(rows, key=lambda row: (row[0], row[1]))


def plot_terms_by_release_date_line(by_album_dir: Path, out_dir: Path) -> Path:
    term_names, rows = _release_rows(by_album_dir)
    dates = [row[0] for row in rows]
    labels = [f"{row[0].date()}\n{row[1]}" for row in rows]

    fig, ax = plt.subplots(figsize=(max(13, len(rows) * 0.55), 8))
    for i, term in enumerate(term_names):
        values = [term_counts.get(term, 0) for _, _, _, term_counts, _ in rows]
        ax.plot(
            dates,
            values,
            color=LINE_COLORS[i % len(LINE_COLORS)],
            linestyle=LINE_STYLES[(i // len(LINE_COLORS)) % len(LINE_STYLES)],
            marker=MARKERS[i % len(MARKERS)],
            linewidth=1.7,
            markersize=3.5,
            label=term,
        )

    ax.set_xticks(dates)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("mentions in album")
    ax.set_title("Term mentions by album release date")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        loc="upper left",
        ncols=1,
        title="terms",
    )
    return _save(fig, out_dir / "album_mentions_by_release_date.png")


def plot_term_song_percent_by_release_date(by_album_dir: Path, out_dir: Path) -> Path:
    term_names, rows = _release_rows(by_album_dir)
    dates = [row[0].date().isoformat() for row in rows]
    labels = [row[1] for row in rows]

    fig = go.Figure()
    for i, term in enumerate(term_names):
        values = [
            (term_song_counts.get(term, 0) / song_count) * 100
            for _, _, song_count, _, term_song_counts in rows
        ]
        customdata = [
            [label, term_song_counts.get(term, 0), song_count]
            for _, label, song_count, _, term_song_counts in rows
        ]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Release date: %{x}<br>"
                    f"Term: {term}<br>"
                    "Songs mentioning: %{customdata[1]} / %{customdata[2]}<br>"
                    "% of songs: %{y:.1f}%<extra></extra>"
                ),
                line={
                    "color": LINE_COLORS[i % len(LINE_COLORS)],
                    "dash": ["solid", "dash", "dashdot", "dot"][
                        (i // len(LINE_COLORS)) % 4
                    ],
                    "width": 2,
                },
                marker={"size": 7},
                mode="lines+markers",
                name=term,
            )
        )

    fig.update_layout(
        title="Songs mentioning each term by album release date",
        xaxis={
            "title": "release date",
            "tickangle": -65,
            "tickmode": "array",
            "tickvals": dates,
            "ticktext": [f"{date}<br>{label}" for date, label in zip(dates, labels)],
        },
        yaxis={"title": "% of songs mentioning term", "range": [0, 100]},
        hovermode="closest",
        legend={"title": {"text": "terms"}},
        margin={"b": 230, "r": 260, "t": 70},
        template="plotly_white",
    )

    out = out_dir / "album_song_percent_by_release_date.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def plot_term_binary_by_release_date(by_album_dir: Path, out_dir: Path) -> Path:
    term_names, rows = _release_rows(by_album_dir)
    dates = [row[0].date().isoformat() for row in rows]
    labels = [row[1] for row in rows]

    fig = go.Figure()
    for i, term in enumerate(term_names):
        values = [
            1 if term_song_counts.get(term, 0) > 0 else 0
            for _, _, _, _, term_song_counts in rows
        ]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                customdata=labels,
                hovertemplate=(
                    "<b>%{customdata}</b><br>"
                    "Release date: %{x}<br>"
                    f"Term: {term}<br>"
                    "Mentioned: %{y}<extra></extra>"
                ),
                line={
                    "color": LINE_COLORS[i % len(LINE_COLORS)],
                    "dash": ["solid", "dash", "dashdot", "dot"][
                        (i // len(LINE_COLORS)) % 4
                    ],
                    "width": 2,
                },
                marker={"size": 7},
                mode="lines+markers",
                name=term,
            )
        )

    fig.update_layout(
        title="Whether each term is mentioned by release",
        xaxis={
            "title": "release date",
            "tickangle": -65,
            "tickmode": "array",
            "tickvals": dates,
            "ticktext": [f"{date}<br>{label}" for date, label in zip(dates, labels)],
        },
        yaxis={
            "title": "mentioned",
            "tickmode": "array",
            "tickvals": [0, 1],
            "ticktext": ["not mentioned", "mentioned"],
            "range": [-0.1, 1.1],
        },
        hovermode="closest",
        legend={"title": {"text": "terms"}},
        margin={"b": 230, "r": 260, "t": 70},
        template="plotly_white",
    )

    out = out_dir / "album_term_binary_by_release_date.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def generate_graphs(
    report_path: Path = Path("reports/term_mentions.json"),
    by_album_dir: Path = Path("reports/by_album"),
    out_dir: Path = Path("reports/graphs"),
) -> list[Path]:
    report = _load_json(report_path)
    written = [
        plot_album_term_heatmap(report, out_dir),
        plot_terms_by_release_date_line(by_album_dir, out_dir),
        plot_term_song_percent_by_release_date(by_album_dir, out_dir),
        plot_term_binary_by_release_date(by_album_dir, out_dir),
    ]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate graphs from term reports.")
    parser.add_argument("--report", type=Path, default=Path("reports/term_mentions.json"))
    parser.add_argument("--by-album-dir", type=Path, default=Path("reports/by_album"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/graphs"))
    args = parser.parse_args()

    written = generate_graphs(args.report, args.by_album_dir, args.out_dir)
    print(f"Wrote {len(written)} graph files under {args.out_dir}")


if __name__ == "__main__":
    main()
