import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _term_rows(report: dict[str, Any]) -> list[tuple[str, int, int]]:
    terms = report["totals"]["terms"]
    rows = [
        (name, int(stats["songs_mentioning"]), int(stats["mentions"]))
        for name, stats in terms.items()
    ]
    return sorted(rows, key=lambda row: row[2], reverse=True)


def plot_total_terms(report: dict[str, Any], out_dir: Path) -> Path:
    rows = _term_rows(report)
    names = [r[0] for r in rows]
    songs_mentioning = [r[1] for r in rows]
    mentions = [r[2] for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(11, max(6, len(rows) * 0.35)))
    ax.barh([i + 0.18 for i in y], mentions, height=0.35, label="mentions")
    ax.barh(
        [i - 0.18 for i in y],
        songs_mentioning,
        height=0.35,
        label="songs mentioning",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("Term mentions vs songs mentioning each term")
    ax.legend()
    return _save(fig, out_dir / "total_terms.png")


def plot_album_total_mentions(report: dict[str, Any], out_dir: Path) -> Path:
    rows: list[tuple[str, int, int]] = []
    for album in report["by_album"]:
        terms = album["terms"]
        mentions = sum(int(stats["mentions"]) for stats in terms.values())
        songs_mentioning = sum(
            int(stats["songs_mentioning"]) for stats in terms.values()
        )
        rows.append((album["album"], songs_mentioning, mentions))
    rows.sort(key=lambda row: row[2], reverse=True)

    names = [r[0] for r in rows]
    mentions = [r[2] for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(12, max(7, len(rows) * 0.32)))
    ax.barh(y, mentions)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("total term mentions")
    ax.set_title("Total tracked term mentions by album")
    return _save(fig, out_dir / "album_total_mentions.png")


def _parse_release_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def _album_release_dates(by_album_dir: Path) -> dict[str, datetime]:
    dates: dict[str, datetime] = {}
    for path in sorted(by_album_dir.glob("*.json")):
        report = _load_json(path)
        album = report.get("album")
        songs = report.get("songs")
        if not isinstance(album, str) or not isinstance(songs, list):
            continue
        song_dates = [
            release_date
            for song in songs
            if isinstance(song, dict)
            for release_date in [_parse_release_date(song.get("release_date"))]
            if release_date is not None
        ]
        if song_dates:
            dates[album] = min(song_dates)
    return dates


def plot_album_mentions_by_release_date(
    report: dict[str, Any], by_album_dir: Path, out_dir: Path
) -> Path:
    release_dates = _album_release_dates(by_album_dir)
    rows: list[tuple[datetime | None, str, int]] = []
    for album in report["by_album"]:
        name = str(album["album"])
        mentions = sum(int(stats["mentions"]) for stats in album["terms"].values())
        rows.append((release_dates.get(name), name, mentions))
    rows.sort(key=lambda row: (row[0] is None, row[0] or datetime.max, row[1]))

    labels = [
        f"{date.date().isoformat()}  {name}" if date is not None else f"unknown  {name}"
        for date, name, _mentions in rows
    ]
    mentions = [row[2] for row in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(12, max(7, len(rows) * 0.34)))
    ax.barh(y, mentions)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("total tracked term mentions")
    ax.set_title("Album term mentions sorted by release date")
    return _save(fig, out_dir / "album_mentions_by_release_date.png")


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


def plot_song_mentions_for_album(report_path: Path, out_dir: Path) -> Path | None:
    report = _load_json(report_path)
    songs = report.get("songs")
    if not isinstance(songs, list):
        return None

    rows: list[tuple[str, int]] = []
    for song in songs:
        terms = song.get("terms", {})
        rows.append(
            (
                str(song.get("title") or "Untitled"),
                sum(int(v) for v in terms.values()),
            )
        )

    names = [r[0] for r in rows]
    mentions = [r[1] for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(11, max(3.5, len(rows) * 0.4)))
    ax.barh(y, mentions)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("total tracked term mentions")
    ax.set_title(f"Song term mentions: {report.get('album', report_path.stem)}")
    return _save(fig, out_dir / f"{report_path.stem}.png")


def generate_graphs(
    report_path: Path = Path("reports/term_mentions.json"),
    by_album_dir: Path = Path("reports/by_album"),
    out_dir: Path = Path("reports/graphs"),
) -> list[Path]:
    report = _load_json(report_path)
    written = [
        plot_total_terms(report, out_dir),
        plot_album_total_mentions(report, out_dir),
        plot_album_mentions_by_release_date(report, by_album_dir, out_dir),
        plot_album_term_heatmap(report, out_dir),
    ]

    album_graph_dir = out_dir / "by_album"
    for album_report in sorted(by_album_dir.glob("*.json")):
        path = plot_song_mentions_for_album(album_report, album_graph_dir)
        if path is not None:
            written.append(path)
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
