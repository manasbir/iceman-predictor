import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List


@dataclass
class Term:
    name: str
    words: List[str]


@dataclass
class CatalogSong:
    title: str
    lyrics: str
    album: str | None
    release_date: datetime | None


@dataclass
class CatalogAlbum:
    """One on-disk album folder: display name and the tracks saved under it."""

    name: str
    songs: List[CatalogSong]


def load_albums_from_disk(base: Path | str = Path("songs/album")) -> List[CatalogAlbum]:
    """Load all ``*.txt`` JSON song payloads grouped by album subdirectory.

    Each file is expected to contain JSON with at least ``title``, ``lyrics``,
    optional ``album``, and optional ``release_date`` (ISO datetime string).
    The album grouping follows the immediate subdirectory name under ``base``.
    """
    root = Path(base)
    if not root.is_dir():
        return []

    albums: List[CatalogAlbum] = []
    for album_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        songs: List[CatalogSong] = []
        for path in sorted(album_dir.glob("*.txt")):
            data = json.loads(path.read_text(encoding="utf-8"))
            title = str(data.get("title") or path.stem)
            lyrics = str(data.get("lyrics") or "")
            album_name = data.get("album")
            if album_name is not None:
                album_name = str(album_name)
            rd_raw = data.get("release_date")
            release_date: datetime | None = None
            if isinstance(rd_raw, str) and rd_raw.strip():
                release_date = datetime.fromisoformat(
                    rd_raw.strip().replace("Z", "+00:00")
                )
            songs.append(
                CatalogSong(
                    title=title,
                    lyrics=lyrics,
                    album=album_name,
                    release_date=release_date,
                )
            )
        if songs:
            albums.append(CatalogAlbum(name=album_dir.name, songs=songs))
    return albums


def iter_catalog_songs(albums: Iterable[CatalogAlbum]) -> Iterator[CatalogSong]:
    for album in albums:
        yield from album.songs


def _count_word_mentions_in_text(lyrics: str, word: str) -> int:
    """Non-overlapping occurrence count for ``word`` in ``lyrics`` (same rules as contains)."""
    lw = lyrics.lower()
    ww = word.lower()
    if not ww:
        return 0
    if re.fullmatch(r"[a-z0-9']+", ww):
        pat = rf"(?<![a-z0-9']){re.escape(ww)}(?![a-z0-9'])"
        return len(re.findall(pat, lw))
    return lw.count(ww)


def _count_term_mentions_in_lyrics(lyrics: str, words: List[str]) -> int:
    if not words:
        return 0
    return sum(_count_word_mentions_in_text(lyrics, w) for w in words)


def term_name_to_words(terms: Iterable[Term]) -> dict[str, List[str]]:
    """First ``Term`` per ``name`` wins; empty names skipped."""
    name_to_words: dict[str, List[str]] = {}
    for t in terms:
        if not t.name or t.name in name_to_words:
            continue
        name_to_words[t.name] = [w for w in t.words if w]
    return name_to_words


def count_songs_mentioning_each_term(
    songs: Iterable[CatalogSong],
    terms: Iterable[Term],
) -> dict[str, int]:
    """Return how many distinct songs match at least one word of each term.

    Case-insensitive. Pure letter/digit/apostrophe search strings use
    word-boundary style matching; others use substring search.

    Keys are ``term.name``. If several terms share a name, only the first is
    used. Terms with no non-empty words contribute a count of zero.
    """
    name_to_words = term_name_to_words(terms)
    counts = {name: 0 for name in name_to_words}
    song_list = list(songs)

    for name, term_words in name_to_words.items():
        if not term_words:
            continue
        for song in song_list:
            if _count_term_mentions_in_lyrics(song.lyrics, term_words) > 0:
                counts[name] += 1
    return counts


def build_term_mention_report(
    albums: List[CatalogAlbum], terms: List[Term]
) -> dict[str, Any]:
    """Totals and per-album stats: songs that mention each term vs raw mention counts."""
    name_to_words = term_name_to_words(terms)
    term_names = list(name_to_words.keys())

    all_songs = list(iter_catalog_songs(albums))
    totals_terms: dict[str, dict[str, int]] = {
        name: {"songs_mentioning": 0, "mentions": 0} for name in term_names
    }

    for song in all_songs:
        for name, words in name_to_words.items():
            if not words:
                continue
            mentions = _count_term_mentions_in_lyrics(song.lyrics, words)
            totals_terms[name]["mentions"] += mentions
            if mentions > 0:
                totals_terms[name]["songs_mentioning"] += 1

    by_album: list[dict[str, Any]] = []
    for album in albums:
        row: dict[str, Any] = {
            "album": album.name,
            "song_count": len(album.songs),
            "terms": {},
        }
        for name, words in name_to_words.items():
            if not words:
                row["terms"][name] = {"songs_mentioning": 0, "mentions": 0}
                continue
            songs_with = 0
            mentions = 0
            for song in album.songs:
                m = _count_term_mentions_in_lyrics(song.lyrics, words)
                mentions += m
                if m > 0:
                    songs_with += 1
            row["terms"][name] = {
                "songs_mentioning": songs_with,
                "mentions": mentions,
            }
        by_album.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "album_count": len(albums),
            "song_count": len(all_songs),
            "terms": totals_terms,
        },
        "by_album": by_album,
    }


def save_term_mention_report(
    path: Path | str,
    albums: List[CatalogAlbum],
    terms: List[Term],
) -> Path:
    """Write JSON report (totals + per-album ``songs_mentioning`` and ``mentions``)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = build_term_mention_report(albums, terms)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def _safe_album_report_filename(album_name: str, used_slugs: dict[str, int]) -> str:
    """Filesystem-safe base name; disambiguate if sanitization collides."""
    bad = '\\/:*?"<>|\x00'
    slug = "".join("_" if c in bad else c for c in album_name).strip() or "album"
    n = used_slugs.get(slug, 0) + 1
    used_slugs[slug] = n
    return f"{slug}.json" if n == 1 else f"{slug}__{n}.json"


def build_album_song_term_mention_report(
    album: CatalogAlbum, terms: List[Term]
) -> dict[str, Any]:
    """Per-song mention counts (summed over each term's search strings)."""
    name_to_words = term_name_to_words(terms)
    songs_out: list[dict[str, Any]] = []
    for song in album.songs:
        rd = (
            song.release_date.isoformat()
            if song.release_date is not None
            else None
        )
        terms_counts: dict[str, int] = {}
        for term_name, words in name_to_words.items():
            terms_counts[term_name] = _count_term_mentions_in_lyrics(
                song.lyrics, words
            )
        songs_out.append(
            {
                "title": song.title,
                "release_date": rd,
                "terms": terms_counts,
            }
        )
    return {
        "album": album.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "songs": songs_out,
    }


def save_per_album_song_term_reports(
    albums: List[CatalogAlbum],
    terms: List[Term],
    out_dir: Path | str = Path("reports") / "by_album",
) -> list[Path]:
    """Write one JSON file per album: each song's hit counts per term."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    used_slugs: dict[str, int] = {}
    written: list[Path] = []
    for album in albums:
        fname = _safe_album_report_filename(album.name, used_slugs)
        path = base / fname
        payload = build_album_song_term_mention_report(album, terms)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


def main():
    terms = [
        # Term(name="iceman", words=["iceman"]),
        Term(name="ovo", words=["ovo"]),
        Term(name="drizzy", words=["drizzy"]),
        Term(name="diamond", words=["diamond"]),
        Term(name="mj", words=["mj", "jordan"]),
        # Term(name="21", words=["21"]),
        # Term(name="toronto", words=["toronto"]),
        Term(name="champagne", words=["champagne"]),
        Term(name="nike", words=["nike"]),
        Term(name="papi", words=["papi"]),
        Term(name="virgil", words=["virgil"]),
        Term(name="lebron", words=["lebron"]),
        Term(name="superbowl", words=["superbowl", "super bowl"]),
        Term(name="raptors", words=["raptors"]),
        # Term(name="crypto", words=["crypto", "bitcoin"]),
        Term(name="asap", words=["a$ap", "asap"]),
        Term(name="kendrick", words=["kendrick"]),
        Term(name="metro", words=["metro"]),
        Term(name="trump", words=["trump"]),
        Term(name="kdot", words=["k-dot", "k dot", "kdot"]),
        Term(name="rick ross", words=["rick ross"]),
        Term(name="compton", words=["compton"]),
        Term(name="pulitzer", words=["pulitzer"]),
        Term(name="kung fu kenny", words=["kung fu kenny"]),
        Term(name="batman", words=["batman", "bat man"]),
    ]

    albums = load_albums_from_disk()
    term_counts = count_songs_mentioning_each_term(iter_catalog_songs(albums), terms)
    report_path = save_term_mention_report(
        Path("reports") / "term_mentions.json",
        albums,
        terms,
    )
    print(f"Loaded {len(albums)} albums, {sum(len(a.songs) for a in albums)} songs")
    print(term_counts)
    print(f"Wrote {report_path}")
    per_album_paths = save_per_album_song_term_reports(albums, terms)
    by_album_dir = (
        per_album_paths[0].parent if per_album_paths else Path("reports/by_album")
    )
    print(
        f"Wrote {len(per_album_paths)} per-album song reports under {by_album_dir}/"
    )


if __name__ == "__main__":
    main()
