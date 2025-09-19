"""Interactive anime search, exploration, and recommendation tool.

This module wires the open Jikan REST API together with a SentenceTransformer
embedding model to provide fuzzy search, smart recommendations, and an
inspection UI built in Tkinter.  A sizeable amount of state is cached between
requests (e.g., embeddings, rating/tag catalogs) so we keep everything in a
single file where shared globals are easy to follow.
"""

import re
import webbrowser
from difflib import SequenceMatcher
from io import BytesIO
import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    from PIL import Image, ImageTk
except ImportError:  # optional for rich detail view
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # dependencies optional for recommendations
    SentenceTransformer = None  # type: ignore
    NearestNeighbors = None  # type: ignore

API_URL = "https://api.jikan.moe/v4/anime"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LIMIT = 25
TOP_RECO_PAGES = 2
TOP_RECO_LIMIT = 25
STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "season",
    "part",
    "movie",
    "special",
    "film",
    "ova",
    "rewrite",
    "remake",
}

embedding_model: Optional[SentenceTransformer] = None
embedding_error: Optional[str] = None

# We keep a shared recommendation corpus (plus fitted KNN model) so repeated
# searches feel instantaneous once the expensive lift has happened.
recommendation_corpus: List[Dict] = []
corpus_embeddings: Optional[np.ndarray] = None
corpus_knn: Optional[NearestNeighbors] = None
active_title_tags: List[str] = []

# Filter widgets are populated dynamically from API responses and the top-anime
# corpus.  These structures back the Tkinter controls.
rating_options: List[str] = []
tag_options: List[str] = []
rating_vars: Dict[str, tk.BooleanVar] = {}
rating_menu: Optional[tk.Menu] = None
rating_button: Optional[ttk.Menubutton] = None
tag_listbox: Optional[tk.Listbox] = None


def ensure_embedding_model() -> bool:
    """Load and cache the sentence embedding model if it has not been loaded."""
    global embedding_model, embedding_error
    if SentenceTransformer is None or NearestNeighbors is None:
        embedding_error = (
            "Recommendation libraries missing. Install 'sentence-transformers' "
            "and 'scikit-learn' to enable suggestions."
        )
        return False
    if embedding_model is not None:
        embedding_error = None
        return True
    try:
        status_var.set("Loading recommendation model... (first use may take a moment)")
        root.update_idletasks()
        embedding_model = SentenceTransformer(MODEL_NAME)
        embedding_error = None
        status_var.set("Recommendation model ready.")
        return True
    except Exception as exc:  # pragma: no cover - defensive
        embedding_error = f"Unable to load recommendation model: {exc}"
        status_var.set("Loaded search results without recommendations.")
        return False


def fetch_top_anime_page(page: int) -> List[Dict]:
    """Fetch a page of globally top-rated anime used to seed recommendations."""
    try:
        response = requests.get(
            "https://api.jikan.moe/v4/top/anime",
            params={"page": page, "limit": TOP_RECO_LIMIT},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return data if isinstance(data, list) else []
    except requests.RequestException:
        return []


def ensure_recommendation_corpus() -> bool:
    """Guarantee that the shared recommendation corpus and KNN index exist."""
    global recommendation_corpus, corpus_embeddings, corpus_knn, embedding_error

    if corpus_knn is not None and corpus_embeddings is not None and recommendation_corpus:
        return True

    if not ensure_embedding_model():
        return False

    seen_ids = {anime.get("mal_id") for anime in recommendation_corpus if anime.get("mal_id")}

    for page in range(1, TOP_RECO_PAGES + 1):
        page_data = fetch_top_anime_page(page)
        for anime in page_data:
            mal_id = anime.get("mal_id")
            if not mal_id or mal_id in seen_ids:
                continue
            if not anime.get("synopsis"):
                continue
            recommendation_corpus.append(anime)
            seen_ids.add(mal_id)

    if not recommendation_corpus:
        embedding_error = "Unable to load recommendation corpus."
        return False

    texts = [row_text(anime) for anime in recommendation_corpus]
    try:
        corpus_embeddings = embedding_model.encode(texts, normalize_embeddings=True)  # type: ignore[arg-type]
    except Exception as exc:
        embedding_error = f"Embedding error while preparing corpus: {exc}"
        recommendation_corpus.clear()
        corpus_embeddings = None
        corpus_knn = None
        return False

    neighbors = min(150, len(recommendation_corpus))
    if neighbors < 2:
        embedding_error = "Recommendation corpus too small."
        return False

    corpus_knn = NearestNeighbors(n_neighbors=neighbors, metric="cosine")  # type: ignore[call-arg]
    corpus_knn.fit(corpus_embeddings)

    update_filter_options_from_list(recommendation_corpus)
    return True


def row_text(anime: Dict) -> str:
    """Build semantic text purely from synopsis and tag-like metadata."""
    parts: List[str] = []
    for field in ("genres", "themes", "studios"):
        items = anime.get(field) or []
        names = [item.get("name") for item in items if isinstance(item, dict) and item.get("name")]
        parts.extend(names)
    synopsis = anime.get("synopsis") or ""
    return f"{synopsis} | {' '.join(parts)}".strip()


def pick_primary_index(results: List[Dict], query: str) -> int:
    """Return the index whose title best matches the raw query."""
    lowered = query.lower()
    for idx, anime in enumerate(results):
        title = anime.get("title") or ""
        if title.lower() == lowered:
            return idx
    return 0


def tokenize_title(title: str) -> List[str]:
    """Return cleaned, stop-word-free tokens for title comparisons."""
    tokens = re.findall(r"[A-Za-z0-9]+", title.lower())
    return [tok for tok in tokens if tok and tok not in STOP_WORDS]


def title_signature(title: str) -> str:
    """Produce a stable signature used to de-duplicate near-identical titles."""
    tokens = tokenize_title(title)
    if not tokens:
        return ""
    return " ".join(sorted(tokens))


def similar_title(primary: str, candidate: str) -> bool:
    """Return True when two titles are effectively the same series/edition."""
    if not primary or not candidate:
        return False
    primary_clean_tokens = tokenize_title(primary)
    candidate_clean_tokens = tokenize_title(candidate)
    if not primary_clean_tokens or not candidate_clean_tokens:
        return False

    primary_clean = " ".join(primary_clean_tokens)
    candidate_clean = " ".join(candidate_clean_tokens)

    if primary_clean == candidate_clean:
        return True
    if primary_clean in candidate_clean or candidate_clean in primary_clean:
        return True

    primary_set = set(primary_clean_tokens)
    candidate_set = set(candidate_clean_tokens)
    if primary_set and candidate_set:
        overlap = len(primary_set & candidate_set)
        smaller = min(len(primary_set), len(candidate_set))
        if smaller and overlap / smaller >= 0.8:
            return True

    ratio = SequenceMatcher(None, primary.lower(), candidate.lower()).ratio()
    return ratio >= 0.85


def split_query_titles(raw_query: str) -> List[str]:
    """Turn a comma/newline separated query into discrete search titles."""
    parts = re.split(r"[,\n]+", raw_query)
    return [part.strip() for part in parts if part.strip()]


def fetch_search_results(title: str, limit: int = DEFAULT_LIMIT) -> Tuple[List[Dict], Optional[str]]:
    """Query the search endpoint and return results along with an error string."""
    try:
        response = requests.get(
            API_URL,
            params={"q": title, "limit": limit},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return [], str(exc)

    results = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return [], "Unexpected response format."
    return results, None


def fetch_primary_anime(title: str, limit: int = 10) -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch one representative anime for a title, preferring exact matches."""
    results, error = fetch_search_results(title, limit=limit)
    if error:
        return None, error
    if not results:
        return None, None
    primary_index = pick_primary_index(results, title)
    return results[primary_index], None


def get_all_tags(anime: Dict) -> List[str]:
    """Collect every descriptive tag-like attribute associated with an anime."""
    tags: List[str] = []
    for field in ("genres", "themes", "demographics", "studios"):
        items = anime.get(field) or []
        names = [item.get("name") for item in items if isinstance(item, dict) and item.get("name")]
        tags.extend(names)
    return tags


def update_filter_options_from_list(anime_list: List[Dict]):
    """Refresh filter option lists based on the supplied anime metadata."""
    global rating_options, tag_options, rating_vars
    if not anime_list:
        return

    existing_ratings = set(rating_options)
    existing_tags = set(tag_options)

    for anime in anime_list:
        rating = anime.get("rating")
        if rating:
            existing_ratings.add(rating)
        for tag in get_all_tags(anime):
            existing_tags.add(tag)

    new_rating_options = sorted(existing_ratings)
    new_tag_options = sorted(existing_tags)

    if new_rating_options != rating_options:
        selected = {key for key, var in rating_vars.items() if var.get()}
        rating_options = new_rating_options
        rebuild_rating_menu(selected)

    if new_tag_options != tag_options:
        selected_tags = get_selected_tags()
        tag_options = new_tag_options
        rebuild_tag_listbox(selected_tags)


def rebuild_rating_menu(previous_selection: Optional[set] = None):
    """Rebuild the ratings multi-select menu while preserving selections."""
    global rating_menu, rating_vars, rating_button
    if rating_menu is None or rating_button is None:
        return

    if previous_selection is None:
        previous_selection = set()

    rating_menu.delete(0, tk.END)
    rating_vars = {}
    for option in rating_options:
        var = tk.BooleanVar(value=option in previous_selection)
        rating_vars[option] = var
        rating_menu.add_checkbutton(
            label=option,
            onvalue=True,
            offvalue=False,
            variable=var,
            command=update_rating_button_label,
        )
    update_rating_button_label()


def rebuild_tag_listbox(previous_selection: Optional[List[str]] = None):
    """Repopulate the tag listbox and restore previously selected items."""
    global tag_listbox
    if tag_listbox is None:
        return

    if previous_selection is None:
        previous_selection = []

    tag_listbox.delete(0, tk.END)
    for option in tag_options:
        tag_listbox.insert(tk.END, option)

    selection_lower = {tag.lower() for tag in previous_selection}
    for idx, option in enumerate(tag_options):
        if option.lower() in selection_lower:
            tag_listbox.selection_set(idx)


def update_rating_button_label():
    """Show a human-friendly summary of rating filter selections."""
    if rating_button is None:
        return
    selected = [opt for opt, var in rating_vars.items() if var.get()]
    if not selected:
        rating_button.config(text="Ratings: All")
    elif len(selected) == 1:
        rating_button.config(text=f"Ratings: {selected[0]}")
    else:
        rating_button.config(text=f"Ratings: {len(selected)} selected")


def get_selected_ratings() -> List[str]:
    """Return the list of ratings currently enabled in the filter."""
    return [opt for opt, var in rating_vars.items() if var.get()]


def get_selected_tags() -> List[str]:
    """Return the list of tag strings currently selected in the filter."""
    if tag_listbox is None:
        return []
    indices = tag_listbox.curselection()
    return [tag_listbox.get(i) for i in indices]


def parse_float(value: str) -> Optional[float]:
    """Safely convert a string to float, returning None when invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> Optional[int]:
    """Safely convert a string to int, returning None when invalid."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def passes_filters(anime: Dict) -> bool:
    """Return True when an anime satisfies all active user-defined filters."""
    score = anime.get("score")
    if score_min_var.get():
        min_score = parse_float(score_min_var.get())
        if min_score is not None:
            if score is None or score < min_score:
                return False
    if score_max_var.get():
        max_score = parse_float(score_max_var.get())
        if max_score is not None:
            if score is None or score > max_score:
                return False

    selected_ratings = get_selected_ratings()
    if selected_ratings:
        rating = anime.get("rating")
        if rating not in selected_ratings:
            return False

    selected_tags = {tag.lower() for tag in get_selected_tags()}
    if selected_tags:
        anime_tags = {tag.lower() for tag in get_all_tags(anime)}
        if not selected_tags.issubset(anime_tags):
            return False

    if year_min_var.get():
        min_year = parse_int(year_min_var.get())
        if min_year is not None:
            year = anime.get("year")
            if year is None or year < min_year:
                return False
    if year_max_var.get():
        max_year = parse_int(year_max_var.get())
        if max_year is not None:
            year = anime.get("year")
            if year is None or year > max_year:
                return False

    return True


def build_recommendations(anime_list: List[Dict], desired_count: int) -> List[Dict]:
    """Produce a filtered list of recommended anime for the supplied seeds."""
    if desired_count <= 0 or not anime_list:
        return []
    if not ensure_embedding_model():
        return []
    if not ensure_recommendation_corpus():
        return []

    update_filter_options_from_list(anime_list)

    texts: List[str] = []
    source_titles: List[str] = []
    source_ids = set()
    seen_signatures = set()

    for anime in anime_list:
        text = row_text(anime)
        if text:
            texts.append(text)
        title = anime.get("title") or ""
        if title:
            source_titles.append(title)
            sig = title_signature(title)
            if sig:
                seen_signatures.add(sig)
            else:
                seen_signatures.add(title.lower())
        mal_id = anime.get("mal_id")
        if mal_id:
            source_ids.add(mal_id)

    if not texts:
        return []

    try:
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)  # type: ignore[arg-type]
    except Exception as exc:
        global embedding_error
        embedding_error = f"Embedding error: {exc}"
        return []

    combined_embedding = np.mean(embeddings, axis=0, keepdims=True)

    if corpus_knn is None or corpus_embeddings is None:
        return []

    neighbor_count = min(max(desired_count * 4, 60), len(recommendation_corpus))
    _, indices = corpus_knn.kneighbors(combined_embedding, n_neighbors=neighbor_count)

    recommendations: List[Dict] = []
    for idx in indices[0]:
        candidate = recommendation_corpus[idx]
        candidate_id = candidate.get("mal_id")
        if candidate_id and candidate_id in source_ids:
            continue
        candidate_title = candidate.get("title") or ""
        if any(similar_title(title, candidate_title) for title in source_titles):
            continue
        candidate_sig = title_signature(candidate_title)
        if candidate_sig and candidate_sig in seen_signatures:
            continue
        if any(
            similar_title(existing.get("title") or "", candidate_title)
            for existing in recommendations
        ):
            continue
        if not passes_filters(candidate):
            continue
        recommendations.append(candidate)
        if candidate_sig:
            seen_signatures.add(candidate_sig)
        elif candidate_title:
            seen_signatures.add(candidate_title.lower())
        if len(recommendations) == desired_count:
            break
    return recommendations


def parse_recommendation_count(raw_value: str) -> int:
    """Clamp the requested recommendation count to a sane 1–50 window."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(50, value))


def register_clickable_title(anime: Dict) -> str:
    """Tag a result-title line so that it launches the detail window on click."""
    tag_name = f"title_{anime.get('mal_id', 'unknown')}_{len(active_title_tags)}"
    active_title_tags.append(tag_name)

    def handle_click(event, current_anime=anime):
        open_detail_window(current_anime)
        return "break"

    results_box.tag_config(tag_name, foreground="#1a73e8", underline=True)
    results_box.tag_bind(tag_name, "<Enter>", lambda _: results_box.config(cursor="hand2"))
    results_box.tag_bind(tag_name, "<Leave>", lambda _: results_box.config(cursor=""))
    results_box.tag_bind(tag_name, "<Button-1>", handle_click)
    return tag_name


def insert_anime_entry(anime: Dict):
    """Render a single anime entry into the results pane with metadata lines."""
    title = anime.get("title") or "Unknown Title"
    rating = anime.get("rating") or "N/A"
    score = anime.get("score")
    score_text = "N/A" if score is None else score
    episodes = anime.get("episodes")
    episodes_text = "N/A" if episodes is None else episodes
    genres = ", ".join(genre.get("name", "") for genre in anime.get("genres", [])) or "N/A"

    tag_name = register_clickable_title(anime)
    results_box.insert(tk.END, title, tag_name)
    results_box.insert(tk.END, "\n")
    results_box.insert(tk.END, f"  Rating: {rating}\n")
    results_box.insert(tk.END, f"  Score: {score_text}\n")
    results_box.insert(tk.END, f"  Episodes: {episodes_text}\n")
    results_box.insert(tk.END, f"  Genres: {genres}\n\n")


def open_trailer(trailer_url: str):
    """Open the trailer link in the user’s browser if one was provided."""
    if trailer_url:
        webbrowser.open(trailer_url, new=2)


def open_detail_window(anime: Dict):
    """Pop up a detail window with artwork, metadata, tags, and synopsis."""
    window = tk.Toplevel(root)
    window.title(anime.get("title") or "Anime Details")
    window.minsize(360, 400)

    container = ttk.Frame(window, padding=12)
    container.pack(fill=tk.BOTH, expand=True)

    title_label = ttk.Label(
        container,
        text=anime.get("title") or "Unknown Title",
        font=("TkDefaultFont", 14, "bold"),
        wraplength=420,
    )
    title_label.pack(anchor=tk.W)

    trailer_info = anime.get("trailer") or {}
    trailer_url = trailer_info.get("url") or trailer_info.get("youtube_id")
    if trailer_url and trailer_info.get("youtube_id") and not trailer_info.get("url"):
        trailer_url = f"https://www.youtube.com/watch?v={trailer_info['youtube_id']}"

    image_displayed = False
    image_url = (
        anime.get("images", {})
        .get("jpg", {})
        .get("large_image_url")
        or anime.get("images", {})
        .get("jpg", {})
        .get("image_url")
    )

    if image_url and Image and ImageTk:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.thumbnail((320, 480))
            photo = ImageTk.PhotoImage(image)
            image_label = ttk.Label(container, image=photo, cursor="hand2" if trailer_url else "")
            image_label.image = photo  # keep reference
            image_label.pack(pady=(10, 10))
            if trailer_url:
                image_label.bind(
                    "<Button-1>", lambda _event: open_trailer(trailer_url)  # open trailer in browser
                )
            image_displayed = True
        except Exception:
            image_displayed = False

    if not image_displayed:
        placeholder = ttk.Label(container, text="Image unavailable.")
        if trailer_url:
            placeholder.bind("<Button-1>", lambda _event: open_trailer(trailer_url))
            placeholder.config(cursor="hand2")
        placeholder.pack(pady=(10, 10))

    if trailer_url:
        trailer_link = ttk.Label(
            container,
            text="Watch Trailer",
            foreground="#1a73e8",
            cursor="hand2",
        )
        trailer_link.pack(anchor=tk.W, pady=(0, 8))
        trailer_link.bind("<Button-1>", lambda _event: open_trailer(trailer_url))

    details = []
    rating = anime.get("rating")
    if rating:
        details.append(f"Rating: {rating}")
    score = anime.get("score")
    if score is not None:
        details.append(f"Score: {score}")
    episodes = anime.get("episodes")
    if episodes is not None:
        details.append(f"Episodes: {episodes}")
    status = anime.get("status")
    if status:
        details.append(f"Status: {status}")
    year = anime.get("year")
    if year:
        details.append(f"Year: {year}")

    if details:
        ttk.Label(
            container,
            text="\n".join(details),
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor=tk.W, pady=(0, 8))

    tag_fields = get_all_tags(anime)
    if tag_fields:
        ttk.Label(
            container,
            text="Tags: " + ", ".join(tag_fields),
            justify=tk.LEFT,
            wraplength=420,
        ).pack(anchor=tk.W, pady=(0, 8))

    synopsis = anime.get("synopsis") or "No description available."
    synopsis_label = ttk.Label(
        container,
        text="Description:",
        font=("TkDefaultFont", 10, "bold"),
    )
    synopsis_label.pack(anchor=tk.W)

    synopsis_box = scrolledtext.ScrolledText(container, wrap=tk.WORD, height=12)
    synopsis_box.insert(tk.END, synopsis)
    synopsis_box.config(state=tk.DISABLED)
    synopsis_box.pack(fill=tk.BOTH, expand=True, pady=(4, 0))


def search_anime(event=None):
    """Entry point for the Search button/Return key; orchestrates the flow."""
    global embedding_error

    raw_query = search_var.get().strip()
    if not raw_query:
        status_var.set("Please enter an anime title to search.")
        return

    titles = split_query_titles(raw_query)
    if not titles:
        status_var.set("Please enter at least one valid title.")
        return

    desired_count = parse_recommendation_count(rec_count_var.get())

    embedding_error = None
    search_button.config(state=tk.DISABLED)
    status_var.set("Searching...")
    root.update_idletasks()

    results_box.config(state=tk.NORMAL)
    results_box.delete("1.0", tk.END)
    for tag in active_title_tags:
        results_box.tag_delete(tag)
    active_title_tags.clear()

    recommendations: List[Dict] = []

    if len(titles) == 1:
        title = titles[0]
        results, error = fetch_search_results(title, limit=DEFAULT_LIMIT)
        update_filter_options_from_list(results)
        if error:
            results_box.insert(tk.END, f"Error fetching data: {error}\n")
            status_var.set("Unable to fetch results.")
        elif not results:
            results_box.insert(tk.END, "No results found. Try another search term.\n")
            status_var.set("No results found.")
        else:
            primary_index = pick_primary_index(results, title)
            primary_anime = results[primary_index]
            insert_anime_entry(primary_anime)

            recommendations = build_recommendations([primary_anime], desired_count)
            if recommendations:
                results_box.insert(
                    tk.END,
                    f"Recommended Similar Titles (top {desired_count}):\n\n",
                )
                for rec in recommendations:
                    insert_anime_entry(rec)
            elif embedding_error:
                results_box.insert(tk.END, f"\nRecommendation note: {embedding_error}\n")

            status_var.set(
                f"Showing best match with {len(recommendations)} recommendation(s)."
            )
    else:
        matches: List[Dict] = []
        missing_titles: List[str] = []
        error_messages: List[str] = []

        for title in titles:
            anime, error = fetch_primary_anime(title)
            if error:
                error_messages.append(f"{title}: {error}")
            if anime:
                matches.append(anime)
            elif not error:
                missing_titles.append(title)

        update_filter_options_from_list(matches)

        if matches:
            results_box.insert(tk.END, "Matches used:\n\n")
            for anime in matches:
                insert_anime_entry(anime)

        if missing_titles:
            results_box.insert(
                tk.END,
                "Titles not found: " + ", ".join(missing_titles) + "\n\n",
            )
        if error_messages:
            results_box.insert(
                tk.END,
                "Errors:\n" + "\n".join(error_messages) + "\n\n",
            )

        if matches:
            recommendations = build_recommendations(matches, desired_count)
            if recommendations:
                results_box.insert(
                    tk.END,
                    f"Recommended Similar Titles (top {desired_count}):\n\n",
                )
                for rec in recommendations:
                    insert_anime_entry(rec)
            elif embedding_error:
                results_box.insert(tk.END, f"\nRecommendation note: {embedding_error}\n")

            status_var.set(
                f"Using {len(matches)} of {len(titles)} titles; "
                f"showing {len(recommendations)} recommendation(s)."
            )
        else:
            status_var.set("No matches found for provided titles.")

    results_box.config(state=tk.DISABLED)
    results_box.config(cursor="")
    search_button.config(state=tk.NORMAL)


root = tk.Tk()
root.title("Anime Search")

search_var = tk.StringVar()
rec_count_var = tk.StringVar(value="3")
status_var = tk.StringVar(value="Enter an anime title to begin.")
score_min_var = tk.StringVar()
score_max_var = tk.StringVar()
year_min_var = tk.StringVar()
year_max_var = tk.StringVar()

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

input_frame = ttk.Frame(main_frame)
input_frame.pack(fill=tk.X)

search_entry = ttk.Entry(input_frame, textvariable=search_var)
search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
search_entry.bind("<Return>", search_anime)

search_button = ttk.Button(input_frame, text="Search", command=search_anime)
search_button.pack(side=tk.RIGHT)

options_frame = ttk.Frame(main_frame)
options_frame.pack(fill=tk.X, pady=(8, 0))

ttk.Label(options_frame, text="Recommendations:").pack(side=tk.LEFT)
rec_spinbox = ttk.Spinbox(
    options_frame,
    from_=1,
    to=50,
    textvariable=rec_count_var,
    width=5,
)
rec_spinbox.pack(side=tk.LEFT, padx=(6, 0))

filters_frame = ttk.LabelFrame(main_frame, text="Filters", padding=8)
filters_frame.pack(fill=tk.BOTH, expand=False, pady=(8, 0))

score_frame = ttk.Frame(filters_frame)
score_frame.pack(fill=tk.X, pady=(0, 6))

ttk.Label(score_frame, text="Score min:").grid(row=0, column=0, sticky=tk.W)
score_min_entry = ttk.Entry(score_frame, textvariable=score_min_var, width=6)
score_min_entry.grid(row=0, column=1, padx=(4, 12))

ttk.Label(score_frame, text="Score max:").grid(row=0, column=2, sticky=tk.W)
score_max_entry = ttk.Entry(score_frame, textvariable=score_max_var, width=6)
score_max_entry.grid(row=0, column=3, padx=(4, 12))

ttk.Label(score_frame, text="Year min:").grid(row=0, column=4, sticky=tk.W)
year_min_entry = ttk.Entry(score_frame, textvariable=year_min_var, width=6)
year_min_entry.grid(row=0, column=5, padx=(4, 12))

ttk.Label(score_frame, text="Year max:").grid(row=0, column=6, sticky=tk.W)
year_max_entry = ttk.Entry(score_frame, textvariable=year_max_var, width=6)
year_max_entry.grid(row=0, column=7, padx=(4, 0))

rating_frame = ttk.Frame(filters_frame)
rating_frame.pack(fill=tk.X, pady=(0, 6))

ttk.Label(rating_frame, text="Ratings:").pack(side=tk.LEFT)

rating_button = ttk.Menubutton(rating_frame, text="Ratings: All")
rating_button.pack(side=tk.LEFT, padx=(6, 0))

rating_menu = tk.Menu(rating_button, tearoff=False)
rating_button["menu"] = rating_menu

update_rating_button_label()

# Tags list
TagListFrame = ttk.Frame(filters_frame)
TagListFrame.pack(fill=tk.BOTH, expand=True)

ttk.Label(TagListFrame, text="Tags:").pack(anchor=tk.W)

tag_list_container = ttk.Frame(TagListFrame)
tag_list_container.pack(fill=tk.BOTH, expand=True)

tag_scrollbar = ttk.Scrollbar(tag_list_container, orient=tk.VERTICAL)
tag_listbox = tk.Listbox(
    tag_list_container,
    selectmode=tk.MULTIPLE,
    height=6,
    exportselection=False,
    yscrollcommand=tag_scrollbar.set,
)
tag_scrollbar.config(command=tag_listbox.yview)
tag_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
tag_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

results_box = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=20, state=tk.DISABLED)
results_box.pack(fill=tk.BOTH, expand=True, pady=10)

status_label = ttk.Label(main_frame, textvariable=status_var)
status_label.pack(fill=tk.X)

search_entry.focus()

root.mainloop()
