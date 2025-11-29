"""
Tkinter Pokédex
- Uses PokéAPI (https://pokeapi.co/)
- Shows: Name, Types, Weight (kg), Height (m), Description
- Simple "real Pokédex" vibe UI with red shell, white screen, and type badges
- Search by name or ID, plus Prev/Next and Random buttons
Requirements:
    pip install requests pillow
"""
import threading
import random
import sys
import io
import re
from dataclasses import dataclass
from typing import List, Optional

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    print("Tkinter not available:", e)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Missing dependency: requests\nInstall with: pip install requests")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None
    ImageTk = None

POKEAPI_BASE = "https://pokeapi.co/api/v2"
MAX_POKEMON_ID = 1025  # update as Pokédex grows

# Type color palette (roughly aligned with common type colors)
TYPE_COLORS = {
    "normal":"#A8A77A","fire":"#EE8130","water":"#6390F0","electric":"#F7D02C","grass":"#7AC74C",
    "ice":"#96D9D6","fighting":"#C22E28","poison":"#A33EA1","ground":"#E2BF65","flying":"#A98FF3",
    "psychic":"#F95587","bug":"#A6B91A","rock":"#B6A136","ghost":"#735797","dragon":"#6F35FC",
    "dark":"#705746","steel":"#B7B7CE","fairy":"#D685AD"
}

@dataclass
class Pokemon:
    id: int
    name: str
    types: List[str]
    height_m: float
    weight_kg: float
    description: str
    image_url: Optional[str]
    evolves_from: Optional[str]
    evolves_to: List[str]


def normalize_search_term(term: str) -> str:
    """Normalize human input so the PokéAPI can understand the identifier."""
    normalized = term.strip().lower()
    replacements = {
        " ": "-",
        "'": "",
        ".": "",
        "é": "e",
        "♀": "-f",
        "♂": "-m",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _find_species_node(chain: dict, target: str) -> Optional[dict]:
    """Locate the node for the requested species within an evolution chain."""
    if not chain:
        return None
    if chain.get("species", {}).get("name") == target:
        return chain
    for child in chain.get("evolves_to", []):
        node = _find_species_node(child, target)
        if node:
            return node
    return None


def _find_parent_species(chain: dict, target: str, parent: Optional[dict] = None) -> Optional[dict]:
    """Find the parent node (previous evolution) for the target species."""
    if not chain:
        return None
    if chain.get("species", {}).get("name") == target:
        return parent
    for child in chain.get("evolves_to", []):
        result = _find_parent_species(child, target, chain)
        if result:
            return result
    return None


def fetch_pokemon(identifier: str) -> Pokemon:
    """Fetch Pokémon core data and species info for description."""
    # 1) /pokemon for stats/types/sprite
    resp = requests.get(f"{POKEAPI_BASE}/pokemon/{identifier.strip().lower()}")
    if resp.status_code != 200:
        raise ValueError(f"Pokémon '{identifier}' not found (HTTP {resp.status_code}).")
    pdata = resp.json()

    pid = pdata["id"]
    name = pdata["name"].capitalize()
    types = [t["type"]["name"].capitalize() for t in pdata["types"]]
    height_m = round(pdata["height"] / 10.0, 2)  # decimeters -> meters
    weight_kg = round(pdata["weight"] / 10.0, 2) # hectograms -> kilograms
    # Prefer official artwork if available
    image_url = (
        pdata.get("sprites", {})
            .get("other", {})
            .get("official-artwork", {})
            .get("front_default")
        or pdata.get("sprites", {}).get("front_default")
    )

    # 2) /pokemon-species for English flavor text
    sresp = requests.get(f"{POKEAPI_BASE}/pokemon-species/{pid}")
    desc = "No description available."
    evolves_from = None
    evolves_to: List[str] = []
    if sresp.status_code == 200:
        sdata = sresp.json()
        # gather English flavor text entries and pick the longest unique snippet
        english_entries: List[str] = []
        for entry in sdata.get("flavor_text_entries", []):
            if entry.get("language", {}).get("name") == "en":
                ftxt = entry.get("flavor_text", "")
                clean = re.sub(r"\s+", " ", ftxt.replace("\f", " ").replace("\n", " ")).strip()
                if clean and clean not in english_entries:
                    english_entries.append(clean)
        if english_entries:
            desc = max(english_entries, key=len)

        evolves_from_species = sdata.get("evolves_from_species")
        if evolves_from_species:
            evolves_from = evolves_from_species.get("name", "").capitalize() or None

        chain_url = sdata.get("evolution_chain", {}).get("url")
        if chain_url:
            chain_resp = requests.get(chain_url)
            if chain_resp.status_code == 200:
                chain_data = chain_resp.json().get("chain", {})
                species_name = pdata.get("species", {}).get("name", "")
                if species_name and chain_data:
                    node = _find_species_node(chain_data, species_name)
                    if node:
                        evolves_to = [child.get("species", {}).get("name", "").capitalize() for child in node.get("evolves_to", []) if child.get("species")]
                    parent_node = _find_parent_species(chain_data, species_name)
                    if parent_node:
                        parent_name = parent_node.get("species", {}).get("name", "")
                        if parent_name:
                            evolves_from = parent_name.capitalize()

    return Pokemon(
        id=pid,
        name=name,
        types=types,
        height_m=height_m,
        weight_kg=weight_kg,
        description=desc,
        image_url=image_url,
        evolves_from=evolves_from,
        evolves_to=evolves_to,
    )

def pil_image_from_url(url: str, size=(280, 280)) -> Optional[ImageTk.PhotoImage]:
    if not PIL_AVAILABLE or not url:
        return None
    try:
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        # contain within box while preserving aspect
        img.thumbnail(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

class PokedexApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pokédex")
        self.geometry("430x680")
        self.resizable(False, False)

        # Pokedex shell colors
        self.configure(bg="#D62828")  # red shell

        self._build_ui()

    def _build_ui(self):
        # Top emblem / title
        header = tk.Frame(self, bg="#D62828")
        header.pack(fill="x", pady=(12, 4))

        title = tk.Label(header, text="Pokédex", font=("Helvetica", 20, "bold"), fg="white", bg="#D62828")
        title.pack()

        # Screen bezel
        bezel = tk.Frame(self, bg="#111111", bd=0, highlightthickness=0)
        bezel.pack(padx=18, pady=8, fill="both", expand=False)

        # Inner screen
        self.screen = tk.Frame(bezel, bg="white", width=380, height=520, bd=0, highlightthickness=0)
        self.screen.pack(padx=6, pady=6)
        self.screen.pack_propagate(False)

        # Greeting + how-to message the first time users open the Pokédex
        self.greeting_label = tk.Label(
            self.screen,
            text="Welcome, Trainer!",
            font=("Helvetica", 14, "bold"),
            fg="#D62828",
            bg="white"
        )
        self.greeting_label.pack(pady=(12, 4))

        self.instructions_label = tk.Label(
            self.screen,
            text="Enter a Pokémon name or Pokédex number, then press Search or use the navigation buttons below.",
            font=("Helvetica", 10),
            wraplength=320,
            justify="center",
            fg="#444444",
            bg="white"
        )
        self.instructions_label.pack(pady=(0, 12))

        # Sprite area
        self.sprite_label = tk.Label(self.screen, bg="white")
        self.sprite_label.pack(pady=(4, 6))

        # Name + ID
        self.name_id_label = tk.Label(self.screen, text="", font=("Helvetica", 16, "bold"), bg="white")
        self.name_id_label.pack(pady=(2, 6))

        # Types row
        self.types_frame = tk.Frame(self.screen, bg="white")
        self.types_frame.pack(pady=(0, 8))

        # Quick stats
        stats_frame = tk.Frame(self.screen, bg="white")
        stats_frame.pack(pady=(0, 8))

        self.height_var = tk.StringVar(value="—")
        self.weight_var = tk.StringVar(value="—")
        tk.Label(stats_frame, text="Height:", font=("Helvetica", 11, "bold"), bg="white", fg="#111111")\
            .grid(row=0, column=0, sticky="e", padx=(0, 4))
        tk.Label(stats_frame, textvariable=self.height_var, bg="white", fg="#111111")\
            .grid(row=0, column=1, sticky="w")
        tk.Label(stats_frame, text="Weight:", font=("Helvetica", 11, "bold"), bg="white", fg="#111111")\
            .grid(row=0, column=2, sticky="e", padx=(18, 4))
        tk.Label(stats_frame, textvariable=self.weight_var, bg="white", fg="#111111")\
            .grid(row=0, column=3, sticky="w")

        # Evolution info
        evo_frame = tk.Frame(self.screen, bg="white")
        evo_frame.pack(pady=(0, 8), padx=8, fill="x")

        tk.Label(evo_frame, text="Evolves from:", font=("Helvetica", 11, "bold"), bg="white")\
            .grid(row=0, column=0, sticky="e", padx=(0, 6), pady=2)
        self.evolves_from_var = tk.StringVar(value="—")
        tk.Label(evo_frame, textvariable=self.evolves_from_var, bg="white", anchor="w")\
            .grid(row=0, column=1, sticky="w", pady=2)

        tk.Label(evo_frame, text="Evolves into:", font=("Helvetica", 11, "bold"), bg="white")\
            .grid(row=1, column=0, sticky="e", padx=(0, 6), pady=2)
        self.evolves_to_var = tk.StringVar(value="—")
        tk.Label(evo_frame, textvariable=self.evolves_to_var, bg="white", anchor="w", wraplength=250, justify="left")\
            .grid(row=1, column=1, sticky="w", pady=2)

        # Description (wrap)
        self.desc_text = tk.Text(
            self.screen,
            width=42,
            height=6,
            wrap="word",
            bd=0,
            padx=8,
            pady=8,
            bg="#FAFAFA",
            fg="#212121"
        )
        self.desc_text.configure(state="disabled")
        self.desc_text.tag_configure("content", foreground="#212121")
        self.desc_text.pack(padx=6, pady=(2, 10))

        # Controls (search, prev/next/random)
        controls = tk.Frame(self, bg="#D62828")
        controls.pack(fill="x", pady=(4, 8), padx=12)

        controls_inner = tk.Frame(controls, bg="#D62828")
        controls_inner.pack(anchor="center")

        self.search_entry = ttk.Entry(controls_inner, width=24)
        self.search_entry.grid(row=0, column=0, padx=(0, 8))
        self.search_entry.bind("<Return>", lambda e: self.search())

        self.search_btn = ttk.Button(controls_inner, text="Search", command=self.search)
        self.search_btn.grid(row=0, column=1, padx=(0, 6))

        self.prev_btn = ttk.Button(controls_inner, text="◀ Prev", command=self.prev_pokemon, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=2, padx=(6, 4))

        self.next_btn = ttk.Button(controls_inner, text="Next ▶", command=self.next_pokemon, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=3, padx=(4, 6))

        self.rand_btn = ttk.Button(controls_inner, text="Random", command=self.random_pokemon)
        self.rand_btn.grid(row=0, column=4)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self, textvariable=self.status_var, anchor="w", bg="#B51D1D", fg="white")
        status.pack(fill="x", side="bottom")

        # Keep reference for PhotoImage
        self._sprite_photo = None
        self.current_id: Optional[int] = None
        self._welcome_visible = True

        # ttk theme tweaks for tighter look
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

    def set_status(self, msg: str):
        self.status_var.set(msg)
        self.update_idletasks()

    def prev_pokemon(self):
        if self.current_id is None:
            return
        new_id = self.current_id - 1 if self.current_id > 1 else MAX_POKEMON_ID
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, str(new_id))
        self.search()

    def next_pokemon(self):
        if self.current_id is None:
            return
        new_id = self.current_id + 1 if self.current_id < MAX_POKEMON_ID else 1
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, str(new_id))
        self.search()

    def random_pokemon(self):
        rid = random.randint(1, MAX_POKEMON_ID)
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, str(rid))
        self.search()

    def search(self):
        raw_identifier = self.search_entry.get().strip()
        if not raw_identifier:
            return
        # Allow entries like "#025" and normalize numeric IDs to plain integers, while preparing names for the API
        normalized = raw_identifier.lstrip("#").strip()
        lookup_identifier = normalized
        if normalized.isdigit():
            normalized = str(int(normalized))
            self.search_entry.delete(0, tk.END)
            self.search_entry.insert(0, normalized)
            lookup_identifier = normalized
        else:
            lookup_identifier = normalize_search_term(normalized)
        self.set_status("Fetching…")
        # fetch off the UI thread
        threading.Thread(target=self._do_fetch, args=(lookup_identifier,), daemon=True).start()

    def _do_fetch(self, identifier: str):
        try:
            p = fetch_pokemon(identifier)
            self.after(0, lambda: self._render(p))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.after(0, lambda: self.set_status("Error"))
        finally:
            pass

    def _render(self, p: Pokemon):
        self._hide_welcome()
        self.current_id = p.id
        # Load sprite
        photo = pil_image_from_url(p.image_url) if p.image_url else None
        if photo is None:
            # Fallback: simple placeholder using a blank colored square
            img = None
            if PIL_AVAILABLE:
                img = Image.new("RGBA", (280, 280), (245,245,245,255))
                photo = ImageTk.PhotoImage(img)
        self._sprite_photo = photo
        if photo:
            self.sprite_label.configure(image=photo, text="")
        else:
            self.sprite_label.configure(text="(no image)", image="", font=("Helvetica", 12), fg="#888888")

        # Name + ID (keep ID for context, but name is the focus requirement)
        self.name_id_label.configure(text=f"#{p.id:03d}  {p.name}")

        # Types badges
        for w in list(self.types_frame.children.values()):
            w.destroy()
        for idx, t in enumerate(p.types):
            tkey = t.lower()
            color = TYPE_COLORS.get(tkey, "#DDDDDD")
            fg = "white" if tkey not in ("electric","ground","grass","ice","fairy","normal") else "black"
            lbl = tk.Label(self.types_frame, text=t, bg=color, fg=fg, font=("Helvetica", 10, "bold"), padx=10, pady=2)
            lbl.grid(row=0, column=idx, padx=4)

        # Stats
        self.height_var.set(f"{p.height_m} m")
        self.weight_var.set(f"{p.weight_kg} kg")

        # Evolution info
        self.evolves_from_var.set(p.evolves_from if p.evolves_from else "—")
        self.evolves_to_var.set(", ".join(p.evolves_to) if p.evolves_to else "—")

        # Description
        self.desc_text.configure(state="normal")
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.tag_remove("content", "1.0", tk.END)
        self.desc_text.insert("1.0", p.description)
        self.desc_text.tag_add("content", "1.0", tk.END)
        self.desc_text.configure(state="disabled")

        # Enable navigation once we have a Pokémon displayed
        self.prev_btn.configure(state=tk.NORMAL)
        self.next_btn.configure(state=tk.NORMAL)

        self.set_status("Ready")

    def _hide_welcome(self):
        if self._welcome_visible:
            for widget in (self.greeting_label, self.instructions_label):
                if widget.winfo_exists():
                    widget.destroy()
            self._welcome_visible = False

def main():
    app = PokedexApp()
    app.mainloop()

if __name__ == "__main__":
    main()
