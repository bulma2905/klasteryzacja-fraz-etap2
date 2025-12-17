import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import unidecode
from typing import List, Dict, Any, Tuple

from rapidfuzz import fuzz

# OpenAI (tylko jeśli wybierzesz embeddingi lub generowanie briefów)
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------
# Konfiguracja strony
# -------------------------------------
st.set_page_config(page_title="🔗 Anti-kanibalizacja (kompatybilna z etapem 1)", layout="wide")
st.title("🔗 Anti-kanibalizacja między klastrami + (opcjonalnie) nowe briefy")

st.sidebar.header("⚙️ Ustawienia")

OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

method = st.sidebar.radio("Metoda wykrywania podobieństwa", ["RapidFuzz", "Embeddingi OpenAI"], index=0)

if method == "RapidFuzz":
    threshold_fuzz = st.sidebar.slider("Próg podobieństwa (RapidFuzz token_set_ratio)", 70, 100, 92, 1)
    threshold_emb = None
else:
    threshold_emb = st.sidebar.slider("Próg podobieństwa (cosine similarity)", 0.70, 0.99, 0.88, 0.01)
    threshold_fuzz = None
    EMB_MODEL = st.sidebar.selectbox("Model embeddingów", ["text-embedding-3-large", "text-embedding-3-small"], index=0)

# czy generować nowe briefy dla scalonych grup
GENERATE_NEW_BRIEFS_FOR_MERGED = st.sidebar.checkbox("Generuj NOWE briefy dla scalonych grup", value=True)
OPENAI_CHAT_MODEL = st.sidebar.selectbox("Model czatu (briefy)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)

# limit fraz wysyłanych do GPT (stabilny prompt)
MAX_PHRASES_FOR_GPT = st.sidebar.slider("Limit fraz wysyłanych do GPT", 20, 400, 140, 10)

# -------------------------------------
# Upload
# -------------------------------------
uploaded_file = st.file_uploader("Wgraj plik: frazy_klastry_briefy.xlsx (z etapu 1)", type=["xlsx"])


# -------------------------------------
# Normalizacja
# -------------------------------------
def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_csvish(s: str) -> List[str]:
    # odporne na None/nan i “dziwne” przecinki
    if s is None:
        return []
    s = str(s)
    if s.lower() == "nan":
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def dedup_list_keep_pretty(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        nx = normalize(x)
        if nx and nx not in seen:
            out.append(str(x).strip())
            seen.add(nx)
    return out

def pick_main_phrase(items: List[str]) -> str:
    items = [x for x in items if x and str(x).lower() != "nan"]
    if not items:
        return ""
    return sorted(items, key=lambda x: (len(normalize(x)), len(str(x))))[0]

def pick_reps_for_gpt_from_full_phrases(full_phrases: List[str], limit: int) -> List[str]:
    """
    Stabilny wybór fraz do promptu:
    - najpierw unikalne,
    - potem sort po krótszych (żeby nie spamować promptu),
    - docinamy do limitu.
    """
    uniq = dedup_list_keep_pretty(full_phrases)
    uniq_sorted = sorted(uniq, key=lambda x: (len(normalize(x)), len(x)))
    return uniq_sorted[:limit]


# -------------------------------------
# Brief generator
# -------------------------------------
def generate_brief(phrases: List[str], client: OpenAI, model: str) -> Dict[str, Any]:
    prompt = f"""
Dla poniższej listy fraz przygotuj dane do planu artykułu.

Frazy: {phrases}

Odpowiedz w formacie:

Intencja: [typ intencji wyszukiwania]
Tytuł: [SEO-friendly, max 70 znaków, naturalny, z głównym keywordem]
Wytyczne: [2–3 zdania opisu oczekiwań użytkownika]
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Jesteś asystentem SEO. Zawsze trzymaj się formatu."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        content = resp.choices[0].message.content.strip()
        result = {"intencja": "", "tytul": "", "wytyczne": ""}
        for line in content.splitlines():
            low = line.lower().strip()
            if low.startswith("intencja:"):
                result["intencja"] = line.split(":", 1)[1].strip()
            elif low.startswith("tytuł:") or low.startswith("tytul:"):
                result["tytul"] = line.split(":", 1)[1].strip()
            elif low.startswith("wytyczne:"):
                result["wytyczne"] = line.split(":", 1)[1].strip()
        return result
    except Exception:
        return {"intencja": "", "tytul": "", "wytyczne": ""}


# -------------------------------------
# Embeddings helper (batch)
# -------------------------------------
def get_embeddings(texts: List[str], client: OpenAI, model: str) -> np.ndarray:
    all_emb = []
    batch_size = 200 if "3-small" in model else 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        all_emb.extend([d.embedding for d in resp.data])

    return np.array(all_emb, dtype=np.float32)


# -------------------------------------
# Graph grouping (connected components)
# -------------------------------------
def build_groups_from_edges(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    groups = []

    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = []
        visited.add(i)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        if len(comp) > 1:
            groups.append(sorted(comp))

    return groups


# -------------------------------------
# MAIN
# -------------------------------------
if uploaded_file:
    # Wczytaj arkusze
    xls = pd.ExcelFile(uploaded_file)

    have_briefs = "Briefy" in xls.sheet_names
    have_klastry = "Klastry" in xls.sheet_names

    if not have_briefs and not have_klastry:
        st.error("Ten plik nie ma arkusza 'Briefy' ani 'Klastry'. Wgraj plik z etapu 1.")
        st.stop()

    # Priorytet: Briefy (bo ma intencja/tytul/wytyczne)
    if have_briefs:
        df = pd.read_excel(uploaded_file, sheet_name="Briefy")
        source_sheet = "Briefy"
    else:
        df = pd.read_excel(uploaded_file, sheet_name="Klastry")
        source_sheet = "Klastry"
        st.warning("Masz tylko arkusz 'Klastry' – można wykryć kanibalizację, ale bez generowania sensownych briefów (brak intencji/tytułu).")

    # Walidacja kolumn pod Twoją wersję
    required_cols = {"cluster_id", "main_phrase"}
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Brakuje kolumn w arkuszu '{source_sheet}': {missing}")
        st.stop()

    # kompatybilność: bierzemy najlepsze dostępne pola
    if "frazy_w_klastrze_pelne" in df.columns:
        full_phr_col = "frazy_w_klastrze_pelne"
    elif "frazy_w_klastrze" in df.columns:
        full_phr_col = "frazy_w_klastrze"
    elif "frazy_w_klastrze_pelne" not in df.columns and "frazy_w_klastrze" not in df.columns:
        # fallback minimalny
        full_phr_col = None

    # Kolumny briefowe (mogą nie istnieć, jeśli wgrano tylko Klastry)
    int_col = "intencja" if "intencja" in df.columns else None
    tit_col = "tytul" if "tytul" in df.columns else None
    gui_col = "wytyczne" if "wytyczne" in df.columns else None

    st.subheader("📄 Podgląd wejścia")
    st.write(f"Arkusz źródłowy: **{source_sheet}**")
    st.dataframe(df.head(20), use_container_width=True)

    # budujemy tekst porównawczy cmp – kompatybilny z Twoim etapem 1
    def row_cmp(r) -> str:
        parts = [normalize(r.get("main_phrase", ""))]
        if int_col:
            parts.append(normalize(r.get(int_col, "")))
        if tit_col:
            parts.append(normalize(r.get(tit_col, "")))
        # im stabilniej, tym lepiej: ale bez pełnych fraz (bo to by rozwaliło porównanie)
        return " | ".join([p for p in parts if p])

    df["cmp"] = df.apply(row_cmp, axis=1)
    texts = df["cmp"].tolist()
    n = len(df)

    if n < 2:
        st.info("Za mało wierszy do porównania.")
        st.stop()

    # OpenAI client (jeśli potrzebny)
    client = None
    if method == "Embeddingi OpenAI" or (GENERATE_NEW_BRIEFS_FOR_MERGED and OPENAI_API_KEY):
        if not OPENAI_API_KEY:
            st.error("Podaj OpenAI API Key (potrzebny dla embeddingów i/lub generowania briefów).")
            st.stop()
        client = OpenAI(api_key=OPENAI_API_KEY)

    # -------------------------
    # krok 1: podobieństwa -> edges
    # -------------------------
    st.subheader("1) Wykrywanie kanibalizacji między klastrami")

    progress = st.progress(0)
    status = st.empty()

    status.text("🔍 Liczę podobieństwa…")

    edges = []
    if method == "RapidFuzz":
        for i in range(n):
            for j in range(i + 1, n):
                sim = fuzz.token_set_ratio(texts[i], texts[j])
                if sim >= threshold_fuzz:
                    edges.append((i, j))
        progress.progress(35)
    else:
        emb = get_embeddings(texts, client, model=EMB_MODEL)
        progress.progress(25)
        status.text("🧠 Liczę macierz cosine similarity…")
        sim_matrix = cosine_similarity(emb)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold_emb:
                    edges.append((i, j))
        progress.progress(45)

    groups = build_groups_from_edges(n, edges)
    progress.progress(60)

    if not groups:
        st.success("✅ Nie wykryto kanibalizacji między klastrami wg ustawionych progów.")
        status.empty()
        progress.empty()
        st.stop()

    st.info(f"Znaleziono **{len(groups)}** grup do scalenia (kanibalizacja między klastrami).")

    # pokaż podejrzenia
    with st.expander("🔎 Pokaż wykryte grupy (podejrzenia kanibalizacji)"):
        for gid, idxs in enumerate(groups, 1):
            preview = df.loc[idxs, ["cluster_id", "main_phrase"]].copy()
            st.write(f"**Grupa {gid}** (scalane wiersze: {len(idxs)})")
            st.dataframe(preview, use_container_width=True)

    # -------------------------
    # krok 2: budowanie wyników finalnych
    # -------------------------
    st.subheader("2) Scalanie i (opcjonalnie) nowe briefy")

    grouped_indices = set()
    results = []

    for gid, idxs in enumerate(groups, 1):
        grouped_indices.update(idxs)

        cluster_ids = [str(df.loc[i, "cluster_id"]) for i in idxs]
        main_phrases = [str(df.loc[i, "main_phrase"]) for i in idxs]

        # pełne frazy
        all_phrases = []
        if full_phr_col:
            for i in idxs:
                all_phrases.extend(split_csvish(df.loc[i, full_phr_col]))
        all_phrases = dedup_list_keep_pretty(all_phrases)

        reps_for_gpt = pick_reps_for_gpt_from_full_phrases(all_phrases, limit=MAX_PHRASES_FOR_GPT)

        # brief
        merged_brief = {"intencja": "", "tytul": "", "wytyczne": ""}

        if GENERATE_NEW_BRIEFS_FOR_MERGED and client is not None and reps_for_gpt:
            status.text(f"📝 Generuję nowy brief dla grupy {gid}/{len(groups)}…")
            merged_brief = generate_brief(reps_for_gpt, client, model=OPENAI_CHAT_MODEL)

        # fallback: jeśli nie generujemy, weź “najlepszy” z istniejących
        if not merged_brief.get("intencja") and int_col:
            vals = [str(df.loc[i, int_col]) for i in idxs]
            merged_brief["intencja"] = pick_main_phrase(vals)
        if not merged_brief.get("tytul") and tit_col:
            vals = [str(df.loc[i, tit_col]) for i in idxs]
            merged_brief["tytul"] = pick_main_phrase(vals)
        if not merged_brief.get("wytyczne") and gui_col:
            vals = [str(df.loc[i, gui_col]) for i in idxs]
            merged_brief["wytyczne"] = pick_main_phrase(vals)

        results.append({
            "status": "SCALONE",
            "group_id": gid,
            "cluster_ids_scalone": ", ".join(cluster_ids),
            "main_phrase": pick_main_phrase(main_phrases),
            "intencja": merged_brief.get("intencja", ""),
            "tytul": merged_brief.get("tytul", ""),
            "wytyczne": merged_brief.get("wytyczne", ""),
            "frazy_w_klastrze_pelne": ", ".join(all_phrases) if all_phrases else "",
            "frazy_reprezentatywne_do_GPT": ", ".join(reps_for_gpt) if reps_for_gpt else "",
            "liczba_klastrow": len(cluster_ids),
            "liczba_fraz_pelnych": len(all_phrases),
        })

    # dodaj niescalone “jak leci”
    leftovers = [i for i in range(n) if i not in grouped_indices]
    for i in leftovers:
        row = df.loc[i].to_dict()
        results.append({
            "status": "POJEDYNCZY",
            "group_id": "",
            "cluster_ids_scalone": str(row.get("cluster_id", "")),
            "main_phrase": str(row.get("main_phrase", "")),
            "intencja": str(row.get(int_col, "")) if int_col else "",
            "tytul": str(row.get(tit_col, "")) if tit_col else "",
            "wytyczne": str(row.get(gui_col, "")) if gui_col else "",
            "frazy_w_klastrze_pelne": str(row.get(full_phr_col, "")) if full_phr_col else "",
            "frazy_reprezentatywne_do_GPT": str(row.get("frazy_reprezentatywne_do_GPT", "")) if "frazy_reprezentatywne_do_GPT" in df.columns else "",
            "liczba_klastrow": 1,
            "liczba_fraz_pelnych": len(split_csvish(row.get(full_phr_col, ""))) if full_phr_col else 0,
        })

    progress.progress(90)
    status.text("💾 Przygotowuję eksport…")

    results_df = pd.DataFrame(results)

    # sort: najpierw scalone
    results_df["_sort"] = results_df["status"].map({"SCALONE": 0, "POJEDYNCZY": 1}).fillna(2)
    results_df = results_df.sort_values(by=["_sort", "liczba_fraz_pelnych"], ascending=[True, False]).drop(columns=["_sort"])

    st.subheader("✅ Finalny plan (po scaleniu kanibalizacji)")
    st.dataframe(results_df, use_container_width=True)

    # Export Excel
    xlsx_buffer = io.BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Final_Briefy", index=False)

        # dodatkowo: wrzuć podejrzenia jako osobny arkusz
        groups_rows = []
        for gid, idxs in enumerate(groups, 1):
            for i in idxs:
                groups_rows.append({
                    "group_id": gid,
                    "cluster_id": df.loc[i, "cluster_id"],
                    "main_phrase": df.loc[i, "main_phrase"],
                    "cmp": df.loc[i, "cmp"],
                })
        pd.DataFrame(groups_rows).to_excel(writer, sheet_name="Wykryte_grupy", index=False)

    xlsx_buffer.seek(0)

    st.download_button(
        label="📥 Pobierz FINALNY Excel (po scaleniu kanibalizacji)",
        data=xlsx_buffer.getvalue(),
        file_name="briefy_po_kanibalizacji.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    progress.progress(100)
    status.text("✅ Gotowe!")





