import io
import os
import time
import pickle
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import unidecode

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🔗 Etap 2: Kanibalizacja (cmp) + Final Briefy",
    layout="wide"
)

st.title("🔗 Etap 2: Kanibalizacja po main_phrase + intencja + tytuł (cmp) → scalanie → finalne briefy")

# -----------------------------
# Sidebar
# -----------------------------
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

OPENAI_CHAT_MODEL = st.sidebar.selectbox(
    "Model czatu (meta + briefy)",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    index=0
)

META_MAX_PHRASES = st.sidebar.slider(
    "META: ile fraz maks. wysłać do modelu na 1 klaster",
    10, 200, 80, 10
)

FINAL_MAX_PHRASES = st.sidebar.slider(
    "FINAL: ile fraz maks. wysłać do modelu na 1 scaloną grupę",
    20, 400, 150, 10
)

method = st.sidebar.radio("Metoda wykrywania kanibalizacji (cmp)", ["RapidFuzz", "Embeddingi OpenAI"])

if method == "RapidFuzz":
    CMP_THRESHOLD = st.sidebar.slider("Próg podobieństwa (RapidFuzz token_set_ratio)", 60, 100, 88, 1)
else:
    CMP_THRESHOLD = st.sidebar.slider("Próg podobieństwa (cosine similarity)", 0.60, 1.00, 0.86, 0.01)

GENERATE_FINAL_FOR = st.sidebar.radio(
    "Finalne briefy generować dla:",
    ["Tylko scalone grupy", "Wszystkie (scalone + pojedyncze)"],
    index=0
)

# Checkpointy (osobne dla META i FINAL)
META_CKPT = "stage2_meta.pkl"
FINAL_CKPT = "stage2_final.pkl"

col_ckpt1, col_ckpt2 = st.sidebar.columns(2)
with col_ckpt1:
    if st.button("🗑️ Wyczyść META ckpt"):
        if os.path.exists(META_CKPT):
            os.remove(META_CKPT)
            st.success("META checkpoint usunięty.")
        else:
            st.info("Brak META checkpointa.")
with col_ckpt2:
    if st.button("🗑️ Wyczyść FINAL ckpt"):
        if os.path.exists(FINAL_CKPT):
            os.remove(FINAL_CKPT)
            st.success("FINAL checkpoint usunięty.")
        else:
            st.info("Brak FINAL checkpointa.")

st.sidebar.info(
    "Logika:\n"
    "1) Wczytaj klastery z etapu 1.\n"
    "2) META (intencja+tytuł) dla każdego klastra.\n"
    "3) cmp = main_phrase | intencja | tytuł → kanibalizacja.\n"
    "4) Scalanie.\n"
    "5) FINAL briefy dopiero po scaleniu."
)

# -----------------------------
# Helpers
# -----------------------------
def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = unidecode.unidecode(s)
    s = re.sub(r"[^a-z0-9\s\|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedup_list_keep_pretty(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        nx = normalize(x)
        if nx and nx not in seen:
            out.append(str(x).strip())
            seen.add(nx)
    return out

def load_ckpt(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return default
    return default

def save_ckpt(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def get_embeddings(client: OpenAI, texts: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def build_edges_fuzz(texts: List[str], threshold: int) -> List[Tuple[int, int, float]]:
    n = len(texts)
    edges = []
    for i in range(n):
        ti = texts[i]
        for j in range(i + 1, n):
            sim = fuzz.token_set_ratio(ti, texts[j])
            if sim >= threshold:
                edges.append((i, j, float(sim)))
    return edges

def build_edges_cosine(embeddings: np.ndarray, threshold: float) -> List[Tuple[int, int, float]]:
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 1.0)
    n = sim.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                edges.append((i, j, float(sim[i, j])))
    return edges

def connected_components(n: int, edges: List[Tuple[int, int, float]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for a, b, _ in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = [False] * n
    comps = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)
        if len(comp) > 1:
            comps.append(sorted(comp))
    return comps

def pick_main_phrase_from_phrases(phrases: List[str]) -> str:
    phrases = [p for p in phrases if p and str(p).strip().lower() != "nan"]
    if not phrases:
        return ""
    return sorted(phrases, key=lambda x: (len(normalize(x)), len(str(x))))[0]

def parse_phrases_cell(cell: Any) -> List[str]:
    if cell is None:
        return []
    s = str(cell)
    if s.strip().lower() == "nan":
        return []
    # split po przecinku, ale bez przesady
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

# -----------------------------
# LLM: META + FINAL
# -----------------------------
def llm_meta(cluster_phrases: List[str], client: OpenAI, model: str) -> Dict[str, str]:
    prompt = f"""
Dla poniższej listy fraz wyznacz META dla jednego artykułu.

Frazy: {cluster_phrases}

Zwróć DOKŁADNIE w formacie:
Intencja: ...
Tytuł: ...  (max 70 znaków)
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Jesteś asystentem SEO. Zwracasz tylko format: Intencja, Tytuł."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    out = {"intencja": "", "tytul": ""}
    for line in content.splitlines():
        low = line.lower().strip()
        if low.startswith("intencja:"):
            out["intencja"] = line.split(":", 1)[1].strip()
        elif low.startswith("tytuł:") or low.startswith("tytul:"):
            out["tytul"] = line.split(":", 1)[1].strip()
    return out

def llm_final_brief(phrases: List[str], client: OpenAI, model: str) -> Dict[str, str]:
    prompt = f"""
Dla poniższej listy fraz przygotuj dane do planu artykułu.

Frazy: {phrases}

Odpowiedz DOKŁADNIE w formacie:

Intencja: [typ intencji wyszukiwania]
Tytuł: [SEO-friendly, max 70 znaków, naturalny, z głównym keywordem]
Wytyczne: [2–3 zdania opisu oczekiwań użytkownika]
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Jesteś asystentem SEO. Zawsze trzymaj się formatu."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    content = resp.choices[0].message.content.strip()
    out = {"intencja": "", "tytul": "", "wytyczne": ""}
    for line in content.splitlines():
        low = line.lower().strip()
        if low.startswith("intencja:"):
            out["intencja"] = line.split(":", 1)[1].strip()
        elif low.startswith("tytuł:") or low.startswith("tytul:"):
            out["tytul"] = line.split(":", 1)[1].strip()
        elif low.startswith("wytyczne:"):
            out["wytyczne"] = line.split(":", 1)[1].strip()
    return out

# -----------------------------
# Input
# -----------------------------
uploaded = st.file_uploader("Wgraj Excel z etapu 1 (arkusz Klastry)", type=["xlsx"])

status = st.empty()
progress = st.progress(0)

def set_status(msg: str, p: int):
    status.text(msg)
    progress.progress(max(0, min(100, p)))

if uploaded and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

    # wczytaj arkusz
    xls = pd.ExcelFile(uploaded)
    sheet = "Klastry" if "Klastry" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(uploaded, sheet_name=sheet)

    st.subheader("Podgląd danych z etapu 1")
    st.dataframe(df.head(20), use_container_width=True)

    # --- autodetekcja kolumn ---
    col_cluster_id = "cluster_id" if "cluster_id" in df.columns else df.columns[0]
    # frazy pełne mogą mieć różne nazwy – próbujemy kilku
    candidates_phr = [c for c in df.columns if c.lower() in ["frazy_w_klastrze_pelne", "frazy_w_klastrze", "frazy", "frazy_do_uzycia", "frazy_w_klastrze_pelne "]]
    if not candidates_phr:
        # fallback: kolumna zawierająca dużo przecinków
        comma_scores = [(c, df[c].astype(str).str.count(",").mean()) for c in df.columns]
        comma_scores.sort(key=lambda x: x[1], reverse=True)
        col_phrases = comma_scores[0][0]
    else:
        col_phrases = candidates_phr[0]

    col_main = "main_phrase" if "main_phrase" in df.columns else None

    # zbuduj rekordy klastrów
    clusters = []
    for _, row in df.iterrows():
        cid = row.get(col_cluster_id)
        phrases = parse_phrases_cell(row.get(col_phrases))
        phrases = [p for p in phrases if p]
        phrases = dedup_list_keep_pretty(phrases)

        mp = str(row.get(col_main)) if col_main else ""
        if not mp or mp.strip().lower() == "nan":
            mp = pick_main_phrase_from_phrases(phrases)

        clusters.append({
            "cluster_id": cid,
            "main_phrase": mp,
            "phrases": phrases
        })

    set_status(f"✅ Wczytano klastrów: {len(clusters)}", 5)

    # -----------------------------
    # STEP A: META per cluster (intencja+tytuł)
    # -----------------------------
    if st.button("A) Wygeneruj META (intencja+tytuł) dla klastrów"):
        meta_map: Dict[str, Dict[str, str]] = load_ckpt(META_CKPT, default={})
        set_status("🧠 META: start…", 10)

        total = len(clusters)
        for i, c in enumerate(clusters, 1):
            key = str(c["cluster_id"])
            if key in meta_map and meta_map[key].get("tytul"):
                continue

            # do meta bierzemy max META_MAX_PHRASES
            ph = c["phrases"][:META_MAX_PHRASES]

            try:
                m = llm_meta(ph, client, OPENAI_CHAT_MODEL)
                meta_map[key] = {
                    "intencja": m.get("intencja", ""),
                    "tytul": m.get("tytul", "")
                }
                save_ckpt(META_CKPT, meta_map)
                time.sleep(0.15)
            except Exception as e:
                logging.warning(f"META error cluster {key}: {e}")
                meta_map[key] = {"intencja": "", "tytul": ""}
                save_ckpt(META_CKPT, meta_map)
                time.sleep(0.5)

            set_status(f"🧠 META {i}/{total}", int(10 + 60 * i / max(total, 1)))

        set_status("✅ META gotowe (zapisane w checkpoint).", 70)

    # pokaż meta jeśli jest
    meta_map = load_ckpt(META_CKPT, default={})
    if meta_map:
        df_meta = pd.DataFrame([
            {
                "cluster_id": c["cluster_id"],
                "main_phrase": c["main_phrase"],
                "intencja": meta_map.get(str(c["cluster_id"]), {}).get("intencja", ""),
                "tytul": meta_map.get(str(c["cluster_id"]), {}).get("tytul", ""),
                "liczba_fraz": len(c["phrases"]),
            }
            for c in clusters
        ])
        st.subheader("META (podgląd)")
        st.dataframe(df_meta, use_container_width=True)

    # -----------------------------
    # STEP B: Cannibalization detection on cmp
    # -----------------------------
    if st.button("B) Wykryj kanibalizację po CMP (main_phrase + intencja + tytuł)"):
        meta_map = load_ckpt(META_CKPT, default={})
        if not meta_map:
            st.error("Najpierw zrób krok A (META). Bez tytułu/intencji nie porównamy cmp.")
        else:
            set_status("🔍 Buduję CMP…", 5)

            cmp_texts = []
            for c in clusters:
                mid = str(c["cluster_id"])
                inten = meta_map.get(mid, {}).get("intencja", "")
                title = meta_map.get(mid, {}).get("tytul", "")
                cmp = f"{c['main_phrase']} | {inten} | {title}"
                cmp_texts.append(normalize(cmp))

            set_status("🔍 Liczę podobieństwa…", 20)

            if method == "RapidFuzz":
                edges = build_edges_fuzz(cmp_texts, threshold=int(CMP_THRESHOLD))
            else:
                embs = get_embeddings(client, cmp_texts, model="text-embedding-3-large")
                edges = build_edges_cosine(embs, threshold=float(CMP_THRESHOLD))

            comps = connected_components(len(clusters), edges)
            set_status(f"✅ Wykryto grup kanibalizacji: {len(comps)}", 60)

            # buduj wynikowe grupy: każda grupa to lista indeksów klastrów
            grouped_idx = set(i for g in comps for i in g)
            singles = [i for i in range(len(clusters)) if i not in grouped_idx]

            # raport grup
            report_groups = []
            for gid, g in enumerate(comps, 1):
                report_groups.append({
                    "group_id": gid,
                    "clusters_count": len(g),
                    "cluster_ids": ", ".join(str(clusters[i]["cluster_id"]) for i in g),
                    "main_phrases": " | ".join(clusters[i]["main_phrase"] for i in g),
                })
            df_groups = pd.DataFrame(report_groups)

            st.session_state["cannibal_groups"] = comps
            st.session_state["cannibal_singles"] = singles

            st.subheader("Grupy kanibalizacji (cmp)")
            if df_groups.empty:
                st.info("Nie znaleziono grup kanibalizacji wg wybranego progu.")
            else:
                st.dataframe(df_groups, use_container_width=True)

            set_status("✅ Kanibalizacja policzona. Teraz możesz scalić i wygenerować FINAL.", 75)

    # -----------------------------
    # STEP C: Merge + FINAL briefs
    # -----------------------------
    if st.button("C) Scal grupy i wygeneruj FINAL briefy"):
        if "cannibal_groups" not in st.session_state:
            st.error("Najpierw zrób krok B (wykrycie kanibalizacji).")
        else:
            meta_map = load_ckpt(META_CKPT, default={})
            final_rows: List[Dict[str, Any]] = load_ckpt(FINAL_CKPT, default=[])

            comps: List[List[int]] = st.session_state["cannibal_groups"]
            singles: List[int] = st.session_state["cannibal_singles"]

            # budujemy listę “prac”: scalone grupy + (opcjonalnie) single
            jobs = []
            for gid, g in enumerate(comps, 1):
                jobs.append(("merged", gid, g))
            if GENERATE_FINAL_FOR == "Wszystkie (scalone + pojedyncze)":
                for i in singles:
                    jobs.append(("single", None, [i]))

            done = len(final_rows)
            total = len(jobs)

            set_status(f"📝 FINAL start: {done}/{total} gotowe z checkpointa", 5)

            for idx, (kind, gid, gidxs) in enumerate(jobs, 1):
                if idx <= done:
                    continue

                # zbierz wszystkie frazy z tych klastrów
                phrases = []
                cluster_ids = []
                cmp_bits = []

                for i in gidxs:
                    cluster_ids.append(str(clusters[i]["cluster_id"]))
                    phrases.extend(clusters[i]["phrases"])
                    mid = str(clusters[i]["cluster_id"])
                    inten = meta_map.get(mid, {}).get("intencja", "")
                    title = meta_map.get(mid, {}).get("tytul", "")
                    cmp_bits.append(f"{clusters[i]['main_phrase']} | {inten} | {title}")

                phrases = dedup_list_keep_pretty(phrases)

                # do modelu docinamy, ale pełna pula zostaje w excelu
                phrases_for_gpt = phrases[:FINAL_MAX_PHRASES]

                set_status(
                    f"📝 FINAL {idx}/{total} | {kind} | klastry: {len(cluster_ids)} | frazy: {len(phrases)} | do GPT: {len(phrases_for_gpt)}",
                    int(10 + 80 * idx / max(total, 1))
                )

                try:
                    brief = llm_final_brief(phrases_for_gpt, client, OPENAI_CHAT_MODEL)
                except Exception as e:
                    logging.warning(f"FINAL error job {idx}: {e}")
                    brief = {"intencja": "", "tytul": "", "wytyczne": ""}

                final_rows.append({
                    "status": "scalone" if kind == "merged" else "pojedynczy",
                    "group_id": gid if kind == "merged" else "",
                    "cluster_ids": ", ".join(cluster_ids),
                    "main_phrase": pick_main_phrase_from_phrases([clusters[i]["main_phrase"] for i in gidxs]),
                    "cmp_debug": " || ".join(cmp_bits),
                    "intencja": brief.get("intencja", ""),
                    "tytul": brief.get("tytul", ""),
                    "wytyczne": brief.get("wytyczne", ""),
                    "frazy_pelne": ", ".join(phrases),
                    "frazy_do_GPT": ", ".join(phrases_for_gpt),
                })

                save_ckpt(FINAL_CKPT, final_rows)
                time.sleep(0.2)

            set_status("✅ FINAL gotowe.", 100)

            st.session_state["final_rows"] = final_rows

    # -----------------------------
    # Export
    # -----------------------------
    meta_map = load_ckpt(META_CKPT, default={})
    final_rows = st.session_state.get("final_rows") or load_ckpt(FINAL_CKPT, default=[])

    if meta_map or final_rows:
        # META sheet
        df_meta = pd.DataFrame([
            {
                "cluster_id": c["cluster_id"],
                "main_phrase": c["main_phrase"],
                "intencja": meta_map.get(str(c["cluster_id"]), {}).get("intencja", ""),
                "tytul": meta_map.get(str(c["cluster_id"]), {}).get("tytul", ""),
                "liczba_fraz": len(c["phrases"]),
                "frazy_pelne": ", ".join(c["phrases"])
            }
            for c in clusters
        ])

        df_final = pd.DataFrame(final_rows)

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df_meta.to_excel(writer, sheet_name="META_klastry", index=False)
            if not df_final.empty:
                df_final.to_excel(writer, sheet_name="FINAL_briefy", index=False)
        out.seek(0)

        st.download_button(
            label="📥 Pobierz Excel (META + FINAL)",
            data=out.getvalue(),
            file_name="etap2_meta_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("Podgląd FINAL (jeśli wygenerowane)")
        if df_final.empty:
            st.info("FINAL jeszcze nie wygenerowane (zrób krok C).")
        else:
            st.dataframe(df_final, use_container_width=True)

elif uploaded and not OPENAI_API_KEY:
    st.warning("Podaj OpenAI API Key w panelu bocznym.")





