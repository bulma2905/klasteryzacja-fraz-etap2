import streamlit as st
import pandas as pd
import numpy as np
import io
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# -------------------------------------
# Konfiguracja strony
# -------------------------------------
st.set_page_config(page_title="🔗 Analiza kanibalizacji + Briefy", layout="wide")

st.title("🔗 Analiza i scalanie kanibalizacji + generowanie briefów")

# API Key
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

# Wybór modelu czatu
OPENAI_CHAT_MODEL = st.sidebar.selectbox("Model czatu", ["gpt-4o-mini", "gpt-4o"], index=0)

# Wybór metody
method = st.sidebar.radio("Metoda analizy podobieństwa", ["RapidFuzz", "Embeddingi OpenAI"])

# Progi
if method == "RapidFuzz":
    threshold = st.sidebar.slider("Próg podobieństwa (RapidFuzz)", 0, 100, 80, 1)
else:
    threshold = st.sidebar.slider("Próg podobieństwa (cosine similarity)", 0.0, 1.0, 0.85, 0.01)

# Upload pliku
uploaded_file = st.file_uploader("Wgraj plik frazy_briefy.xlsx", type=["xlsx"])


# -------------------------------------
# Funkcja do generowania briefu
# -------------------------------------
def generate_brief(frazy, client, model):
    prompt = f"""
Dla poniższej listy fraz przygotuj dane do planu artykułu.

Frazy: {frazy}

Odpowiedz w formacie:

Intencja: [typ intencji wyszukiwania]
Frazy: [lista fraz long-tail, rozdzielona przecinkami]
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
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        result = {"intencja": "", "frazy": frazy, "tytul": "", "wytyczne": ""}
        for line in content.splitlines():
            low = line.lower()
            if low.startswith("intencja:"):
                result["intencja"] = line.split(":", 1)[1].strip()
            elif low.startswith("frazy:"):
                result["frazy"] = line.split(":", 1)[1].strip()
            elif low.startswith("tytuł:") or low.startswith("tytul:"):
                result["tytul"] = line.split(":", 1)[1].strip()
            elif low.startswith("wytyczne:"):
                result["wytyczne"] = line.split(":", 1)[1].strip()
        return result
    except:
        return {"intencja": "", "frazy": frazy, "tytul": "", "wytyczne": ""}


# -------------------------------------
# Funkcja do embeddingów
# -------------------------------------
def get_embeddings(texts, client, model="text-embedding-3-large"):
    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data])


# -------------------------------------
# Logika główna
# -------------------------------------
if uploaded_file and OPENAI_API_KEY:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Podgląd danych wejściowych")
    st.dataframe(df.head())

    client = OpenAI(api_key=OPENAI_API_KEY)

    titles = df["tytul"].astype(str).tolist()
    n = len(df)

    # Tworzymy krawędzie grafu zależnie od metody
    edges = []

    if method == "RapidFuzz":
        for i in range(n):
            for j in range(i + 1, n):
                sim = fuzz.ratio(titles[i], titles[j])
                if sim >= threshold:
                    edges.append((i, j))
    else:  # Embeddingi
        embeddings = get_embeddings(titles, client, "text-embedding-3-large")
        sim_matrix = cosine_similarity(embeddings)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    edges.append((i, j))

    # Grupowanie (DFS na grafie)
    groups, visited = [], set()

    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for a, b in edges:
            if a == node and b not in visited:
                dfs(b, group)
            elif b == node and a not in visited:
                dfs(a, group)

    for i in range(n):
        if i not in visited:
            group = []
            dfs(i, group)
            if len(group) > 1:  # tylko grupy >1
                groups.append(group)

    results = []
    grouped_indices = set()

    # Briefy dla scalonych grup
    for gid, group in enumerate(groups, 1):
        cluster_ids = [df.loc[i, "cluster_id"] for i in group]
        frazy = []
        for i in group:
            frazy.extend(str(df.loc[i, "frazy"]).split(", "))
        frazy = list(set(frazy))  # unikalne frazy

        brief = generate_brief(", ".join(frazy), client, OPENAI_CHAT_MODEL)

        results.append({
            "status": "scalone",
            "group_id": gid,
            "cluster_ids": ", ".join(map(str, cluster_ids)),
            "main_phrase": df.loc[group[0], "main_phrase"],
            "intencja": brief["intencja"],
            "frazy": brief["frazy"],
            "tytul": brief["tytul"],
            "wytyczne": brief["wytyczne"]
        })

        grouped_indices.update(group)

    # Briefy dla niescalonych (pojedynczych) klastrów
    leftovers = set(range(n)) - grouped_indices
    for i in leftovers:
        frazy = str(df.loc[i, "frazy"])
        brief = generate_brief(frazy, client, OPENAI_CHAT_MODEL)
        results.append({
            "status": "pojedynczy",
            "group_id": "",
            "cluster_ids": str(df.loc[i, "cluster_id"]),
            "main_phrase": df.loc[i, "main_phrase"],
            "intencja": brief["intencja"],
            "frazy": brief["frazy"],
            "tytul": brief["tytul"],
            "wytyczne": brief["wytyczne"]
        })

    if results:
        st.subheader("📑 Kompletny plan artykułów (scalone + niescalone)")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        # Eksport do Excela
        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Briefy", index=False)
        xlsx_buffer.seek(0)

        st.download_button(
            label="📥 Pobierz wszystkie briefy",
            data=xlsx_buffer,
            file_name="briefy_pelne.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.success("✅ Nie znaleziono żadnych wyników.")

