"""
Phase 1: Ingestion (Building Your Knowledge Base)
Documents → Chunking → Embedding → Vector DB Storage

Phase 2: Retrieval (Finding Relevant Knowledge)
User Query → Embed Query → Search Vector DB → Retrieve Top-K Chunks

Phase 3: Generation (Creating the Answer)
Query + Retrieved Context → LLM → Grounded Answer
"""

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import numpy as np
from openai import OpenAI
import faiss # oppure faiss-gpu se hai una GPU NVIDIA con CUDA
from time import perf_counter
from pathlib import Path

DOCUMENT_PATH = Path(__file__).resolve().parent / "knowledgebase"

#EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modello di embedding locale, non richiede Ollama
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Modello multilingue adatto anche all'italiano
EMBEDDING_DEVICE = "cpu" # oppure cuda per GPU Nvidia

MODEL_NAME = "llama3.2:3b" #"qwen2.5:7b" #qwen3.5
TEMPERATURE = 1

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # valore fittizio, Ollama non lo verifica
)

# Fase 1: Ingestione (Costruzione del tuo Knowledge Base)
## Fase 1.1: Caricamento dei documenti
def load_documents():
    if not DOCUMENT_PATH.is_dir():
        raise FileNotFoundError(f"Directory dei documenti non trovata: {DOCUMENT_PATH}")

    documents = [
        file_path.read_text(encoding="utf-8").strip()
        for file_path in sorted(DOCUMENT_PATH.glob("*.txt"))
    ]
    documents = [document for document in documents if document]

    if not documents:
        raise ValueError(f"Nessun documento .txt trovato in {DOCUMENT_PATH}")

    return documents

## Fase 1.2: Chunking dei documenti
def chunk_documents(documents: list, chunk_size: int = 500, overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]  # Try these in order
    )

    chunks = []
    for document in documents:
        chunks.extend(splitter.split_text(document))

    return chunks


## Fase 1.3: Creazioned egli embeddings
def create_embeddings(documents: list, model) -> list:
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')  # FAISS vuole float32

    print("[d] Embeddings\n- shape:", embeddings.shape, "\n- dtype: ", embeddings.dtype)
    return embeddings

    #il numero 4 rappresenta la quantità dei tuoi dati. Significa che in questo momento hai trasformato in vettori esattamente 4 elementi (che siano 4 frasi, 4 parole o 4 immagini).
    # La prima dimensione indica il numero di documenti; la seconda e' la dimensione dei vettori (384 con questo modello).

    # Il numero 384 è la dimensione (o lunghezza) del vettore. Significa che ogni singola frase è stata trasformata in una sequenza di 384 numeri decimali (creata dal tuo modello di Intelligenza Artificiale, come ad esempio un modello di SentenceTransformers).


## Fase 1.3: Creazione del database vettoriale FAISS
def vectorize_docs(dimension: int, embedded_doc: list):
    #L'indice più semplice è IndexFlatL2 (ricerca esatta per distanza euclidea) o IndexFlatIP (per dot product, utile se vuoi cosine similarity su vettori normalizzati):

    # Quando andrai a creare il tuo indice FAISS, la prima cosa che FAISS ti chiederà sarà: "Ok, vuoi creare una mappa, ma quanto devono essere lunghi i vettori che cammineranno dentro questa mappa?".
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embedded_doc))
    print("[i] indice: ", index.ntotal)
    return index


# Fase 2: Retrieval (Finding Relevant Knowledge)
def embed_prompt(prompt: str, k: int = 1) -> list:
    embeded_prompt = np.asarray(model.encode([prompt]), dtype='float32')
    distance, indices = index.search(embeded_prompt, k=k)  # Cerca i k chunk più vicini
    print("Distances:", distance, "Indices:", indices) # Distances: [[0.5583499 1.4144423 1.5191814]] Indices: [[0 2 3]]
    return [chunks[i] for i in indices[0]]  # Restituisce i documenti corrispondenti agli indici trovati

# Fase 3: Generation (Creating the Answer)
def rag_function(user_query: str):

    retrived_chunks = embed_prompt(user_query)

    prompt = f"""Answer the question based only on this context:\n\"\"\"
{chr(10).join(retrived_chunks)}.\"\"\"
Question: {user_query}.
If the answer is not contained within the text below, say "I don't know".""" # char(10) = \n in ASCII

    print("[i] Retrieved Chunks:")
    for chunk_number, chunk in enumerate(retrived_chunks, start=1):
        print(f"\n--- CHUNK {chunk_number} ---\n{chunk}")
    print("\n--------------------")
    print("[i] Prompt for LLM:", prompt, "\n--------------------")

    return client.chat.completions.create(
        model=MODEL_NAME,
        stream=True,
        stream_options={"include_usage": True},
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "Rispondi sempre e solo in italiano, indipendentemente dalla lingua del messaggio ricevuto."},
            {"role": "user", "content": prompt}
        ]
    )

def print_dashboard(rows):
    label_width = max(len(label) for label, _ in rows)
    value_width = min(max(len(value) for _, value in rows), 56)
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    title_width = label_width + value_width + 3

    print()
    print(border)
    print(f"| {'METRICHE RISPOSTA':^{title_width}} |")
    print(border)

    for label, value in rows:
        if len(value) > value_width:
            value = value[:value_width - 3] + "..."
        print(f"| {label:<{label_width}} | {value:<{value_width}} |")

    print(border)

if __name__ == "__main__":
    print("Sysyem RAG Initialization ...")

    model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE) # model = SentenceTransformer(EMBEDDING_MODEL, device='cuda')  # 384-dim embeddings in locale, GPU no ollama, ma lo scarica sempre?

    print(f"[i] Modello di embedding: {EMBEDDING_MODEL} (max {model.max_seq_length} token)")

    documents = load_documents()
    print("[+] Documents loaded")

    chunks = chunk_documents(documents)
    print(f"[i] Creati {len(chunks)} chunk")    

    embedded_docs = create_embeddings(chunks, model)
    print("[+] Embeddings Creati")

    index = vectorize_docs(embedded_docs.shape[1], embedded_docs)

    while True:
        try:
            user_input = input("> ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("\nExiting...")
            break

        started_at = perf_counter()

        response = rag_function(user_input)

        usage = None
        finish_reason = "N/D"
        response_id = "N/D"
        response_model = response.model if hasattr(response, "model") else "N/D"

        for chunk in response:
            response_id = getattr(chunk, "id", None) or response_id
            response_model = getattr(chunk, "model", None) or response_model
            usage = getattr(chunk, "usage", None) or usage

            if chunk.choices:
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                content = choice.delta.content
                if content:
                    print(content, end="", flush=True)

        elapsed = perf_counter() - started_at
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        print_dashboard([
            ("Stato", "COMPLETATA"),
            ("Modello", response_model),
            ("Token input (system + user)", str(input_tokens) if input_tokens is not None else "N/D"),
            ("Token output", str(output_tokens) if output_tokens is not None else "N/D"),
            ("Token totali", str(total_tokens) if total_tokens is not None else "N/D"),
            ("Durata", f"{elapsed:.2f} s"),
            ("Temperatura", str(TEMPERATURE)),
            ("Motivo fine", finish_reason),
            ("ID risposta", response_id),
        ])