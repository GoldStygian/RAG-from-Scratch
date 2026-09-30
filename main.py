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
import hashlib
import json

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = BASE_DIR / "knowledgebase"
CACHE_DIR = BASE_DIR / "index_cache"
MANIFEST_FILE = CACHE_DIR / "manifest.json"

#EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modello di embedding locale, non richiede Ollama
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Modello multilingue adatto anche all'italiano
EMBEDDING_DEVICE = "cpu" # oppure cuda per GPU Nvidia

MODEL_NAME = "llama3.2:3b" #"qwen2.5:7b" #qwen3.5
TEMPERATURE = 0.3

FIND_REVELANT_KNOWLEDGE = 4

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # valore fittizio, Ollama non lo verifica
)

# Fase 1: Ingestione (Costruzione del tuo Knowledge Base)
## Fase 1.1: Caricamento dei documenti
def load_documents():
    if not DOCUMENT_PATH.is_dir():
        raise FileNotFoundError(f"[-] Directory dei documenti non trovata: {DOCUMENT_PATH}")

    documents = []
    for file_path in sorted(DOCUMENT_PATH.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            documents.append({
                "name": file_path.name,
                "content": content,
            })

    if not documents:
        raise ValueError(f"Nessun documento .txt trovato in {DOCUMENT_PATH}")

    return documents

## Fase 1.2: Chunking dei documenti
# TO DEL 
def chunk_document(documents: list, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=CHUNK_SEPARATORS
    )

    chunks = []
    for document in documents:
        document_name = document.get("name", "unknown")
        document_content = document.get("content", "")

        for chunk in splitter.split_text(document_content):
            chunks.append({
                "name": document_name,
                "content": chunk,
            })

    return chunks

## Fase 1.2: Chunking di un solo documento
def chunk_document(name: str, content: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=CHUNK_SEPARATORS,  # Try these in order
    )
    return [{"name": name, "content": piece} for piece in splitter.split_text(content)]

## Fase 1.3: Creazioned egli embeddings
def create_embeddings(chunks: list, model: SentenceTransformer) -> list:
    content_only = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(content_only, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')  # FAISS vuole float32

    print("[d] Embeddings\n- shape:", embeddings.shape, "\n- dtype: ", embeddings.dtype)
    return embeddings

    #il numero 4 rappresenta la quantità dei tuoi dati. Significa che in questo momento hai trasformato in vettori esattamente 4 elementi (che siano 4 frasi, 4 parole o 4 immagini).
    # La prima dimensione indica il numero di documenti; la seconda e' la dimensione dei vettori (384 con questo modello).

    # Il numero 384 è la dimensione (o lunghezza) del vettore. Significa che ogni singola frase è stata trasformata in una sequenza di 384 numeri decimali (creata dal tuo modello di Intelligenza Artificiale, come ad esempio un modello di SentenceTransformers).


## Fase 1.4: Cache su disco (manifest + un file di cache per ogni documento)
def config_signature() -> str:
    """Tutto ciò da cui dipendono chunk ed embedding: se cambia, la cache va rifatta."""
    return json.dumps(
        {
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "separators": CHUNK_SEPARATORS,
        },
        sort_keys=True,
    )
 
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cache_paths(filename: str) -> tuple:
    """Nomi di cache univoci per file, basati sull'hash del nome (evita problemi con caratteri strani)."""
    key = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{key}.chunks.json", CACHE_DIR / f"{key}.embeddings.npy"
 
def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if manifest.get("config") == config_signature():
            return manifest
        print("[i] Configurazione cambiata (modello/chunking): cache invalidata")
    return {"config": config_signature(), "files": {}}


## Fase 1.5: Costruzione (o ricarica dalla cache) della knowledge base
def build_knowledge_base(model: SentenceTransformer) -> tuple:
    """
    Per ogni file .txt in DOCUMENT_PATH:
    - se è invariato rispetto all'ultima esecuzione (stesso hash) → carica chunk ed embedding dalla cache
    - se è nuovo o modificato → chunking + embedding, poi salva in cache
    File rimossi dalla cartella → la loro cache viene eliminata
 
    Ritorna (chunks, embeddings) pronti per costruire l'indice FAISS.
    """
    if not DOCUMENT_PATH.is_dir():
        raise FileNotFoundError(f"[-] Directory dei documenti non trovata: {DOCUMENT_PATH}")
    CACHE_DIR.mkdir(exist_ok=True)
 
    manifest = load_manifest()
    current_files = {p.name: p for p in sorted(DOCUMENT_PATH.glob("*.txt"))}
 
    if not current_files:
        raise ValueError(f"Nessun documento .txt trovato in {DOCUMENT_PATH}")
 
    # File rimossi dalla cartella dall'ultima esecuzione: elimina la loro cache
    for name in set(manifest["files"]) - set(current_files):
        chunks_path, emb_path = cache_paths(name)
        chunks_path.unlink(missing_ok=True)
        emb_path.unlink(missing_ok=True)
        del manifest["files"][name]
        print(f"[-] Rimosso dalla cache: {name}")
 
    all_chunks = []
    all_embeddings = []
 
    for name, path in current_files.items():
        current_hash = file_hash(path)
        chunks_path, emb_path = cache_paths(name)
        entry = manifest["files"].get(name)
 
        if entry and entry["hash"] == current_hash and chunks_path.exists() and emb_path.exists():
            # File invariato → carica dalla cache, niente da ricalcolare
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            embeddings = np.load(emb_path)
            print(f"[=] Da cache: {name} ({len(chunks)} chunk)")
        else:
            # File nuovo o modificato → chunking + embedding, poi salva
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                manifest["files"].pop(name, None)
                continue
 
            chunks = chunk_document(name, content)
            embeddings = create_embeddings(chunks, model)
 
            chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
            np.save(emb_path, embeddings)
            manifest["files"][name] = {"hash": current_hash, "n_chunks": len(chunks)}
            print(f"[+] Elaborato e messo in cache: {name} ({len(chunks)} chunk)")
 
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)
 
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
 
    return all_chunks, np.vstack(all_embeddings).astype("float32")

## Fase 1.6: Creazione del database vettoriale FAISS
def vectorize_docs(dimension: int, embedded_doc: list) -> faiss.Index:
    #L'indice più semplice è IndexFlatL2 (ricerca esatta per distanza euclidea) o IndexFlatIP (per dot product, utile se vuoi cosine similarity su vettori normalizzati):

    # Quando andrai a creare il tuo indice FAISS, la prima cosa che FAISS ti chiederà sarà: "Ok, vuoi creare una mappa, ma quanto devono essere lunghi i vettori che cammineranno dentro questa mappa?".
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embedded_doc))
    print("[i] indice: ", index.ntotal)
    return index


# Fase 2: Retrieval (Finding Relevant Knowledge)
def get_revelant_chunks(chunks: list, prompt: str, k: int, embed_model: SentenceTransformer, index: faiss.Index) -> list:
        
    embeded_prompt = np.asarray(embed_model.encode([prompt]), dtype='float32')

    k = min (k, FIND_REVELANT_KNOWLEDGE) # evita indici -1 se ci sono meno di k vettori totali

    distance, indices = index.search(embeded_prompt, k=k)  # Cerca i k chunk più vicini
    print("[i] Distances:", distance, "Indices:", indices) # Distances: [[0.5583499 1.4144423 1.5191814]] Indices: [[0 2 3]]
    return [chunks[i] for i in indices[0] if i != -1]  # Restituisce i documenti corrispondenti agli indici trovati


# Fase 3: Generation (Creating the Answer)
def generate_answare(retrived_chunks, user_prompt: str):

    context = "\n".join(chunk['name'] +":"+ chunk["content"] for chunk in retrived_chunks)
    prompt = f"""Answer the question based only on the context and always reporting the source. If the answer is not contained within the text below, say "I don't know. CONTEXT:\n\"\"\" {context}.\"\"\"\nQUESTION: {user_prompt}.""" # char(10) = \n in ASCII

    print("[i] Retrieved Chunks:")
    for chunk_number, chunk in enumerate(retrived_chunks, start=1):
        print(f"\n--- CHUNK {chunk_number} ({chunk['name']}) ---\n{chunk['content']}")
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

    print(f"[i] Modello di embedding: {EMBEDDING_MODEL} (max {model.max_seq_length} token)")
 
    chunks, embedded_docs = build_knowledge_base(model)
    print(f"[i] Knowledge base pronta: {len(chunks)} chunk totali")    

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

        retrived_chunks = get_revelant_chunks(chunks, user_input, FIND_REVELANT_KNOWLEDGE, model, index)
        response = generate_answare(retrived_chunks, user_input)

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