# RAG from Scratch

Implementazione di un sistema **RAG (Retrieval-Augmented Generation)** in Python a scopo didattico.

Gira interamente **in locale**:, LLM servito da **Ollama**, nessuna chiamata a un'API esterna a pagamento.

Il sistema RAG proposto ha le seguenti caratteristiche:
- leggere documenti testuali dalla cartella `knowledgebase`
- dividerli in chunk
- trasformare i chunk in embedding
- salvare i vettori in un database vettoriale
- recuperare i chunk più rilevanti per una query
- inviare il contesto all'LLM per generare una risposta basata sui documenti

## Struttura del progetto

- `main.py` – logica end-to-end (ingestion, retrieval, generation)
- `knowledgebase/` – documenti testo usati come base conoscitiva
- `index_cache/` – cache di chunk ed embedding (generata automaticamente)
- `pyproject.toml` – dipendenze del progetto

## Come funziona

1. I documenti in `knowledgebase/` vengono letti e divisi in chunk con overlap (`RecursiveCharacterTextSplitter`, lunghezza misurata in token, non caratteri).
2. Ogni chunk viene trasformato in un embedding con `intfloat/multilingual-e5-base` (modello multilingue, adatto anche all'italiano).
3. Gli embedding finiscono in un indice FAISS (`IndexFlatIP`, similarità coseno su vettori normalizzati).
4. Alla domanda dell'utente viene calcolato l'embedding e recuperati i **K chunk più simili**.
5. I chunk trovati vengono inseriti nel prompt e passati a un modello locale via Ollama, che risponde **solo** sulla base del contesto fornito — se l'informazione non c'è, lo dichiara.


## Caricamento e cache

La parte di cache è implementata in `main.py` e ha due obiettivi principali:

- evitare di ricalcolare chunk ed embedding per file già invariati
- invalidare la cache quando cambiano la configurazione o il contenuto del documento

Il comportamento è il seguente:

- `config_signature()` genera un digest della configurazione corrente (modello di embedding, chunk size, overlap, separatori)
- `file_hash()` calcola l'hash di ogni file `.txt`
- `cache_paths(filename)` crea un path univoco per ogni documento, per separare i file di chunk e embedding
- `load_manifest()` legge il manifest JSON e controlla se la configurazione è cambiata
- `build_knowledge_base()`:
  - controlla i file presenti in `knowledgebase`
  - elimina la cache dei file rimossi
  - verifica se un file è invariato confrontando hash e file cache
  - se è nuovo o modificato, ricalcola chunk ed embedding e salva tutto in `index_cache/`
  - se è già valido, carica direttamente i dati dalla cache

In pratica, la cache è per-documento e viene invalidata se:

- cambia il modello di embedding
- cambiano i parametri di chunking
- cambia il contenuto del file

## Requisiti

- Python 3.11+
- [Ollama](https://ollama.com) installato e in esecuzione localmente
- connessione internet per scaricare il modello di embedding (e il modello LLM, la prima volta)

## Installazione

Nella cartella del progetto esegui:

```bash
uv sync
```

## Esecuzione 

```bash
ollama serve # Assicurati che Ollama sia in esecuzione
ollama pull llama3.2:3b # Poi verifica e scarica il modello richiesto
uv run main.py # esegui il RAG
# oppure: python main.py
```

Il programma costruisce (o ricarica dalla cache) la knowledge base, poi apre un prompt interattivo: scrivi una domanda e premi invio. `exit` per uscire.

Ad ogni risposta viene stampato un riepilogo con modello usato, token consumati, durata e motivo di terminazione.


## Configurazione

I parametri principali sono in cima a `main.py`:

| Parametro | Descrizione |
|---|---|
| `EMBEDDING_MODEL` | Modello di embedding (default: `intfloat/multilingual-e5-base`) |
| `MODEL_NAME` | Modello LLM servito da Ollama (default: `llama3.2:3b`) |
| `CHUNK_SIZE_RATIO` / `CHUNK_OVERLAP_RATIO` | Dimensione e overlap dei chunk, come frazione della finestra massima del modello di embedding |
| `FIND_REVELANT_KNOWLEDGE` | Quanti chunk recuperare per ogni domanda (K) |
| `TEMPERATURE` | Temperatura del modello generativo |


## Setup di test

Configurazione con cui il progetto è stato validato:

- **Vector store**: FAISS (`IndexFlatIP`)
- **Modello di embedding**: `paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: `llama3.2:3b` via Ollama

## Nota

Per aumentare i limiti di download del modello Hugging Face, puoi configurare una variabile d'ambiente `HF_TOKEN`.

```powershell
$env:HF_TOKEN = "<tuo_token_hf>"
```

Questo non è obbligatorio, ma aiuta a evitare limiti di download e rate limit più rigidi.

## Roadmap

- [ ] Comandi dedicati per forzare l'aggiornamento della cache
- [ ] Supporto ad altri formati di documento oltre a `.txt`/`.md`
- [ ] Valutazione quantitativa della qualità del retrieval