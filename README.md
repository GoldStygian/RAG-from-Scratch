# RAG from Scratch

Questo progetto è una mini implementazione di un sistema RAG (Retrieval-Augmented Generation) in Python a scopo didattico.

Il sistema RAG proposto ha le seguenti caratteristiche:
- leggere documenti testuali dalla cartella `knowledgebase`
- dividerli in chunk
- trasformare i chunk in embedding
- salvare i vettori in un database vettoriale
- recuperare i chunk più rilevanti per una query
- inviare il contesto all'LLM per generare una risposta basata sui documenti

## Struttura del progetto

- `main.py` – logica principale del progetto
- `knowledgebase/` – documenti testo usati come base conoscitiva
- `index_cache/` – cache locale dei chunk e degli embedding generati
- `pyproject.toml` – dipendenze del progetto


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
- Internet per scaricare i modelli di embedding e, se necessario, i modelli LLM
- Ollama installato e in esecuzione localmente

## Installazione

Nella cartella del progetto esegui:

```bash
uv sync
```

## Esecuzione 

### Avvio di Ollama

Assicurati che Ollama sia in esecuzione:

```bash
ollama serve
```

Poi verifica e scarica il modello richiesto:

```bash
ollama pull llama3.2:3b
```

### Avvio del main

Avvia il progetto con:

```bash
uv run .\main.py
```

oppure:

```bash
python main.py
```

## Come funziona

1. Il programma legge tutti i file `.txt` presenti in `knowledgebase/`
2. I documenti vengono divisi in chunk
3. Ogni chunk viene trasformato in embedding
4. I vettori vengono memorizzati in FAISS
5. Quando inserisci una domanda, viene calcolato l'embedding della query
6. Il sistema recupera i chunk più simili
7. Il contesto viene passato al modello LLM
8. La risposta viene generata usando solo le informazioni trovate nei documenti

## Test

Il RAG è stato eseguito inizialmente con le seguenti caratteristiche

Database vettoriale: FAISS
Modello di Embvedding: paraphrase-multilingual-MiniLM-L12-v2
LLM testato: llama3.2:3b

## Nota

Per aumentare i limiti di download del modello Hugging Face, puoi configurare una variabile d'ambiente `HF_TOKEN`.

```powershell
$env:HF_TOKEN = "<tuo_token_hf>"
```

Questo non è obbligatorio, ma aiuta a evitare limiti di download e rate limit più rigidi.

## Upcoming Features

- Implementazione di comandi specifici per: aggiornare la cache ecc..