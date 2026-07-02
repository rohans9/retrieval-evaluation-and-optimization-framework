# Workflow 1: Corpus To Retrieval

## Goal

Run an end-to-end retrieval flow from a document directory without writing Python code.

## Steps

1. Ingest documents.
2. Chunk corpus.
3. Generate embeddings.
4. Build index.
5. Retrieve results for a query.

## Commands

```bash
retrieval ingest examples/sample_data/documents --config configs/examples/small_collection.yaml
retrieval chunk examples/sample_data/documents --config configs/examples/small_collection.yaml
retrieval retrieval embed --corpus-path data/processed/processed_corpus.json --config configs/examples/small_collection.yaml
retrieval retrieval index --corpus-path data/processed/processed_corpus.json --config configs/examples/small_collection.yaml --index-path data/index
retrieval retrieval retrieve --query "What is the maternity leave policy?" --config configs/examples/small_collection.yaml --index-path data/index --top-k 3
```
