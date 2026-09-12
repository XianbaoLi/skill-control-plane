# Dense Index V0.1

`DenseIndexV1` is a versioned JSON artifact for precomputed BigModel corpus
vectors.

```json
{
  "version": "dense-index-v1",
  "embedding_model": "embedding-3",
  "dimensions": 2048,
  "records": [
    {
      "skill_id": "example",
      "representation_hash": "sha256-hex",
      "embedding": [0.0, 1.0]
    }
  ]
}
```

Corpus embeddings are built offline. Production sidecar startup loads a
versioned Dense Index and performs only query embedding at runtime.

The index stores only `skill_id`, the SHA-256 of the exact production Dense
representation, and its vector. Retrieval Cards and Dense Index are both
offline artifacts. The sidecar never rebuilds, repairs, incrementally updates,
or caches around a stale index. Missing, malformed, incompatible, duplicate,
or representation-stale indexes fail closed; there is no BM25-only fallback.
