# RAG System Module Documentation

## Overview

The RAG (Retrieval-Augmented Generation) System module implements a complete pipeline for enhancing LLM responses with retrieved contextual information. This enables more accurate, grounded, and factual responses.

## Architecture

### Core Components

#### `RAGSystem`
The main orchestrator that coordinates all RAG operations.

**Key Methods:**
- `ingest_document(text, doc_id, metadata)`: Add a document to the system
- `ingest_documents(documents)`: Batch document ingestion
- `retrieve(query, k)`: Find relevant documents for a query
- `query(question, k)`: Complete RAG query with answer generation
- `get_stats()`: System statistics

#### `VectorStore`
In-memory vector database for similarity search.

**Features:**
- Cosine similarity search
- Efficient retrieval
- Document metadata storage

#### `Embedder`
Converts text into vector embeddings.

**Note:** Current implementation uses mock embeddings. In production, use:
- Sentence Transformers
- OpenAI Embeddings
- Cohere Embeddings
- Custom embedding models

#### `DocumentChunker`
Splits documents into manageable chunks with overlap.

**Configuration:**
- `chunk_size`: Size of each chunk (default: 500 characters)
- `chunk_overlap`: Overlap between chunks (default: 50 characters)

#### `Document`
Represents a document or document chunk.

**Attributes:**
- `content`: The text content
- `metadata`: Additional information
- `doc_id`: Unique document identifier
- `chunk_id`: Chunk number within document

#### `RetrievalResult`
Contains a retrieved document with relevance score.

**Attributes:**
- `document`: The retrieved Document
- `score`: Similarity score (0.0 to 1.0)
- `rank`: Position in results

## Usage

### Basic RAG Pipeline

```python
from modules.rag_system import RAGSystem

# Initialize system
rag = RAGSystem(
    embedding_dimension=384,
    chunk_size=500,
    chunk_overlap=50,
    top_k=5
)

# Ingest documents
documents = [
    {
        'doc_id': 'doc1',
        'text': 'Your document content here...',
        'metadata': {'source': 'example.pdf', 'page': 1}
    }
]
rag.ingest_documents(documents)

# Query the system
result = rag.query("What is machine learning?")

print(f"Answer: {result['answer']}")
print(f"Retrieved {result['metadata']['num_retrieved']} documents")
```

### Retrieval Only

```python
# Just retrieve relevant documents without generation
results = rag.retrieve("machine learning concepts", k=3)

for result in results:
    print(f"Rank {result.rank}: {result.document.doc_id}")
    print(f"Score: {result.score:.4f}")
    print(f"Content: {result.document.content[:200]}...")
```

### Advanced Document Ingestion

```python
# Ingest with rich metadata
document = {
    'doc_id': 'research_paper_2024',
    'text': full_paper_text,
    'metadata': {
        'title': 'Advanced AI Research',
        'authors': ['John Doe', 'Jane Smith'],
        'year': 2024,
        'category': 'research',
        'keywords': ['AI', 'machine learning', 'neural networks']
    }
}

num_chunks = rag.ingest_document(
    text=document['text'],
    doc_id=document['doc_id'],
    metadata=document['metadata']
)

print(f"Document split into {num_chunks} chunks")
```

## How RAG Works

### 1. Document Ingestion Phase

```
Raw Document → Chunking → Embedding → Vector Store
```

1. **Chunking**: Split documents into smaller pieces
   - Maintains context with overlap
   - Preserves semantic boundaries
   
2. **Embedding**: Convert text to vectors
   - Captures semantic meaning
   - Enables similarity search

3. **Storage**: Store vectors with metadata
   - Efficient retrieval
   - Preserve document structure

### 2. Query Phase

```
User Query → Embedding → Similarity Search → Context Generation → LLM → Answer
```

1. **Query Embedding**: Convert query to vector
2. **Retrieval**: Find most similar document chunks
3. **Context Assembly**: Combine retrieved chunks
4. **Generation**: LLM generates answer using context

## Configuration Options

### Chunk Size

```python
# Smaller chunks: More precise retrieval, but may lose context
rag = RAGSystem(chunk_size=200)

# Larger chunks: More context, but less precise
rag = RAGSystem(chunk_size=1000)
```

### Chunk Overlap

```python
# More overlap: Better context preservation
rag = RAGSystem(chunk_size=500, chunk_overlap=100)

# Less overlap: More unique chunks, faster processing
rag = RAGSystem(chunk_size=500, chunk_overlap=25)
```

### Retrieval Settings

```python
# Retrieve more documents for broader context
result = rag.query(question, k=10)

# Retrieve fewer for focused context
result = rag.query(question, k=3)
```

## Integration with Real Embeddings

### Using Sentence Transformers

```python
from sentence_transformers import SentenceTransformer

class RealEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text(self, text):
        return self.model.encode(text)
    
    def embed_batch(self, texts):
        return self.model.encode(texts)

# Replace the mock embedder
rag.embedder = RealEmbedder()
```

### Using OpenAI Embeddings

```python
import openai

class OpenAIEmbedder:
    def __init__(self):
        self.model = "text-embedding-ada-002"
    
    def embed_text(self, text):
        response = openai.Embedding.create(
            input=text,
            model=self.model
        )
        return response['data'][0]['embedding']
```

## Integration with Vector Databases

For production use, replace the in-memory vector store with:

### FAISS

```python
import faiss

# Create FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(embeddings)

# Search
distances, indices = index.search(query_embedding, k=5)
```

### ChromaDB

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=doc_ids
)

# Query
results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)
```

### Pinecone

```python
import pinecone

pinecone.init(api_key="your-api-key")
index = pinecone.Index("rag-index")

# Upsert vectors
index.upsert(vectors=[(id, embedding, metadata)])

# Query
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

## Best Practices

### 1. Document Preparation

- **Clean text**: Remove unnecessary formatting
- **Structure preservation**: Keep headings, lists intact
- **Metadata richness**: Add comprehensive metadata

### 2. Chunking Strategy

- **Semantic boundaries**: Split at natural breaks (paragraphs, sections)
- **Size vs. Context**: Balance chunk size with context needs
- **Overlap**: Use 10-20% overlap for context continuity

### 3. Retrieval Optimization

- **Relevant top_k**: Start with 3-5, adjust based on results
- **Reranking**: Consider reranking retrieved chunks
- **Filtering**: Use metadata for pre-filtering

### 4. Generation Quality

- **Context length**: Don't exceed LLM context window
- **Source attribution**: Include document sources in answers
- **Hallucination prevention**: Instruct LLM to use only provided context

## Performance Optimization

### Batch Processing

```python
# Ingest multiple documents at once
rag.ingest_documents(document_list)
```

### Caching

```python
# Cache embeddings for frequently queried texts
embedding_cache = {}

def cached_embed(text):
    if text not in embedding_cache:
        embedding_cache[text] = embedder.embed_text(text)
    return embedding_cache[text]
```

### Indexing

```python
# Use approximate nearest neighbor search for large datasets
# FAISS, HNSW, or Annoy for faster retrieval
```

## Common Use Cases

1. **Customer Support**: Answer questions using knowledge base
2. **Research Assistant**: Find relevant papers and generate summaries
3. **Document Q&A**: Query internal documents
4. **Code Search**: Find relevant code snippets
5. **Legal/Compliance**: Search regulations and policies

## Limitations

- Mock embeddings in current implementation
- In-memory storage (not suitable for large-scale)
- Simple cosine similarity (no hybrid search)
- No reranking or query expansion

## Future Enhancements

- [ ] Real embedding models integration
- [ ] Production vector database support
- [ ] Hybrid search (dense + sparse)
- [ ] Query expansion and reformulation
- [ ] Reranking mechanisms
- [ ] Multi-modal RAG (images, tables)
- [ ] Incremental indexing
- [ ] Distributed processing

## References

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [Building RAG-based LLM Applications](https://huggingface.co/blog/ray-rag)
