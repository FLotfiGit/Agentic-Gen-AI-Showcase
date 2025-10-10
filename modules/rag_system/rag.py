"""
Retrieval-Augmented Generation (RAG) System Module

This module implements a complete RAG system that combines document retrieval
with generation for enhanced question-answering capabilities.

Key Features:
- Document ingestion and chunking
- Vector embeddings and similarity search
- Context-aware generation
- Support for multiple vector stores
- Semantic search capabilities
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class Document:
    """Represents a document or document chunk"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    chunk_id: Optional[int] = None


@dataclass
class RetrievalResult:
    """Represents a retrieved document with its relevance score"""
    document: Document
    score: float
    rank: int


class VectorStore:
    """
    Simple in-memory vector store for document embeddings.
    In production, use FAISS, Pinecone, Chroma, or similar.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.documents: List[Document] = []
        self.index_to_doc: Dict[int, int] = {}
    
    def add_documents(self, documents: List[Document], embeddings: List[np.ndarray]):
        """Add documents and their embeddings to the store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        for doc, embedding in zip(documents, embeddings):
            if embedding.shape[0] != self.dimension:
                raise ValueError(f"Embedding dimension {embedding.shape[0]} doesn't match store dimension {self.dimension}")
            
            idx = len(self.vectors)
            self.vectors.append(embedding)
            self.documents.append(doc)
            self.index_to_doc[idx] = len(self.documents) - 1
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Find the k most similar documents to the query embedding.
        Uses cosine similarity.
        """
        if not self.vectors:
            return []
        
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension doesn't match store dimension")
        
        # Calculate cosine similarity
        similarities = []
        query_norm = np.linalg.norm(query_embedding)
        
        for idx, doc_embedding in enumerate(self.vectors):
            doc_norm = np.linalg.norm(doc_embedding)
            if query_norm > 0 and doc_norm > 0:
                similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
                similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for idx, score in similarities[:k]:
            doc_idx = self.index_to_doc[idx]
            results.append((self.documents[doc_idx], float(score)))
        
        return results


class Embedder:
    """
    Mock embedder for demonstration purposes.
    In production, use sentence-transformers, OpenAI embeddings, or similar.
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for text.
        This is a mock implementation - use real embeddings in production.
        """
        # Simple hash-based mock embedding
        # In production, use: sentence-transformers, OpenAI, Cohere, etc.
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dimension).astype(np.float32)
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        return [self.embed_text(text) for text in texts]


class DocumentChunker:
    """Splits documents into chunks for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > len(chunk_text) // 2:  # Only break if in latter half
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunk = Document(
                content=chunk_text.strip(),
                metadata={**metadata, 'chunk_range': (start, end)},
                doc_id=doc_id,
                chunk_id=chunk_id
            )
            chunks.append(chunk)
            
            chunk_id += 1
            start = end - self.chunk_overlap
        
        return chunks


class RAGSystem:
    """
    Complete Retrieval-Augmented Generation system.
    
    This system:
    1. Ingests and chunks documents
    2. Creates embeddings and stores them
    3. Retrieves relevant documents for queries
    4. Generates responses using retrieved context
    """
    
    def __init__(
        self,
        embedding_dimension: int = 384,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        self.embedder = Embedder(dimension=embedding_dimension)
        self.vector_store = VectorStore(dimension=embedding_dimension)
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.top_k = top_k
        self.documents_by_id: Dict[str, str] = {}
    
    def ingest_document(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Ingest a document into the RAG system.
        
        Args:
            text: The document text
            doc_id: Unique identifier for the document
            metadata: Optional metadata about the document
        """
        metadata = metadata or {}
        self.documents_by_id[doc_id] = text
        
        # Chunk the document
        chunks = self.chunker.chunk_text(text, doc_id, metadata)
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)
        
        # Store in vector store
        self.vector_store.add_documents(chunks, embeddings)
        
        return len(chunks)
    
    def ingest_documents(self, documents: List[Dict[str, Any]]):
        """
        Ingest multiple documents.
        
        Args:
            documents: List of dicts with 'text', 'doc_id', and optional 'metadata'
        """
        total_chunks = 0
        for doc in documents:
            chunks = self.ingest_document(
                text=doc['text'],
                doc_id=doc['doc_id'],
                metadata=doc.get('metadata', {})
            )
            total_chunks += chunks
        return total_chunks
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The search query
            k: Number of results to return (defaults to self.top_k)
            
        Returns:
            List of RetrievalResult objects
        """
        k = k or self.top_k
        
        # Embed the query
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.similarity_search(query_embedding, k=k)
        
        # Format results
        retrieval_results = []
        for rank, (document, score) in enumerate(results):
            retrieval_results.append(
                RetrievalResult(document=document, score=score, rank=rank + 1)
            )
        
        return retrieval_results
    
    def generate_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """
        Generate context string from retrieval results for the LLM.
        """
        context_parts = []
        for result in retrieval_results:
            context_parts.append(
                f"[Document {result.document.doc_id}, Chunk {result.document.chunk_id}]\n"
                f"{result.document.content}\n"
            )
        return "\n".join(context_parts)
    
    def query(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a complete RAG query: retrieve + generate.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with retrieved documents and generated answer
        """
        # Retrieve relevant documents
        retrieval_results = self.retrieve(question, k=k)
        
        # Generate context
        context = self.generate_context(retrieval_results)
        
        # In production, call an LLM with the context and question
        # For now, return a structured response
        answer = self._generate_answer(question, context)
        
        return {
            'question': question,
            'retrieved_documents': [
                {
                    'doc_id': r.document.doc_id,
                    'chunk_id': r.document.chunk_id,
                    'score': r.score,
                    'content': r.document.content[:200] + '...' if len(r.document.content) > 200 else r.document.content
                }
                for r in retrieval_results
            ],
            'context': context,
            'answer': answer,
            'metadata': {
                'num_retrieved': len(retrieval_results),
                'top_score': retrieval_results[0].score if retrieval_results else 0.0
            }
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer using the context.
        In production, this would call an LLM API.
        """
        # Placeholder - in production, call LLM with prompt:
        # f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context:"
        return f"[Answer would be generated by LLM using the retrieved context. Question: {question}]"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            'num_documents': len(self.documents_by_id),
            'num_chunks': len(self.vector_store.documents),
            'embedding_dimension': self.embedder.dimension,
            'chunk_size': self.chunker.chunk_size
        }


if __name__ == "__main__":
    print("=== RAG System Demo ===\n")
    
    # Create RAG system
    rag = RAGSystem(chunk_size=200, top_k=3)
    
    # Sample documents
    documents = [
        {
            'doc_id': 'doc1',
            'text': """Artificial Intelligence (AI) is intelligence demonstrated by machines, 
            in contrast to natural intelligence displayed by humans and animals. Leading AI 
            textbooks define the field as the study of intelligent agents: any device that 
            perceives its environment and takes actions that maximize its chance of successfully 
            achieving its goals. Machine learning is a subset of AI that provides systems the 
            ability to automatically learn and improve from experience without being explicitly 
            programmed.""",
            'metadata': {'source': 'AI Encyclopedia', 'category': 'definitions'}
        },
        {
            'doc_id': 'doc2',
            'text': """Deep Learning is part of a broader family of machine learning methods based 
            on artificial neural networks with representation learning. Learning can be supervised, 
            semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, 
            deep belief networks, recurrent neural networks and convolutional neural networks have been 
            applied to fields including computer vision, speech recognition, natural language processing, 
            and more.""",
            'metadata': {'source': 'ML Guide', 'category': 'techniques'}
        }
    ]
    
    # Ingest documents
    print("Ingesting documents...")
    total_chunks = rag.ingest_documents(documents)
    print(f"Ingested {len(documents)} documents into {total_chunks} chunks\n")
    
    # Query the system
    question = "What is machine learning?"
    print(f"Question: {question}\n")
    
    result = rag.query(question)
    
    print("Retrieved Documents:")
    for doc in result['retrieved_documents']:
        print(f"  - Doc {doc['doc_id']} (Chunk {doc['chunk_id']}): Score = {doc['score']:.4f}")
        print(f"    Content: {doc['content']}\n")
    
    print(f"Answer: {result['answer']}\n")
    print(f"Stats: {rag.get_stats()}")
