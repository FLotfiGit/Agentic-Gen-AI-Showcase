"""
Example: RAG System

This example demonstrates how to build and use a Retrieval-Augmented Generation
system for enhanced question-answering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.rag_system import RAGSystem


def main():
    print("=" * 60)
    print("RAG SYSTEM EXAMPLE - Retrieval-Augmented Generation")
    print("=" * 60)
    print()
    
    # Create RAG system
    print("Initializing RAG System...")
    rag = RAGSystem(
        embedding_dimension=384,
        chunk_size=300,
        chunk_overlap=50,
        top_k=3
    )
    print(f"✅ RAG System initialized")
    print()
    
    # Example 1: Ingest documents about AI
    print("Example 1: Document Ingestion")
    print("-" * 60)
    
    documents = [
        {
            'doc_id': 'ai_overview',
            'text': """
            Artificial Intelligence (AI) is the simulation of human intelligence processes 
            by machines, especially computer systems. These processes include learning 
            (the acquisition of information and rules for using the information), reasoning 
            (using rules to reach approximate or definite conclusions) and self-correction. 
            AI systems can be categorized as either weak AI or strong AI. Weak AI, also 
            known as narrow AI, is designed to perform a specific task, such as facial 
            recognition. Strong AI, also known as artificial general intelligence (AGI), 
            refers to machines that exhibit human cognitive abilities.
            """,
            'metadata': {'source': 'AI Encyclopedia', 'topic': 'General AI'}
        },
        {
            'doc_id': 'machine_learning',
            'text': """
            Machine Learning (ML) is a subset of artificial intelligence that provides 
            systems the ability to automatically learn and improve from experience without 
            being explicitly programmed. ML focuses on the development of computer programs 
            that can access data and use it to learn for themselves. The process of learning 
            begins with observations or data, such as examples, direct experience, or 
            instruction, in order to look for patterns in data and make better decisions 
            in the future based on the examples that we provide.
            """,
            'metadata': {'source': 'ML Guide', 'topic': 'Machine Learning'}
        },
        {
            'doc_id': 'deep_learning',
            'text': """
            Deep Learning is a subset of machine learning based on artificial neural 
            networks with multiple layers. These neural networks attempt to simulate the 
            behavior of the human brain—albeit far from matching its ability—allowing it 
            to "learn" from large amounts of data. While a neural network with a single 
            layer can still make approximate predictions, additional hidden layers can help 
            to optimize and refine for accuracy. Deep learning drives many artificial 
            intelligence applications and services that improve automation.
            """,
            'metadata': {'source': 'DL Textbook', 'topic': 'Deep Learning'}
        },
        {
            'doc_id': 'nlp',
            'text': """
            Natural Language Processing (NLP) is a branch of artificial intelligence that 
            helps computers understand, interpret and manipulate human language. NLP draws 
            from many disciplines, including computer science and computational linguistics, 
            in its pursuit to fill the gap between human communication and computer 
            understanding. NLP is used in many applications such as chatbots, voice assistants, 
            sentiment analysis, machine translation, and text summarization.
            """,
            'metadata': {'source': 'NLP Handbook', 'topic': 'NLP'}
        }
    ]
    
    total_chunks = rag.ingest_documents(documents)
    print(f"Ingested {len(documents)} documents into {total_chunks} chunks")
    print()
    
    # Example 2: Query the system
    print("Example 2: Question Answering")
    print("-" * 60)
    
    questions = [
        "What is machine learning?",
        "What are the types of AI?",
        "How does deep learning work?",
        "What applications use NLP?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = rag.query(question, k=2)
        
        print(f"A: {result['answer']}")
        print(f"   Retrieved {result['metadata']['num_retrieved']} documents")
        print(f"   Top relevance score: {result['metadata']['top_score']:.4f}")
        print(f"   Sources:")
        for doc in result['retrieved_documents']:
            print(f"     - {doc['doc_id']} (chunk {doc['chunk_id']}, score: {doc['score']:.4f})")
    
    print()
    
    # Example 3: Retrieval only
    print("Example 3: Semantic Search (Retrieval Only)")
    print("-" * 60)
    
    query = "neural networks and learning from data"
    print(f"Query: '{query}'")
    print()
    
    retrieval_results = rag.retrieve(query, k=3)
    
    print(f"Found {len(retrieval_results)} relevant chunks:")
    for result in retrieval_results:
        print(f"\n  Rank {result.rank}: {result.document.doc_id} (Chunk {result.document.chunk_id})")
        print(f"  Score: {result.score:.4f}")
        print(f"  Preview: {result.document.content[:150]}...")
    
    print()
    
    # Example 4: System statistics
    print("Example 4: System Statistics")
    print("-" * 60)
    stats = rag.get_stats()
    print(f"Number of documents: {stats['num_documents']}")
    print(f"Number of chunks: {stats['num_chunks']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print(f"Chunk size: {stats['chunk_size']}")
    print()
    
    print("=" * 60)
    print("✅ RAG System examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
