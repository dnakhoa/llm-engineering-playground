"""
RAG (Retrieval-Augmented Generation) Example
=============================================
This file demonstrates a complete RAG pipeline using LangChain and ChromaDB.

For the interactive step-by-step version, open rag_systems.ipynb instead.

Prerequisites:
    pip install -r requirements.txt
    cp ../.env.example ../.env   # add your OPENAI_API_KEY
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI   # replaces deprecated HuggingFaceEmbeddings/HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================================
# STEP 1: Prepare Sample Documents
# ============================================================================

sample_documents = [
    """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals.
    """,
    """
    Machine Learning (ML) is a subset of artificial intelligence that provides 
    systems the ability to automatically learn and improve from experience 
    without being explicitly programmed. Machine learning focuses on the 
    development of computer programs that can access data and use it to learn for themselves.
    """,
    """
    Deep Learning is a subset of machine learning that uses neural networks 
    with many layers (deep neural networks). It's particularly effective for 
    tasks like image recognition, natural language processing, and speech recognition.
    Popular frameworks include TensorFlow, PyTorch, and Keras.
    """,
    """
    Natural Language Processing (NLP) is a branch of AI that helps computers 
    understand, interpret, and manipulate human language. NLP draws from many 
    disciplines including computer science and computational linguistics.
    Applications include translation, sentiment analysis, and chatbots.
    """,
    """
    Large Language Models (LLMs) are language models notable for their ability 
    to achieve general-purpose language generation and understanding. They acquire 
    these abilities by learning from massive amounts of text data. 
    Examples include GPT-4, Claude, Llama, and PaLM.
    """
]

# ============================================================================
# STEP 2: Text Chunking
# ============================================================================

def chunk_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    
    Key parameters:
    - chunk_size: Number of characters per chunk
    - chunk_overlap: Overlap between chunks to maintain context
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.create_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

# ============================================================================
# STEP 3: Create Embeddings and Vector Store
# ============================================================================

def create_vector_store(chunks):
    """
    Create embeddings and store them in a vector database.
    
    We're using:
    - HuggingFace embeddings (free, runs locally)
    - ChromaDB (lightweight vector store)
    """
    # Initialize embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Saves to disk
    )
    
    print("Vector store created successfully!")
    return vectorstore, embeddings

# ============================================================================
# STEP 4: Set Up Retriever
# ============================================================================

def setup_retriever(vectorstore):
    """
    Configure the retriever with search parameters.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # Return top 2 most similar chunks
    )
    return retriever

# ============================================================================
# STEP 5: Create Custom Prompt
# ============================================================================

custom_prompt_template = """
You are an AI assistant specialized in explaining technology concepts.
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say so clearly.
Always cite which document section you're referencing.

Context:
{context}

Question: {question}

Helpful Answer (with citations):
"""

def create_prompt():
    """Create a custom prompt template."""
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

# ============================================================================
# STEP 6: Set Up QA Chain
# ============================================================================

def create_qa_chain(retriever, prompt):
    """
    Create the RetrievalQA chain that combines retrieval + generation.
    
    Uses the shared provider — works with OpenAI, Anthropic, DeepSeek, etc.
    Set your API key in .env and run.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from shared.provider import chat, get_client, get_model_name
    
    # Wrap the shared chat() in a LangChain-compatible LLM
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks import CallbackManagerForLLMRun
    
    class SharedLLM(LLM):
        model_name: str = ""
        
        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
            return chat(prompt)
        
        @property
        def _llm_type(self) -> str:
            return "shared-provider"
    
    llm = SharedLLM(model_name=get_model_name())
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# ============================================================================
# STEP 7: Query the System
# ============================================================================

def query_system(qa_chain, question):
    """
    Query the RAG system and display results.
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    result = qa_chain.invoke({"query": question})
    
    print(f"Answer: {result['result']}")
    print("\nSource Documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\n[{i}] {doc.page_content[:150]}...")
    
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete RAG pipeline."""
    print("🚀 RAG System Demo")
    print("="*60)
    
    # Step 1: Chunk documents
    chunks = chunk_documents(sample_documents)
    
    # Step 2: Create vector store
    vectorstore, embeddings = create_vector_store(chunks)
    
    # Step 3: Setup retriever
    retriever = setup_retriever(vectorstore)
    
    # Step 4: Create prompt
    prompt = create_prompt()
    
    # Step 5: Create QA chain
    qa_chain = create_qa_chain(retriever, prompt)
    
    # Step 6: Test queries
    test_questions = [
        "What is the difference between AI and Machine Learning?",
        "How does Deep Learning work?",
        "What are some applications of NLP?"
    ]
    
    for question in test_questions:
        query_system(qa_chain, question)
    
    print("\n✅ Demo complete!")
    print("\nNext steps:")
    print("1. Replace FakeListLLM with a real LLM (OpenAI, Anthropic, etc.)")
    print("2. Load your own documents instead of sample text")
    print("3. Experiment with different chunk sizes and retrieval strategies")
    print("4. Add metadata filtering for more precise retrieval")

if __name__ == "__main__":
    main()
