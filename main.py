# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, RagSequenceForGeneration, RagTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Import numpy

app = FastAPI(
    title="Chat bot API",
    description="This is a RAG-enabled chatbot API built with FastAPI. Access interactive documentation at /docs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*" for all origins
    allow_credentials=True,
    allow_methods=["*"],                      # Allow all methods including OPTIONS
    allow_headers=["*"],                      # Allow all headers
)

# --- DialoGPT (Fallback Model) ---
dialogpt_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
dialogpt_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# --- RAG Pipeline Components ---
rag_tokenizer = None
rag_model = None
# rag_pipeline = None # We will remove the pipeline for RAG and use model.generate directly
rag_enabled = False

try:
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    rag_enabled = True
    print("RAG pipeline (using BART-base) loaded successfully.")

except Exception as e:
    rag_enabled = False
    print(f"Failed to load RAG pipeline: {e}. RAG functionality will be disabled.")
    print("Please ensure you have the correct models and tokenizer dependencies properly installed.")
    print("Consider running: pip install transformers[torch] accelerate") # Suggest common installs for models

# Simulated document store (EXPANDED KNOWLEDGE BASE)
knowledge_base = [
    "FastAPI is a modern Python web framework for building APIs quickly and efficiently.",
    "RAG stands for Retrieval-Augmented Generation. It retrieves documents to enhance generation quality.",
    "DialoGPT is a transformer-based conversational model developed by Microsoft.",
    "Chroma is a lightweight open-source vector database used for RAG-style applications.",
    "The capital of France is Paris.",
    "The Earth revolves around the Sun.",
    "Python is a high-level, interpreted programming language.",
    "Machine learning is a subset of artificial intelligence.",
    "Nagpur is a major city in the Indian state of Maharashtra.",
    # --- ADDED FACTS FOR RAG TO ANSWER ---
    "Emmanuel Macron is the current president of France.",
    "The official language of France is French.",
    "India is a country in South Asia.",
    "The Prime Minister of India is Narendra Modi.",
    "The national animal of India is the tiger.",
    "The currency of India is the Indian Rupee (INR).",
    "The national bird of India is the peacock."
]

# Initialize TF-IDF Vectorizer for simple retrieval simulation
if rag_enabled: # Only initialize if RAG models loaded
    tfidf_vectorizer = TfidfVectorizer().fit(knowledge_base)
    knowledge_base_vectors = tfidf_vectorizer.transform(knowledge_base)


# Session memory for DialoGPT (global for simplicity, consider a more robust solution for production)
chat_history_ids = None

class Message(BaseModel):
    message: str
    use_rag: bool = False # New field to optionally enable RAG for a query

@app.post("/chat", summary="Chat with the bot (supports RAG and DialoGPT fallback)")
def chat(msg: Message):
    """
    Endpoint to interact with the chatbot.
    You can optionally enable RAG for specific queries by setting `use_rag` to `true`.
    If RAG is enabled but no relevant context is found, or if RAG fails to load,
    the bot will fall back to DialoGPT.
    """
    global chat_history_ids
    user_input = msg.message

    print(f"\n--- New Request ---")
    print(f"User Input: '{user_input}'")
    print(f"Use RAG requested: {msg.use_rag}")
    print(f"RAG model loaded: {rag_enabled}")


    # --- RAG Logic (if enabled and RAG model is loaded) ---
    if msg.use_rag and rag_enabled:
        print("Attempting RAG retrieval...")
        # Step 1: Improved Simulated Document Retrieval using TF-IDF Cosine Similarity
        query_vector = tfidf_vectorizer.transform([user_input])
        similarities = cosine_similarity(query_vector, knowledge_base_vectors).flatten()
        most_similar_doc_idx = np.argmax(similarities)
        
        retrieval_threshold = 0.25 # Slightly lowered to increase chances of finding context for short queries

        if similarities[most_similar_doc_idx] > retrieval_threshold:
            context = knowledge_base[most_similar_doc_idx]
            print(f"Retrieved context: '{context}' (Similarity: {similarities[most_similar_doc_idx]:.2f})")
        else:
            context = ""
            print(f"No sufficiently relevant context found for RAG. Max similarity: {similarities[most_similar_doc_idx]:.2f} (Threshold: {retrieval_threshold}). Falling back to DialoGPT.")


        if context:
            # Step 2: Use RAG Model for generation with context
            # For general text-to-text models like BART, we explicitly add context to the prompt
            query_for_rag_generation = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:"
            print(f"Sending to RAG model for generation: '{query_for_rag_generation}'")
            
            try:
                # Tokenize the input including context
                input_ids = rag_tokenizer.encode(query_for_rag_generation, return_tensors='pt', truncation=True, max_length=512)
                
                # Generate a response using the model's generate method
                output_ids = rag_model.generate(
                    input_ids,
                    max_new_tokens=100, # Generate up to 100 new tokens
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=rag_tokenizer.eos_token_id # Important for generation
                )
                
                # Decode the generated text, skipping the input prompt
                generated_text = rag_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Post-processing to remove the input prompt from the generated text
                # Find where the actual answer starts after the "Answer:" part
                answer_marker = "Answer:"
                if answer_marker in generated_text:
                    response = generated_text.split(answer_marker, 1)[-1].strip()
                else:
                    response = generated_text.strip() # If marker not found, take the whole thing
                
                # If the generated response is too short or generic, consider falling back
                if len(response) < 5 or response.lower() in ["answer:", "", ".", "i don't know.", "sorry, i cannot answer that."]: 
                     print(f"RAG generated a very short/empty/generic response: '{response}'. Falling back to DialoGPT.")
                     pass # Fall through to DialoGPT
                else:
                    print(f"RAG generated response: '{response}'")
                    return {"reply": response, "source": "RAG", "context_used": context}

            except Exception as e:
                print(f"Error during RAG generation: {e}. Falling back to DialoGPT.")
                # Fallback to DialoGPT if RAG generation fails
        else:
            print("No relevant context found for RAG. Falling back to DialoGPT.")

    # --- DialoGPT (Fallback/Default if RAG not used or failed) ---
    print("Using DialoGPT for response.")
    new_input_ids = dialogpt_tokenizer.encode(user_input + dialogpt_tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        input_ids = new_input_ids

    # Generate a response with DialoGPT
    chat_history_ids = dialogpt_model.generate(
        input_ids,
        max_length=500, 
        pad_token_id=dialogpt_tokenizer.eos_token_id,
        do_sample=True, 
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = dialogpt_tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"DialoGPT response: '{response}'")
    return {"reply": response, "source": "DialoGPT"}
