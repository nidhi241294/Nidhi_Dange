# 🤖 Chatbot PoC using FastAPI & React.js with RAG

This is a Proof of Concept (PoC) chatbot application leveraging a combination of FastAPI (Python backend) and React.js (frontend), enhanced with a **Retrieval-Augmented Generation (RAG)** application for more informed and context-aware conversational AI. The RAG system utilizes a pre-trained language model and a vector database for efficient information retrieval.

---

## 📌 Features

- Natural language chatbot with **Retrieval-Augmented Generation (RAG)**
- Context-aware responses based on retrieved information
- Utilizes a pre-trained language model for conversational AI
- Built-in CORS support for frontend-backend communication
- Clean and minimal React UI
- Scroll to bottom on new messages
- Enter key support for quick messaging
- REST API endpoint with JSON support

---

## 📁 Project Structure

chatbot-poc/

├── backend/

      ├── main.py              # FastAPI backend with RAG integration

├── frontend/

      └── src/

           ├── App.js           # React UI logic

           ├── App.css          # Chat styling

├── data/                    # Directory for knowledge base/documents 

├── README.md                # Project documentation


---

## 🛠️ Technologies Used

| Layer      | Technology       |
|------------|------------------|
| Frontend   | React.js, Axios  |
| Backend    | FastAPI, Pydantic, Uvicorn |
| AI Model   | Pre-trained Language Model (e.g., from Hugging Face Transformers) |
| RAG System | Vector Database (e.g., FAISS, ChromaDB), Sentence Transformers |
| ML Framework | PyTorch / TensorFlow |

---

## 🔧 Backend Setup – FastAPI

##  Navigate to Backend Folder

 uvicorn main:app --reload

 Backend will be served at: http://localhost:8000

## 🚀 Frontend Setup – React.js

## Navigate to Frontend Folder
 
 npm start
 
 Frontend will be served at: http://localhost:3000

## 🔗 API Endpoint

Method ----  URL   ----     Description   

POST   ----    /chat ----     Sends message, returns bot reply enriched by RAG

## Export to Sheets

✅ Example Request
JSON

{
  "message": "What is the capital of France?"
}

✅ Example Response
JSON

{
  "reply": "The capital of France is Paris."
}


## 🧠 Notes
This PoC uses a global session. For multi-user environments, consider using unique session IDs or persistent storage.

The RAG system's effectiveness depends on the quality and comprehensiveness of your provided knowledge base/documents.

This setup is ideal for demonstration, testing, and learning purposes.

## 🔮 Future Improvements
Implement dynamic document ingestion for the RAG system.

Add user authentication and multi-session support.

Integrate a database for storing chat history.

Enhance UI with avatars, timestamps, and more responsive design elements.

Dockerize the application for easier deployment.

Explore different pre-trained language models and vector databases to optimize RAG performance.

## 🧪 Testing
Access the FastAPI Swagger UI for manual testing of API endpoints at:

http://localhost:8000/docs

## 📜 License
MIT License — Free to use, modify, and distribute.

## 👨‍💻 Author

 Built by Nidhi Dange
 
 GitHub: https://github.com/nidhi241294/Nidhi_Dange
