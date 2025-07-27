// App.js
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [useRag, setUseRag] = useState(false); // New state for RAG checkbox
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }; 

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const newUserMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]); // Use functional update for state

    try {
      const res = await axios.post('http://localhost:8000/chat', {
        message: input,
        use_rag: useRag // <--- IMPORTANT: Send the use_rag state
      });

      const { reply, source, context_used } = res.data;

      // Format bot reply to include source and context if available
      let botReplyText = `Bot (${source}): ${reply}`;
      if (source === 'RAG' && context_used) {
        // Truncate context for display if too long
        const displayContext = context_used.length > 150 
                               ? context_used.substring(0, 150) + '...' 
                               : context_used;
        botReplyText += `\n(Context: "${displayContext}")`;
      }

      const newBotMessage = { sender: 'bot', text: botReplyText };
      setMessages((prevMessages) => [...prevMessages, newBotMessage]);

    } catch (err) {
      console.error("Error sending message:", err);
      setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: 'Error connecting to server or processing request.' }]);
    }

    setInput('');
  };

  return (
    <div className="chat-container">
      <h2>🤖 HuggingFace Chatbot</h2>
      <div className="chat-box">
        {messages.map((msg, index) => (
          <div key={index} className={`chat-msg ${msg.sender}`}>
            <span>{msg.text}</span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type your message..."
        />
        <button onClick={handleSend}>Send</button>
        <label className="rag-toggle">
          <input 
            type="checkbox" 
            checked={useRag} 
            onChange={(e) => setUseRag(e.target.checked)} 
          /> 
          Use RAG
        </label>
      </div>
    </div>
  );
}

export default App;