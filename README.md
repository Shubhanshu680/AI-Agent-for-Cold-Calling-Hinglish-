# AI-Agent-for-Cold-Calling-Hinglish-

# **Hinglish AI Cold Calling Agent**  

## **Project Overview**  
This project aims to design and implement an **AI agent capable of conducting personalized and human-like cold calls in Hinglish** for three business use cases:  

1. **Demo Scheduling:** Scheduling ERP system product demos.  
2. **Candidate Interviewing:** Conducting initial screening interviews.  
3. **Payment/Order Follow-up:** Reminding customers to release payments or place orders.  

The agent is designed to **understand context, personalize interactions, and exhibit human-like conversational abilities in Hinglish** using an **LLM, Speech Recognition (STT), and Text-to-Speech (TTS).**  

## **Time Allotment: 8 Hours**  

### **Overall Structure & Implementation Progress**  

### **1. Setup (✅ Completed - 30 minutes)**  
✔️ Set up a **Python environment** with required libraries:  
```bash
sudo apt-get install espeak  
pip install transformers datasets langchain-community pydub SpeechRecognition pyttsx3 nltk  
```  
✔️ Installed additional resources:  
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### **2. Data & Model Selection (✅ Completed - 1 hour)**  

✔️ **Curated example dialogues** for each use case (ERP demo, job interviews, payment follow-ups).  
✔️ Created a **knowledge base** with:
   - ERP product details (for demo scheduling).  
   - Candidate profiles (for interviews).  
   - Customer payment/order details (for follow-ups).  
✔️ Selected **`google/flan-t5-base`** as the **LLM for Hinglish text generation.**  
✔️ Chose **`pyttsx3`** for **Hinglish Text-to-Speech (TTS)** and **`SpeechRecognition`** for **Speech-to-Text (STT)** processing.  
✔️ Developed a **basic Hinglish translation mechanism** for improved engagement.  

---

### **3. Agent Design & Implementation (✅ Partially Completed - 3 hours)**  

#### **Key Implementations:**  
✔️ **LLM Integration:** Used `google/flan-t5-base` for natural Hinglish conversations.  
✔️ **Prompt Engineering:** Developed effective prompts for each scenario with **human-like Hinglish phrasing.**  
✔️ **State Management:** Implemented a **conversation state tracking mechanism** to ensure continuity.  
✔️ **Cold Call Scenarios:** Implemented **basic flows** for **ERP demo scheduling, candidate interviewing, and payment follow-ups.**  

#### **Partially Implemented Features:**  
🔶 **Hinglish Handling:** Implemented basic Hinglish translations but **lacks contextual understanding for mixed-language responses.**  
🔶 **Tool/API Integration:** No API integration (e.g., calendar for scheduling, CRM for customer data). **Currently, responses are simulated.**  

---

### **4. Evaluation & Refinement (✅ Partially Completed - 2 hours)**  

✔️ **Simulated conversations** to evaluate agent performance.  
✔️ Defined evaluation metrics:  
   - **Success Rate**: How often the agent successfully completes the conversation goal.  
   - **Engagement Score**: How natural and human-like the Hinglish conversation feels.  
   - **Response Accuracy**: Whether the agent correctly responds to user inputs.  

#### **Areas for Improvement:**  
🔶 **Refinement Needed:** Some responses **still feel robotic**, requiring **fine-tuning of prompts and better Hinglish phrasing.**  
🔶 **Speech Recognition Issues:** STT occasionally **misinterprets Hinglish words**, affecting conversation flow.  

---

### **5. Demonstration & Explanation (✅ Completed - 1.5 hours)**  

✔️ Recorded **Loom video demonstration** covering:  
   - Agent's **design choices**  
   - **Datasets and models** used  
   - **Challenges faced & solutions**  
   - **Live demonstration** of the agent handling each cold call scenario  

---

### **6. Final Tasks (✅ Completed - 30 minutes)**  

✔️ **Implemented basic error handling** for:  
   - Missing user responses  
   - Incorrect inputs  
✔️ **Code Documentation:** Added **comments for clarity.**  
✔️ Ensured **all components work correctly** within the current implementation scope.  

---

## **Challenges Faced & Solutions**  

### **1. Hinglish Language Processing Issues**  
- **Challenge:** Hinglish lacks standard grammar, making LLM responses inconsistent.  
- **Solution:** Used **prompt tuning** and **predefined Hinglish translation logic** to improve responses.  

### **2. Speech-to-Text (STT) Errors**  
- **Challenge:** Hinglish words were often **misrecognized.**  
- **Solution:** Applied **basic error handling** to re-prompt users when STT misinterpreted responses.  

### **3. Lack of API Integrations**  
- **Challenge:** Could not integrate external APIs (e.g., Google Calendar, CRM).  
- **Solution:** Used **simulated responses** to represent these functionalities for now.  

### **4. Conversational Flow Issues**  
- **Challenge:** Conversations sometimes felt **too scripted.**  
- **Solution:** **Dynamically adjusted prompts** based on past responses to improve natural engagement.  

---

## **Final Features Status**  

| Feature                           | Status  | Notes |
|------------------------------------|---------|------|
| **Interactive Conversation Flows** | ✅ Completed | Basic flows for all three scenarios implemented |
| **Hinglish Support**               | 🔶 Partially Implemented | LLM generates Hinglish responses, but STT errors persist |
| **Speech Recognition (STT)**       | 🔶 Partially Implemented | Works but struggles with Hinglish pronunciation |
| **Text-to-Speech (TTS)**           | ✅ Completed | `pyttsx3` used for Hinglish speech output |
| **State Tracking**                 | ✅ Completed | Conversation state management implemented |
| **Refined Hinglish Prompts**       | 🔶 Partially Implemented | Needs more **natural** phrasing improvements |
| **Error Handling**                 | ✅ Completed | Basic error handling for missing inputs |
| **Demonstration Video**            | ✅ Completed | Loom video recorded |

---

## **Future Improvements**  

1. **Better Hinglish Processing**  
   - Fine-tune LLM for **natural Hinglish conversations.**  
   - Improve **Hinglish STT/TTS** models for smoother speech recognition.  

2. **API Integrations**  
   - Connect with **Google Calendar** for real demo scheduling.  
   - Integrate **CRM APIs** to fetch real customer details.  

3. **Advanced Speech Features**  
   - **Better Hinglish STT handling** using fine-tuned **whisper models.**  
   - **More natural Hinglish speech synthesis** using `gTTS` instead of `pyttsx3`.  

4. **More Personalized Conversations**  
   - Implement **memory** so the agent can remember past user interactions.  

---

## **Conclusion**  

This project successfully implemented a **basic Hinglish AI cold-calling agent** capable of handling three key business scenarios. While **core functionalities are complete**, additional **fine-tuning** and **API integrations** are required for a fully production-ready system.  

✔️ **Successfully demonstrated agent capabilities** for Hinglish cold calls.  
🔶 **Hinglish NLP and STT handling need improvements.**  ❌ **No real API integrations (Calendar, CRM).**  



