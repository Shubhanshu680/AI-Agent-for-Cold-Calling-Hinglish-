# AI-Agent-for-Cold-Calling-Hinglish-
Here’s an improved version of your README with additional details, including challenges faced during development:

---

# **Combined README for AI Agents**

This repository contains three AI agents designed for different business use cases. Each agent is built to interact with users in **Hinglish** (a mix of Hindi and English) and leverages the `google/flan-t5-base` language model for text generation. 

These AI agents are designed to enhance automation in candidate interviewing, ERP demo scheduling, and payment/order follow-ups, making business operations more efficient and accessible to a broader audience.

---

## **Overview of AI Agents**

### **1. Hinglish Candidate Interviewing Agent**
#### **Description**
The **Hinglish Candidate Interviewing Agent** conducts preliminary interviews for different job roles, making the hiring process more efficient. It engages candidates in Hinglish, ensuring a comfortable and interactive experience for individuals fluent in both Hindi and English.

#### **Key Features**
- **Interactive Interview Simulation**: Simulates the entire interview process, including greetings, questioning, feedback, and conclusion.
- **Multi-Role Support**: Conducts interviews for roles such as Software Engineer, Data Analyst, Marketing Executive, Sales Manager, and HR Recruiter.
- **Dynamic Questioning**: Adapts its questions based on candidate responses.
- **Candidate Profiles**: Simulates predefined candidate profiles for better evaluation.

#### **Setup Instructions**
1. Install the required libraries:
   ```bash
   sudo apt-get install espeak
   pip install transformers datasets langchain-community pydub SpeechRecognition pyttsx3
   ```
2. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
3. Run the agent:
   ```bash
   python Candidate_Interviewing.py
   ```

---

### **2. Hinglish ERP Demo Scheduling Agent**
#### **Description**
The **Hinglish ERP Demo Scheduling Agent** interacts with potential customers to explain the benefits of an ERP system and schedules demonstrations based on customer availability.

#### **Key Features**
- **Conversational AI for ERP Sales**: Engages customers in Hinglish and explains ERP system features, benefits, and pricing.
- **Demo Scheduling**: Collects customer preferences and schedules demos accordingly.
- **Personalized Engagement**: Adjusts the conversation based on customer responses.
- **Audio Output**: Converts the conversation to audio for review purposes.

#### **Setup Instructions**
1. Install dependencies:
   ```bash
   sudo apt-get install espeak
   pip install transformers datasets langchain-community pydub SpeechRecognition pyttsx3
   ```
2. Download required NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
3. Run the agent:
   ```bash
   python Demo_Scheduling.py
   ```

---

### **3. Hinglish Payment/Order Follow-up Agent**
#### **Description**
The **Hinglish Payment/Order Follow-up Agent** automates follow-ups with customers regarding pending payments or incomplete orders. It engages users in Hinglish, making the process more natural and effective.

#### **Key Features**
- **Automated Payment Reminders**: Notifies customers about pending payments and follows up if payments are not received.
- **Order Completion Reminders**: Encourages customers to complete their orders.
- **Customer Engagement in Hinglish**: Communicates effectively in Hinglish for better accessibility.
- **Payment Confirmation Handling**: Confirms received payments and acknowledges customer cooperation.

#### **Setup Instructions**
1. Install necessary packages:
   ```bash
   sudo apt-get install espeak
   pip install transformers datasets langchain-community pydub SpeechRecognition pyttsx3
   ```
2. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```
3. Run the agent:
   ```bash
   python Payment_Follow-up.py
   ```

## **Common Components Across All Agents**
### **1. Language Model**
All agents use `google/flan-t5-base` for text generation. This model provides coherent and contextually appropriate responses in Hinglish.

### **2. Hinglish Translation Mechanism**
A predefined dictionary-based translation system converts English text into Hinglish, improving accessibility.

### **3. Text-to-Speech (TTS)**
The `pyttsx3` library enables agents to speak responses, improving user engagement.

### **4. Speech Recognition**
The `SpeechRecognition` library allows the agents to process spoken input, although this feature is still under refinement.

### **5. Conversation State Management**
Each agent maintains a conversation state, ensuring coherent interactions.

---

## **Challenges Faced During Development**
Developing these AI agents came with several challenges, including:

### **1. Hinglish Language Processing**
- Hinglish lacks a standardized grammar, making it difficult for NLP models to understand and generate accurate responses.
- Creating a translation dictionary and fine-tuning responses required extensive testing.

### **2. Speech-to-Text (STT) and Text-to-Speech (TTS) Issues**
- **TTS Pronunciation Problems**: The `pyttsx3` library does not always pronounce Hinglish words naturally.
- **STT Accuracy Issues**: Speech recognition struggled with mixed-language input, especially when switching between Hindi and English.

### **3. Context Retention in Conversations**
- Ensuring the AI remembers the context of a conversation was a challenge, especially for longer discussions.
- Implementing a state-tracking mechanism improved conversation continuity.

### **4. Job Role-Specific Questioning**
- Different job roles required different sets of interview questions, making the system more complex.
- Handling varied responses dynamically required additional logic.

### **5. Scheduling and Follow-ups**
- The ERP demo scheduling and payment follow-up agents had to handle multiple customer scenarios, requiring extensive testing to cover different responses.

### **6. API Integration (Future Enhancement)**
- These agents currently function independently but could be integrated with external systems for scheduling, payments, and CRM updates.

## **Features Status**

### **✅ Completed Features**
- **Conversational Flow**: Each agent can conduct complete interactions.
- **Hinglish Support**: All agents effectively communicate in Hinglish.
- **Scenario Handling**: Agents can manage various scenarios (e.g., job roles, ERP features, payment reminders).

### **🟡 Partially Implemented Features**
- **TTS Optimization**: Needs improvements in Hinglish pronunciation.
- **Speech Recognition**: Needs better support for mixed-language input.

### **❌ Unfinished Features**
- **Advanced Feedback Mechanisms**: The current feedback system provides only basic responses.
- **API Integrations**: Could be extended to integrate with CRM, payment gateways, and scheduling tools.



