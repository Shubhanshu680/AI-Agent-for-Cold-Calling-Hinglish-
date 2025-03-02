# AI-Agent-for-Cold-Calling-Hinglish-

# Combined README for AI Agents

This repository contains three AI agents designed for different business use cases. Each agent is built to interact with users in **Hinglish** (a mix of Hindi and English) and leverages the `google/flan-t5-base` language model for text generation. Below is an overview of the three agents, their functionalities, and instructions for setting up and running them.

---

## 1. **Hinglish Candidate Interviewing Agent**
### Description
The **Hinglish Candidate Interviewing Agent** simulates initial screening interviews for various job roles. It conducts interviews in Hinglish, making it accessible to candidates comfortable with both Hindi and English. The agent supports multiple job roles, including Software Engineer, Data Analyst, Marketing Executive, Sales Manager, and HR Recruiter.

### Key Features
- **Interactive Interview Simulation**: Conducts full interviews, including introduction, questioning, feedback, and closing.
- **Hinglish Support**: Conducts interviews in Hinglish for better accessibility.
- **Multiple Job Roles**: Supports interviews for various job roles with role-specific questions.
- **Candidate Profiles**: Simulates interviews with predefined candidate profiles.

### Setup Instructions
1. Install required libraries:
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
   python Candidate_Interviewing.py
   ```

---

## 2. **Hinglish ERP Demo Scheduling Agent**
### Description
The **Hinglish ERP Demo Scheduling Agent** simulates cold calls to schedule demonstrations for an ERP system. It interacts with potential customers in Hinglish, explains the benefits of the ERP system, and schedules demos based on customer availability.

### Key Features
- **ERP System Knowledge**: Provides detailed information about ERP features, benefits, and pricing.
- **Hinglish Support**: Conducts conversations in Hinglish for better customer engagement.
- **Demo Scheduling**: Schedules demos based on customer availability and preferred contact methods.
- **Audio Output**: Saves conversation audio files for review.

### Setup Instructions
1. Install required libraries:
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
   python Demo_Scheduling.py
   ```

---

## 3. **Hinglish Payment/Order Follow-up Agent**
### Description
The **Hinglish Payment/Order Follow-up Agent** simulates follow-up calls for payment reminders and order placement. It interacts with customers in Hinglish, reminding them of pending payments or incomplete orders and confirming when payments are released.

### Key Features
- **Payment Reminders**: Sends reminders for pending payments and follows up if payments are not released.
- **Order Placement**: Encourages customers to complete pending orders.
- **Hinglish Support**: Conducts conversations in Hinglish for better customer engagement.
- **Payment Confirmation**: Confirms when payments are released and thanks customers for their cooperation.

### Setup Instructions
1. Install required libraries:
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

---

## Common Components Across All Agents

### Language Model
All agents use the `google/flan-t5-base` model for text generation. This model is a smaller but capable model that works well for generating responses in Hinglish.

### Hinglish Translation
A predefined dictionary is used to convert English text to Hinglish, making the interactions more accessible to users comfortable with both Hindi and English.

### Text-to-Speech (TTS)
The `pyttsx3` library is used for converting text to speech, allowing the agents to speak their responses.

### Conversation State Management
Each agent manages the state of the conversation, tracking the progress of the interaction and generating appropriate responses based on the current state.


## Features Status

### Completed Features
- **Interactive Simulation**: All agents can conduct full conversations with users.
- **Hinglish Support**: All agents support Hinglish for better user engagement.
- **Multiple Scenarios**: Each agent supports multiple scenarios (e.g., job roles, ERP features, payment reminders).

### Partially Implemented Features
- **Text-to-Speech (TTS)**: TTS functionality is implemented but not fully integrated into all conversation flows.
- **Speech Recognition**: Speech recognition is implemented but not fully integrated.

### Unfinished Features
- **Advanced Feedback Generation**: Feedback provided by the agents is currently generic and could be improved.
- **Integration with External APIs**: The agents could be extended to integrate with external APIs for enhanced functionality.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This combined README provides an overview of the three AI agents, their functionalities, and instructions for setting up and running them. Each agent is designed to interact with users in Hinglish, making them accessible to a wider audience.



