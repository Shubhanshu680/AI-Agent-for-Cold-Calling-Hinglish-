# Install required libraries
!sudo apt-get install espeak
!pip install transformers datasets langchain-community pydub SpeechRecognition pyttsx3

# Import libraries
import os
import json
import random
import datetime
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
import pyttsx3
import speech_recognition as sr
from IPython.display import Audio, display
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the TTS engine
engine = pyttsx3.init()

# Test TTS
engine.say("Hello, this is a test of the text-to-speech engine.")
engine.runAndWait()

# PART 2: CONFIGURATION
# Setting up configuration parameters
config = {
    "model_name": "google/flan-t5-base",  # Using a smaller but capable model that works well in Colab
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "job_roles": {
        "Software Engineer": ["Python", "Java", "SQL", "Problem Solving"],
        "Data Analyst": ["Excel", "SQL", "Data Visualization", "Statistics"],
        "Marketing Executive": ["SEO", "Social Media", "Content Writing", "Campaign Management"],
        "Sales Manager": ["Negotiation", "CRM", "Lead Generation", "Communication"],
        "HR Recruiter": ["Talent Acquisition", "Interviewing", "Employee Relations", "Onboarding"]
    },
    "candidates": [
        {"name": "Rahul Sharma", "role": "Software Engineer", "experience": "2 years", "skills": ["Python", "Java", "SQL"]},
        {"name": "Priya Verma", "role": "Data Analyst", "experience": "3 years", "skills": ["Excel", "SQL", "Data Visualization"]},
        {"name": "Amit Patel", "role": "Marketing Executive", "experience": "1 year", "skills": ["SEO", "Social Media", "Content Writing"]},
        {"name": "Neha Singh", "role": "Sales Manager", "experience": "5 years", "skills": ["Negotiation", "CRM", "Lead Generation"]},
        {"name": "Suresh Kumar", "role": "HR Recruiter", "experience": "4 years", "skills": ["Talent Acquisition", "Interviewing", "Onboarding"]}
    ]
}

# PART 3: INTERVIEW KNOWLEDGE BASE
# Define job role details
job_role_info = {
    "Software Engineer": {
        "description": "Develop and maintain software applications using programming languages like Python and Java.",
        "key_skills": ["Python", "Java", "SQL", "Problem Solving"],
        "questions": [
            "Can you explain a project where you used Python?",
            "How do you handle debugging in Java?",
            "What is your experience with SQL databases?"
        ]
    },
    "Data Analyst": {
        "description": "Analyze data to provide insights and support decision-making processes.",
        "key_skills": ["Excel", "SQL", "Data Visualization", "Statistics"],
        "questions": [
            "How do you clean and preprocess data?",
            "Can you explain a time when you used SQL to solve a problem?",
            "What tools do you use for data visualization?"
        ]
    },
    "Marketing Executive": {
        "description": "Plan and execute marketing campaigns to promote products or services.",
        "key_skills": ["SEO", "Social Media", "Content Writing", "Campaign Management"],
        "questions": [
            "How do you measure the success of a marketing campaign?",
            "Can you describe a successful social media campaign you managed?",
            "What SEO strategies have you implemented?"
        ]
    },
    "Sales Manager": {
        "description": "Lead sales teams and manage client relationships to achieve revenue targets.",
        "key_skills": ["Negotiation", "CRM", "Lead Generation", "Communication"],
        "questions": [
            "How do you handle difficult clients?",
            "Can you describe a successful sales strategy you implemented?",
            "What CRM tools have you used?"
        ]
    },
    "HR Recruiter": {
        "description": "Recruit and onboard talent to meet organizational needs.",
        "key_skills": ["Talent Acquisition", "Interviewing", "Employee Relations", "Onboarding"],
        "questions": [
            "How do you source candidates for hard-to-fill roles?",
            "Can you describe your approach to conducting interviews?",
            "What onboarding processes have you implemented?"
        ]
    }
}

# PART 4: LANGUAGE MODEL SETUP
# Setting up the language model
def setup_language_model():
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(config["model_name"])

    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        repetition_penalty=config["repetition_penalty"]
    )

    # Create a HuggingFacePipeline instance for LangChain
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, tokenizer, model

# PART 5: HINGLISH LANGUAGE PROCESSING
# Predefined dictionary for English-to-Hinglish translation
hinglish_dict = {
    "thank you": "dhanyavaad",
    "experience": "anubhav",
    "skills": "kaushal",
    "project": "project",
    "explain": "samjhaana",
    "describe": "batana",
    "successful": "safal",
    "strategy": "ranneeti",
    "handle": "sambhaalna",
    "tools": "sadhane",
    "measure": "mapna",
    "approach": "dhang",
    "process": "prakriya",
    "candidate": "ummeedvaar",
    "recruit": "bharti karna",
    "onboard": "shuru karna",
    "interview": "sawaal-jawaab",
    "role": "bhumika",
    "team": "samuh",
    "client": "grahak",
    "data": "data",
    "analyze": "vishleshan karna",
    "campaign": "muhim",
    "communication": "sampark",
    "problem": "samasya",
    "solution": "samadhan",
    "time": "samay",
    "please": "kripaya",
    "great": "bahut badhiya",
    "perfect": "bilkul sahi",
    "sorry": "maaf kijiye",
    "share": "saajha karna",
    "best": "sabse behtar",
    "way": "tareeka",
    "send": "bhejna",
    "meeting": "baithak",
    "looking forward": "utsuk hoon",
    "questions": "sawaal",
    "feel free": "swatantra mahsoos karein",
    "wonderful": "shandaar",
    "day": "din"
}

def english_to_hinglish(text):
    """Convert English text to Hinglish using a predefined dictionary"""
    words = text.split()
    hinglish_words = []
    for word in words:
        lower_word = word.lower()
        # Randomly decide whether to translate the word (30% chance)
        if lower_word in hinglish_dict and random.random() < 0.3:
            hinglish_words.append(hinglish_dict[lower_word])
        else:
            hinglish_words.append(word)
    return " ".join(hinglish_words)

# PART 6: CONVERSATION STATE MANAGEMENT
class InterviewState:
    def __init__(self, candidate_info):
        self.candidate_info = candidate_info
        self.current_state = "introduction"
        self.interview_progress = 0
        self.questions_asked = []
        self.answers_received = []
        self.feedback = None

    def update_state(self, candidate_input, agent_response):
        """Update conversation state based on the latest interaction"""
        # Record the interaction
        self.questions_asked.append(agent_response)
        self.answers_received.append(candidate_input)

        # Update state based on interview progress
        if self.current_state == "introduction":
            self.current_state = "questioning"
        elif self.current_state == "questioning":
            self.interview_progress += 1
            if self.interview_progress >= 3:  # Ask 3 questions
                self.current_state = "feedback"
        elif self.current_state == "feedback":
            self.current_state = "closing"

        return self.current_state

    def get_state_summary(self):
        """Return a summary of the current interview state"""
        summary = f"Current stage: {self.current_state.replace('_', ' ').title()}"
        summary += f"\nQuestions asked: {len(self.questions_asked)}"
        summary += f"\nInterview progress: {self.interview_progress}/3"
        if self.feedback:
            summary += f"\nFeedback: {self.feedback}"
        return summary
    
# PART 7: INTERVIEW AGENT IMPLEMENTATION
class HinglishInterviewAgent:
    def __init__(self):
        print("Initializing Candidate Interviewing Agent...")
        self.candidates = config["candidates"]
        self.job_roles = config["job_roles"]
        self.current_candidate = None
        self.interview_state = None

    def start_new_interview(self, candidate_idx=None):
        """Start a new interview with a selected candidate"""
        # If no candidate_idx provided, randomly select a candidate
        if candidate_idx is None:
            candidate_idx = random.randint(0, len(self.candidates) - 1)

        self.current_candidate = self.candidates[candidate_idx]
        self.interview_state = InterviewState(self.current_candidate)

        # Generate introduction message
        intro_message = self.generate_introduction()
        return intro_message

    def generate_introduction(self):
        """Generate an introduction message for the interview"""
        candidate_name = self.current_candidate["name"]
        role = self.current_candidate["role"]

        intro_templates = [
            f"Hello {candidate_name}, thank you for joining us today. This is an initial screening interview for the {role} position. Shall we begin?",
            f"Namaste {candidate_name}, aaj ke interview ke liye dhanyavaad. Ye {role} role ke liye ek screening interview hai. Kya hum shuru kar sakte hain?",
            f"Good morning {candidate_name}, thank you for your time. This is a screening interview for the {role} role. Are you ready to start?"
        ]

        # Select a random introduction template
        intro = random.choice(intro_templates)

        # Convert to Hinglish if it's in English
        if not any(word in intro for word in ["Namaste", "aaj", "ke"]):
            intro = english_to_hinglish(intro)

        return intro

    def process_response(self, candidate_input):
        """Process candidate response and generate agent's next message"""
        # Convert Hinglish input to English for better processing
        english_input = hinglish_to_english(candidate_input)

        # Get the current state
        state = self.interview_state.current_state
        role = self.current_candidate["role"]

        # Generate response based on conversation state
        if state == "introduction":
            response = "Great! Let's start with a few questions about your experience and skills."

        elif state == "questioning":
            questions = job_role_info[role]["questions"]
            if self.interview_state.interview_progress < len(questions):
                response = questions[self.interview_state.interview_progress]
            else:
                response = "Thank you for answering the questions. Let's move to feedback."

        elif state == "feedback":
            response = "Based on your answers, you seem like a strong candidate. We will contact you soon with the next steps."
            self.interview_state.feedback = "Positive"
            self.interview_state.current_state = "closing"

        elif state == "closing":
            response = "Thank you for your time. Have a great day!"

        # Convert response to Hinglish
        hinglish_response = english_to_hinglish(response)

        # Update interview state
        self.interview_state.update_state(english_input, hinglish_response)

        return hinglish_response

    def get_current_state(self):
        """Return the current interview state"""
        if self.interview_state:
            return self.interview_state.get_state_summary()
        return "No active interview."

    def get_interview_history(self):
        """Return the interview history"""
        if self.interview_state:
            history = ""
            for i, (question, answer) in enumerate(zip(self.interview_state.questions_asked, self.interview_state.answers_received)):
                history += f"Question {i+1}: {question}\n"
                history += f"Answer: {answer}\n"
            return history
        return "No interview history."

# PART 8: INTERACTIVE DEMO FOR GOOGLE COLAB
def run_interactive_interview():
    """Run an interactive demo of the Hinglish Interview Agent"""
    print("Initializing Hinglish Candidate Interviewing Agent Demo...")
    agent = HinglishInterviewAgent()

    print("\n=== Candidate Interviewing Agent ===")
    print("This demo simulates an initial screening interview for a job role.")
    print("You'll play the role of the candidate responding to the agent's questions.")

    # Display available candidates
    print("\nAvailable candidates for simulation:")
    for i, candidate in enumerate(config["candidates"]):
        print(f"{i+1}. {candidate['name']} ({candidate['role']})")

    # Ask user to select a candidate
    while True:
        try:
            candidate_idx = int(input("\nSelect a candidate (1-5) or press Enter for random selection: ")) - 1
            if 0 <= candidate_idx < len(config["candidates"]):
                break
            else:
                print("Invalid selection. Please choose a number between 1 and 5.")
        except ValueError:
            candidate_idx = None
            break

    # Start the interview
    intro_message = agent.start_new_interview(candidate_idx)
    print("\n--- Interview Started ---")
    print(f"Agent: {intro_message}")

    # Interactive conversation loop
    while True:
        # Get candidate input
        text_input = input("\nYour response (type 'quit' to end the interview): ")
        if text_input.lower() == 'quit':
            break

        # Process response
        agent_response = agent.process_response(text_input)
        print(f"\nAgent: {agent_response}")

        # Check if interview should end
        if agent.interview_state.current_state == "closing":
            print("\n--- Interview completed! ---")
            print(agent.get_current_state())
            break

    # Display interview summary
    print("\n=== Interview Summary ===")
    print(agent.get_current_state())
    print("\n=== Interview History ===")
    print(agent.get_interview_history())


# PART 9: RUN THE DEMO
if __name__ == "__main__":
    run_interactive_interview()