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
    "available_slots": {
        "Monday": ["10:00 AM", "2:00 PM", "4:00 PM"],
        "Tuesday": ["11:00 AM", "1:00 PM", "3:00 PM"],
        "Wednesday": ["9:00 AM", "12:00 PM", "5:00 PM"],
        "Thursday": ["10:00 AM", "3:00 PM", "4:00 PM"],
        "Friday": ["11:00 AM", "2:00 PM", "4:00 PM"]
    },
    "erp_features": [
        "Inventory Management",
        "Financial Accounting",
        "Human Resources",
        "Supply Chain Management",
        "Customer Relationship Management",
        "Business Intelligence",
        "Manufacturing Management"
    ],
    "companies": [
        {"name": "Sharma Electronics", "industry": "Electronics Manufacturing", "needs": ["Inventory Management", "Manufacturing Management"]},
        {"name": "Kumar Textiles", "industry": "Textile Manufacturing", "needs": ["Supply Chain Management", "Inventory Management"]},
        {"name": "Verma Pharmaceuticals", "industry": "Pharmaceutical", "needs": ["Compliance Management", "Supply Chain Management"]},
        {"name": "Singh Retail Solutions", "industry": "Retail", "needs": ["Inventory Management", "Customer Relationship Management"]},
        {"name": "Patel Construction", "industry": "Construction", "needs": ["Project Management", "Financial Accounting"]}
    ],
    "audio_output_dir": "audio_outputs"  # New parameter for audio output directory
}

# Create audio output directory if it doesn't exist
if not os.path.exists(config["audio_output_dir"]):
    os.makedirs(config["audio_output_dir"])

# PART 3: ERP SYSTEM KNOWLEDGE BASE
# Define product info for the ERP system
erp_product_info = {
    "name": "VyaparERP Pro",
    "version": "2023.2",
    "description": "A comprehensive ERP solution designed for Indian businesses of all sizes.",
    "features": {
        "Inventory Management": {
            "description": "Complete tracking of inventory with barcode integration, automatic reordering, and stock valuation.",
            "benefits": "Reduce inventory costs by 20% and prevent stockouts."
        },
        "Financial Accounting": {
            "description": "GST-compliant accounting with automated tax calculations, invoicing, and financial reporting.",
            "benefits": "Simplify tax compliance and get real-time financial insights."
        },
        "Human Resources": {
            "description": "Employee management with attendance tracking, payroll processing, and performance evaluation.",
            "benefits": "Streamline HR processes and improve employee satisfaction."
        },
        "Supply Chain Management": {
            "description": "End-to-end supply chain visibility with vendor management, procurement, and logistics tracking.",
            "benefits": "Optimize supply chain operations and reduce lead times."
        },
        "Customer Relationship Management": {
            "description": "Customer database, lead management, and sales pipeline tracking with analytics.",
            "benefits": "Increase customer retention and boost sales conversion rates."
        },
        "Business Intelligence": {
            "description": "Advanced reporting and analytics with customizable dashboards and KPI tracking.",
            "benefits": "Make data-driven decisions with actionable insights."
        },
        "Manufacturing Management": {
            "description": "Production planning, shop floor control, and quality management integration.",
            "benefits": "Improve production efficiency and product quality."
        }
    },
    "pricing": {
        "basic": "₹15,000 per month (up to 10 users)",
        "professional": "₹25,000 per month (up to 25 users)",
        "enterprise": "₹40,000 per month (unlimited users)"
    },
    "implementation_time": "4-6 weeks for standard implementation",
    "support": "24/7 dedicated support team based in India",
    "customization": "Available for all plans with additional charges"
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
    "solution": "samadhan",
    "company": "company",
    "operations": "karyapranali",
    "efficiency": "kushalta",
    "management": "prabandhan",
    "improve": "sudhaarna",
    "processes": "prakriyayein",
    "features": "visheshtayein",
    "specific": "vishesh",
    "business": "vyapaar",
    "time": "samay",
    "discuss": "charcha karna",
    "helping": "madad kar raha hai",
    "clients": "grahak",
    "increase": "badhna",
    "focus": "dhyan",
    "believe": "vishwaas hai",
    "streamline": "saral banana",
    "address": "sambhaalna",
    "needs": "aavashyaktaayein",
    "briefly": "sankshipt mein",
    "explain": "samjhaana",
    "how": "kaise",
    "sales": "vikray",
    "improve": "behtar banana",
    "inventory": "samaan",
    "manufacturing": "nirmaan",
    "system": "pranali",
    "demo": "pradarshan",
    "schedule": "nirdhaarit karna",
    "availability": "upalabdhata",
    "prefer": "pasand karna",
    "details": "jankari",
    "contact": "sampark",
    "confirmation": "pustika",
    "specific": "vishesh",
    "focus": "dhyan",
    "thank you": "dhanyavaad",
    "great": "bahut badhiya",
    "perfect": "bilkul sahi",
    "sorry": "maaf kijiye",
    "please": "kripaya",
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
class ConversationState:
    def __init__(self, company_info):
        self.company_info = company_info
        self.current_state = "introduction"
        self.demo_scheduled = False
        self.preferred_day = None
        self.preferred_time = None
        self.preferred_contact_method = None
        self.contact_details = None
        self.call_history = []
        self.interested = None
        self.objections = []

    def update_state(self, customer_input, agent_response):
        """Update conversation state based on the latest interaction"""
        # Record the interaction
        self.call_history.append({
            "customer": customer_input,
            "agent": agent_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Update state based on conversation progress
        if self.current_state == "introduction":
            if any(word in customer_input.lower() for word in ["yes", "sure", "okay", "go ahead"]):
                self.current_state = "explaining_benefits"
                self.interested = True
            elif any(word in customer_input.lower() for word in ["not interested", "busy", "don't need", "no thanks"]):
                self.current_state = "closing"
                self.interested = False

        elif self.current_state == "explaining_benefits":
            if any(word in customer_input.lower() for word in ["schedule", "demo", "appointment", "meeting"]):
                self.current_state = "scheduling"
            elif any(word in customer_input.lower() for word in ["expensive", "cost", "complex", "difficult"]):
                self.current_state = "handling_objections"
                self.objections.append(customer_input)

        elif self.current_state == "scheduling":
            if self.preferred_day and not self.preferred_time:
                times = self.available_slots[self.preferred_day]
                response = f"Great! I've marked you down for {self.preferred_day}. We have slots at {', '.join(times)}. Which time works best for you?"
            elif self.preferred_day and self.preferred_time and not self.preferred_contact_method:
                response = f"Perfect! I've scheduled you for {self.preferred_day} at {self.preferred_time}. Would you prefer the demo via Zoom, Google Meet, or phone call?"
            elif self.preferred_contact_method and not self.contact_details:
                response = f"Great! Could you please share your {self.preferred_contact_method} details so I can send you the confirmation and meeting details?"
            else:
                response = "Let's schedule a demo. Which day would work best for you - Monday, Tuesday, Wednesday, Thursday, or Friday?"

        elif self.current_state == "confirmation":
            if any(word in customer_input.lower() for word in ["yes", "confirm", "sure"]):
                self.demo_scheduled = True
                self.current_state = "closing"

        return self.current_state

    def get_state_summary(self):
        """Return a summary of the current conversation state"""
        summary = f"Current stage: {self.current_state.replace('_', ' ').title()}"

        if self.interested is not None:
            summary += f"\nCustomer interest level: {'Interested' if self.interested else 'Not clearly interested'}"

        if self.objections:
            summary += f"\nRaised objections: {', '.join(self.objections)}"

        if self.preferred_day:
            summary += f"\nPreferred day: {self.preferred_day}"

        if self.preferred_time:
            summary += f"\nPreferred time: {self.preferred_time}"

        if self.preferred_contact_method:
            summary += f"\nPreferred contact method: {self.preferred_contact_method}"

        if self.contact_details:
            summary += f"\nContact details provided: {self.contact_details}"

        if self.demo_scheduled:
            summary += f"\nDemo scheduled: Yes"

        return summary

    def get_conversation_history(self):
        """Return the conversation history in a structured format"""
        if not self.call_history:
            return "No previous conversation."

        history = ""
        for i, interaction in enumerate(self.call_history[-3:]):  # Only include the last 3 interactions
            history += f"Customer: {interaction['customer']}\n"
            history += f"Agent: {interaction['agent']}\n"

        return history

# PART 7: AUDIO OUTPUT FUNCTIONS
def save_audio_output(text, filename):
    """Save text as audio file"""
    filepath = os.path.join(config["audio_output_dir"], filename)
    engine = pyttsx3.init()
    engine.save_to_file(text, filepath)
    engine.runAndWait()
    print(f"Audio saved to: {filepath}")
    return filepath

def save_conversation_as_audio(conversation_history, filename):
    """Save entire conversation as a single audio file"""
    full_text = ""
    for i, interaction in enumerate(conversation_history):
        if i > 0:
            full_text += "Customer: " + interaction['customer'] + ". "
        full_text += "Agent: " + interaction['agent'] + ". "
    
    filepath = os.path.join(config["audio_output_dir"], filename)
    engine = pyttsx3.init()
    engine.save_to_file(full_text, filepath)
    engine.runAndWait()
    print(f"Full conversation audio saved to: {filepath}")
    return filepath

# PART 8: SIMPLIFIED AGENT IMPLEMENTATION - TEXT-ONLY VERSION
def hinglish_to_english(text):
    """Convert Hinglish text to English
    This is a simplified implementation that will try to convert
    common Hinglish words back to English"""
    
    # Create reverse dictionary for Hinglish to English conversion
    english_dict = {v: k for k, v in hinglish_dict.items()}
    
    words = text.split()
    english_words = []
    for word in words:
        lower_word = word.lower()
        # If word exists in our dictionary, translate it
        if lower_word in english_dict:
            english_words.append(english_dict[lower_word])
        else:
            english_words.append(word)
    return " ".join(english_words)

class HinglishERPAgent:
    def __init__(self):
        print("Initializing ERP Demo Scheduling Agent...")
        self.companies = config["companies"]
        self.available_slots = config["available_slots"]
        self.erp_info = erp_product_info
        self.current_company = None
        self.conversation_state = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audio_files = []

    def start_new_call(self, company_idx=None):
        """Start a new cold call with a selected company"""
        # If no company_idx provided, randomly select a company
        if company_idx is None:
            company_idx = random.randint(0, len(self.companies) - 1)

        self.current_company = self.companies[company_idx]
        self.conversation_state = ConversationState(self.current_company)

        # Generate introduction message
        intro_message = self.generate_introduction()
        
        # Save intro as audio
        audio_file = save_audio_output(
            intro_message, 
            f"{self.session_id}_intro.mp3"
        )
        self.audio_files.append(audio_file)
        
        return intro_message

    def generate_introduction(self):
        """Generate an introduction message for the cold call"""
        company_name = self.current_company["name"]
        industry = self.current_company["industry"]

        intro_templates = [
            f"Hello, this is Rahul from VyaparERP Pro. Am I speaking with someone from the IT or operations department at {company_name}?",
            f"Namaste, main Rahul bol raha hoon VyaparERP Pro se. Kya main {company_name} ke IT ya operations department se baat kar sakta hoon?",
            f"Good morning, my name is Rahul calling from VyaparERP Pro. I'd like to discuss how our ERP solution is helping companies in the {industry} sector improve their operations. Do you have a few minutes to talk?"
        ]

        # Select a random introduction template
        intro = random.choice(intro_templates)

        # Convert to Hinglish if it's in English
        if not any(word in intro for word in ["Namaste", "main", "bol raha"]):
            intro = english_to_hinglish(intro)

        return intro

    def process_response(self, customer_input):
        """Process customer response and generate agent's next message"""
        # Convert Hinglish input to English for better processing
        english_input = hinglish_to_english(customer_input)

        # Get the current state
        state = self.conversation_state.current_state
        company_needs = self.current_company["needs"]
        industry = self.current_company["industry"]

        # Generate response based on conversation state
        if state == "introduction":
            if any(word in english_input.lower() for word in ["yes", "sure", "okay", "go ahead"]):
                response = f"Thank you for your time. VyaparERP Pro is a comprehensive ERP solution specially designed for {industry} companies. Our clients typically see a 30% increase in operational efficiency. Given your focus on {', '.join(company_needs)}, I believe our solution could significantly improve your business processes. Would you like to hear more about these specific features?"
            else:
                response = f"I understand you're busy. VyaparERP Pro has helped many {industry} businesses streamline their operations, especially in {company_needs[0]}. Could I briefly explain how our solution addresses your specific needs?"

        elif state == "explaining_benefits":
            if "sales" in english_input.lower():
                response = "To improve sales, VyaparERP Pro offers advanced Customer Relationship Management (CRM) and Business Intelligence tools. These features help you track leads, manage customer interactions, and analyze sales performance. Would you like to see a demo of these features?"
            elif any(word in english_input.lower() for word in ["schedule", "demo", "appointment", "meeting"]):
                self.conversation_state.current_state = "scheduling"
                avail_days = list(self.available_slots.keys())
                response = f"I'd be happy to schedule a demo for you. We have availability this {avail_days[0]} at {self.available_slots[avail_days[0]][0]} or {avail_days[2]} at {self.available_slots[avail_days[2]][1]}. Which would work better for you?"
            else:
                feature_descriptions = []
                for need in company_needs:
                    if need in erp_product_info["features"]:
                        feature_descriptions.append(f"{need}: {erp_product_info['features'][need]['benefits']}")

                response = f"Our VyaparERP Pro would be perfect for your needs. Specifically: {'. '.join(feature_descriptions)}. We offer 24/7 support from our India-based team and implementation takes only 4-6 weeks. Would you be interested in scheduling a demo to see these features in action?"

        elif state == "handling_objections":
            if "expensive" in english_input.lower():
                response = "I understand cost is a concern. VyaparERP Pro offers flexible pricing plans starting at ₹15,000 per month, and our clients typically see a return on investment within 6 months. Would you like to hear more about our pricing options?"
            elif "complex" in english_input.lower():
                response = "I understand ease of use is important. VyaparERP Pro is designed with user-friendly interfaces and comes with dedicated training and support to ensure a smooth transition. Would you like to see a demo of how easy it is to use?"
            else:
                response = "I understand your concerns. Many of our clients had similar thoughts before trying our solution. What makes VyaparERP Pro different is its ease of use and specialized features for your industry. Would it help to see a demonstration of how it works in practice?"

        elif state == "scheduling":
            if self.conversation_state.preferred_day and not self.conversation_state.preferred_time:
                times = self.available_slots[self.conversation_state.preferred_day]
                response = f"Great! I've marked you down for {self.conversation_state.preferred_day}. We have slots at {', '.join(times)}. Which time works best for you?"
            elif self.conversation_state.preferred_day and self.conversation_state.preferred_time and not self.conversation_state.preferred_contact_method:
                response = f"Perfect! I've scheduled you for {self.conversation_state.preferred_day} at {self.conversation_state.preferred_time}. Would you prefer the demo via Zoom, Google Meet, or phone call?"
            elif self.conversation_state.preferred_contact_method and not self.conversation_state.contact_details:
                response = f"Great! Could you please share your {self.conversation_state.preferred_contact_method} details so I can send you the confirmation and meeting details?"
            else:
                response = "Let's schedule a demo. Which day would work best for you - Monday, Tuesday, Wednesday, Thursday, or Friday?"

        elif state == "confirmation":
            if any(word in english_input.lower() for word in ["yes", "confirm", "sure"]):
                self.conversation_state.current_state = "closing"
                self.conversation_state.demo_scheduled = True
                response = f"Excellent! I've confirmed your demo for {self.conversation_state.preferred_day} at {self.conversation_state.preferred_time}. Our product specialist will contact you via {self.conversation_state.preferred_contact_method}. They'll customize the demo to showcase our {', '.join(company_needs)} features that are most relevant to your business. Thank you for your time today, and we look forward to showing you how VyaparERP Pro can transform your operations. Have a great day!"

        elif state == "closing":
            response = "Thank you for your time today! If you have any questions before the demo, please feel free to contact me. Have a wonderful day!"

        # Convert response to Hinglish
        hinglish_response = english_to_hinglish(response)

        # Update conversation state
        self.conversation_state.update_state(english_input, hinglish_response)
        
        # Save response as audio file
        msg_count = len(self.conversation_state.call_history)
        audio_file = save_audio_output(
            hinglish_response, 
            f"{self.session_id}_response_{msg_count}.mp3"
        )
        self.audio_files.append(audio_file)

        return hinglish_response

    def get_current_state(self):
        """Return the current conversation state"""
        if self.conversation_state:
            return self.conversation_state.get_state_summary()
        return "No active conversation."

    def get_conversation_history(self):
        """Return the conversation history"""
        if self.conversation_state:
            return self.conversation_state.get_conversation_history()
        return "No conversation history."
        
    def save_full_conversation_audio(self):
        """Save the full conversation as a single audio file"""
        if not self.conversation_state or not self.conversation_state.call_history:
            return "No conversation to save."
            
        filepath = save_conversation_as_audio(
            self.conversation_state.call_history,
            f"{self.session_id}_full_conversation.mp3"
        )
        return filepath

# PART 9: INTERACTIVE DEMO FOR GOOGLE COLAB
def run_interactive_demo():
    """Run an interactive demo of the Hinglish ERP cold calling agent"""
    print("Initializing Hinglish ERP Cold Calling Agent Demo...")
    agent = HinglishERPAgent()

    print("\n=== Demo Scheduling Agent for VyaparERP Pro ===")
    print("This demo simulates a cold call to schedule an ERP system demonstration.")
    print("You'll play the role of the customer responding to the agent's call.")
    print(f"Audio files will be saved to: {os.path.abspath(config['audio_output_dir'])}")

    # Display available companies
    print("\nAvailable companies for simulation:")
    for i, company in enumerate(config["companies"]):
        print(f"{i+1}. {company['name']} ({company['industry']})")

    # Ask user to select a company
    while True:
        try:
            company_idx = int(input("\nSelect a company (1-5) or press Enter for random selection: ")) - 1
            if 0 <= company_idx < len(config["companies"]):
                break
            else:
                print("Invalid selection. Please choose a number between 1 and 5.")
        except ValueError:
            company_idx = None
            break

    # Start the call
    intro_message = agent.start_new_call(company_idx)
    print("\n--- Call Started ---")
    print(f"Agent: {intro_message}")
    
    # Play the introduction (optional)
    engine.say(intro_message)
    engine.runAndWait()

    # Interactive conversation loop
    while True:
        # Get customer input
        text_input = input("\nYour response (type 'quit' to end the demo): ")
        if text_input.lower() == 'quit':
            break

        # Process response
        agent_response = agent.process_response(text_input)
        print(f"\nAgent: {agent_response}")
        
        # Play the response (optional)
        engine.say(agent_response)
        engine.runAndWait()

        # Check if call should end
        if agent.conversation_state.current_state == "closing" and agent.conversation_state.demo_scheduled:
            print("\n--- Demo successfully scheduled! ---")
            print(agent.get_current_state())
            break
        elif agent.conversation_state.current_state == "closing" and not agent.conversation_state.demo_scheduled:
            print("\n--- Call ended without scheduling a demo ---")
            print(agent.get_current_state())
            break

    # Display call summary
    print("\n=== Call Summary ===")
    print(agent.get_current_state())
    
    # Save the full conversation as a single audio file
    full_audio_path = agent.save_full_conversation_audio()
    print(f"\nFull conversation audio saved to: {full_audio_path}")
    
    # List all generated audio files
    print("\n=== Generated Audio Files ===")
    for audio_file in agent.audio_files:
        print(f"- {audio_file}")
    
    # Display conversation history
    print("\n=== Conversation History ===")
    history = agent.conversation_state.call_history
    for i, interaction in enumerate(history):
        print(f"Customer: {interaction['customer']}")
        print(f"Agent: {interaction['agent']}")
        print(f"Timestamp: {interaction['timestamp']}")
        print("-" * 50)

# PART 10: RUN THE DEMO
if __name__ == "__main__":
    run_interactive_demo()

from IPython.display import Audio

# Replace 'audio_outputs/20250302_135914_intro.mp3' with the correct file path if necessary
Audio("audio_outputs/20250302_142728_full_conversation.mp3")
