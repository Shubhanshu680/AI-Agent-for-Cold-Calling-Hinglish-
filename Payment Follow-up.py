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
    "customers": [
        {"name": "Rahul Traders", "order_id": "ORD12345", "amount_due": 50000, "due_date": "2023-11-15", "contact": "rahul.traders@example.com"},
        {"name": "Priya Textiles", "order_id": "ORD67890", "amount_due": 75000, "due_date": "2023-11-20", "contact": "priya.textiles@example.com"},
        {"name": "Amit Electronics", "order_id": "ORD54321", "amount_due": 100000, "due_date": "2023-11-25", "contact": "amit.electronics@example.com"},
        {"name": "Neha Pharma", "order_id": "ORD98765", "amount_due": 60000, "due_date": "2023-11-30", "contact": "neha.pharma@example.com"},
        {"name": "Suresh Retail", "order_id": "ORD13579", "amount_due": 45000, "due_date": "2023-12-05", "contact": "suresh.retail@example.com"}
    ],
    "payment_methods": ["Bank Transfer", "UPI", "Credit Card", "Cheque"],
    "order_statuses": ["Pending", "Processing", "Shipped", "Delivered"]
}

# PART 3: PAYMENT/ORDER KNOWLEDGE BASE
# Define payment and order details
payment_order_info = {
    "reminder_templates": [
        "Hello, this is a reminder that your payment of ₹{amount_due} for order {order_id} is due on {due_date}. Please release the payment at the earliest.",
        "Namaste, ye reminder hai ki aapka ₹{amount_due} ka payment, order {order_id} ke liye, {due_date} tak due hai. Kripya payment jald se jald release karein.",
        "Hi, we noticed that your payment for order {order_id} is still pending. The due date is {due_date}. Kindly process the payment of ₹{amount_due} soon."
    ],
    "follow_up_templates": [
        "Hello, we haven't received your payment for order {order_id}. The due date was {due_date}. Please release ₹{amount_due} immediately to avoid late fees.",
        "Namaste, hume aapka payment order {order_id} ke liye nahi mila hai. Due date {due_date} thi. Kripya ₹{amount_due} ka payment turant release karein taaki late fees na lage.",
        "Hi, your payment for order {order_id} is overdue. The due date was {due_date}. Please release ₹{amount_due} at the earliest to avoid any issues."
    ],
    "payment_confirmation_templates": [
        "Thank you for releasing the payment of ₹{amount_due} for order {order_id}. We appreciate your promptness.",
        "Dhanyavaad, order {order_id} ke liye ₹{amount_due} ka payment release karne ke liye. Hum aapki turant karvahi ki kadar karte hain.",
        "We have received your payment of ₹{amount_due} for order {order_id}. Thank you for your cooperation."
    ],
    "order_placement_templates": [
        "Hello, we noticed that you haven't placed your order yet. Would you like to proceed with the order now?",
        "Namaste, humne dekha ki aapne abhi tak order nahi diya hai. Kya aap abhi order dena chahenge?",
        "Hi, your cart is still pending. Would you like to complete the order now?"
    ]
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
    "payment": "bhugtaan",
    "order": "order",
    "due": "niyat",
    "date": "tareekh",
    "reminder": "yaad dilaane wala",
    "release": "release karna",
    "amount": "raashi",
    "immediately": "turant",
    "pending": "pending",
    "received": "praapt",
    "appreciate": "kadar karna",
    "cooperation": "sahyog",
    "cart": "cart",
    "complete": "poorna karna",
    "proceed": "aage badhna",
    "noticed": "dekha",
    "issues": "samasyaayein",
    "late fees": "deri fees",
    "promptness": "turant karvahi",
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
class PaymentOrderState:
    def __init__(self, customer_info):
        self.customer_info = customer_info
        self.current_state = "reminder"
        self.payment_released = False
        self.order_placed = False
        self.call_history = []

    def update_state(self, customer_input, agent_response):
        """Update conversation state based on the latest interaction"""
        # Record the interaction
        self.call_history.append({
            "customer": customer_input,
            "agent": agent_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Update state based on conversation progress
        if self.current_state == "reminder":
            if any(word in customer_input.lower() for word in ["paid", "released", "done"]):
                self.current_state = "confirmation"
                self.payment_released = True
            elif any(word in customer_input.lower() for word in ["not yet", "busy", "later"]):
                self.current_state = "follow_up"

        elif self.current_state == "follow_up":
            if any(word in customer_input.lower() for word in ["paid", "released", "done"]):
                self.current_state = "confirmation"
                self.payment_released = True
            else:
                self.current_state = "closing"

        elif self.current_state == "confirmation":
            self.current_state = "closing"

        return self.current_state

    def get_state_summary(self):
        """Return a summary of the current conversation state"""
        summary = f"Current stage: {self.current_state.replace('_', ' ').title()}"
        summary += f"\nPayment released: {'Yes' if self.payment_released else 'No'}"
        summary += f"\nOrder placed: {'Yes' if self.order_placed else 'No'}"
        return summary

# PART 7: PAYMENT/ORDER FOLLOW-UP AGENT IMPLEMENTATION
class HinglishPaymentOrderAgent:
    def __init__(self):
        print("Initializing Payment/Order Follow-up Agent...")
        self.customers = config["customers"]
        self.payment_methods = config["payment_methods"]
        self.order_statuses = config["order_statuses"]
        self.current_customer = None
        self.payment_order_state = None

    def start_new_call(self, customer_idx=None):
        """Start a new follow-up call with a selected customer"""
        # If no customer_idx provided, randomly select a customer
        if customer_idx is None:
            customer_idx = random.randint(0, len(self.customers) - 1)

        self.current_customer = self.customers[customer_idx]
        self.payment_order_state = PaymentOrderState(self.current_customer)

        # Generate reminder message
        reminder_message = self.generate_reminder()
        return reminder_message

    def generate_reminder(self):
        """Generate a reminder message for payment/order follow-up"""
        customer_name = self.current_customer["name"]
        order_id = self.current_customer["order_id"]
        amount_due = self.current_customer["amount_due"]
        due_date = self.current_customer["due_date"]

        reminder_templates = payment_order_info["reminder_templates"]
        reminder = random.choice(reminder_templates).format(
            amount_due=amount_due,
            order_id=order_id,
            due_date=due_date
        )

        # Convert to Hinglish if it's in English
        if not any(word in reminder for word in ["Namaste", "kripya", "jald"]):
            reminder = english_to_hinglish(reminder)

        return reminder

    def process_response(self, customer_input):
        """Process customer response and generate agent's next message"""
        # Convert Hinglish input to English for better processing
        english_input = hinglish_to_english(customer_input)

        # Get the current state
        state = self.payment_order_state.current_state
        order_id = self.current_customer["order_id"]
        amount_due = self.current_customer["amount_due"]

        # Generate response based on conversation state
        if state == "reminder":
            if any(word in english_input.lower() for word in ["paid", "released", "done"]):
                response = payment_order_info["payment_confirmation_templates"][0].format(
                    amount_due=amount_due,
                    order_id=order_id
                )
            else:
                response = payment_order_info["follow_up_templates"][0].format(
                    amount_due=amount_due,
                    order_id=order_id,
                    due_date=self.current_customer["due_date"]
                )

        elif state == "follow_up":
            if any(word in english_input.lower() for word in ["paid", "released", "done"]):
                response = payment_order_info["payment_confirmation_templates"][1].format(
                    amount_due=amount_due,
                    order_id=order_id
                )
            else:
                response = "We will follow up again later. Thank you for your time."

        elif state == "confirmation":
            response = "Thank you for your cooperation. Have a great day!"

        elif state == "closing":
            response = "Thank you for your time. Have a great day!"

        # Convert response to Hinglish
        hinglish_response = english_to_hinglish(response)

        # Update conversation state
        self.payment_order_state.update_state(english_input, hinglish_response)

        return hinglish_response

    def get_current_state(self):
        """Return the current conversation state"""
        if self.payment_order_state:
            return self.payment_order_state.get_state_summary()
        return "No active conversation."

    def get_conversation_history(self):
        """Return the conversation history"""
        if self.payment_order_state:
            history = ""
            for i, interaction in enumerate(self.payment_order_state.call_history[-3:]):  # Only include the last 3 interactions
                history += f"Customer: {interaction['customer']}\n"
                history += f"Agent: {interaction['agent']}\n"
            return history
        return "No conversation history."

# PART 8: INTERACTIVE DEMO FOR GOOGLE COLAB
def run_interactive_follow_up():
    """Run an interactive demo of the Hinglish Payment/Order Follow-up Agent"""
    print("Initializing Hinglish Payment/Order Follow-up Agent Demo...")
    agent = HinglishPaymentOrderAgent()

    print("\n=== Payment/Order Follow-up Agent ===")
    print("This demo simulates a follow-up call for payment or order placement.")
    print("You'll play the role of the customer responding to the agent's reminders.")

    # Display available customers
    print("\nAvailable customers for simulation:")
    for i, customer in enumerate(config["customers"]):
        print(f"{i+1}. {customer['name']} (Order ID: {customer['order_id']}, Amount Due: ₹{customer['amount_due']})")

    # Ask user to select a customer
    while True:
        try:
            customer_idx = int(input("\nSelect a customer (1-5) or press Enter for random selection: ")) - 1
            if 0 <= customer_idx < len(config["customers"]):
                break
            else:
                print("Invalid selection. Please choose a number between 1 and 5.")
        except ValueError:
            customer_idx = None
            break

    # Start the call
    reminder_message = agent.start_new_call(customer_idx)
    print("\n--- Call Started ---")
    print(f"Agent: {reminder_message}")

    # Interactive conversation loop
    while True:
        # Get customer input
        text_input = input("\nYour response (type 'quit' to end the call): ")
        if text_input.lower() == 'quit':
            break

        # Process response
        agent_response = agent.process_response(text_input)
        print(f"\nAgent: {agent_response}")

        # Check if call should end
        if agent.payment_order_state.current_state == "closing":
            print("\n--- Call completed! ---")
            print(agent.get_current_state())
            break

    # Display call summary
    print("\n=== Call Summary ===")
    print(agent.get_current_state())
    print("\n=== Conversation History ===")
    print(agent.get_conversation_history())

# PART 9: RUN THE DEMO
if __name__ == "__main__":
    run_interactive_follow_up()