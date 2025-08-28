"""
step1_basic_chatbot.py
Basic Console Health Chatbot
A simple chatbot that acknowledges user symptoms and provides basic responses
"""

def display_welcome():
    """Display welcome message and instructions"""
    print("=" * 50)
    print("WELCOME TO PUBLIC HEALTH ASSISTANT")
    print("=" * 50)
    print("I'm here to help you with basic health information.")
    print("Please describe your symptoms and I'll provide guidance.")
    print("Type 'quit' to exit the program.")
    print("-" * 50)

def get_user_input():
    """Get symptoms input from user"""
    return input("\nYour symptoms: ").strip()

def process_symptoms(symptoms):
    """Process user symptoms and generate appropriate response"""
    # Convert to lowercase for easier processing
    symptoms_lower = symptoms.lower()
    
    # Check for exit command
    if symptoms_lower in ['quit', 'exit', 'bye']:
        return None
    
    # Check if input is empty
    if not symptoms:
        return "Please describe your symptoms so I can help you better."
    
    # Generate acknowledgment response
    response = f"I understand you're experiencing: {symptoms}\n"
    response += "Thank you for sharing your symptoms with me.\n"
    response += "Please note: This is basic information only. "
    response += "For proper medical advice, consult a healthcare professional."
    
    return response

def display_response(response):
    """Display bot response to user"""
    print("\nHealth Assistant:")
    print("-" * 30)
    print(response)
    print("-" * 30)

def run_chatbot():
    """Main function to run the chatbot"""
    # Display welcome message
    display_welcome()
    
    # Main chat loop
    while True:
        # Get user input
        user_symptoms = get_user_input()
        
        # Process symptoms
        bot_response = process_symptoms(user_symptoms)
        
        # Check if user wants to quit
        if bot_response is None:
            print("\nThank you for using Public Health Assistant!")
            print("Stay healthy and take care!")
            break
        
        # Display response
        display_response(bot_response)

def main():
    """Entry point of the program"""
    try:
        run_chatbot()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please restart the program.")

# Run the chatbot when script is executed directly
if __name__ == "__main__":
    main()



