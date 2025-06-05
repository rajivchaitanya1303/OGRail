# gatekeeper.py

from llm_connector import client
from guardrail import LLMGuardrail

class Gatekeeper:
    def __init__(self, guardrail: LLMGuardrail):
        self.guardrail = guardrail

    def query_llm(self, messages):
        """
        Query the LLM using a list of chat messages.
        """
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",  # Ensure this model name is correct
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during LLM query: {e}")
            return None

    def validate(self, messages):
        """
        Validate the last user message and the LLM's response based on guardrails.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'.

        Returns:
            dict: Result indicating 'allowed' or 'blocked' with reasons and details.
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("Input must be a list of messages.")

        # Extract the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content")
                break

        if last_user_message is None:
            raise ValueError("No user message found to validate.")

        # Step 1: Validate user input
        input_valid, input_issues = self.guardrail.validate_input(last_user_message, llm_response=None)

        if not input_valid:
            return {
                "status": "blocked",
                "reason": "Input validation failed",
                "issues": input_issues
            }

        # Step 2: Query the LLM
        llm_response = self.query_llm(messages)

        if llm_response is None:
            return {
                "status": "error",
                "reason": "LLM query failed"
            }

        # Step 3: Validate LLM output
        output_valid, output_issues = self.guardrail.validate_output(llm_response)

        if not output_valid:
            return {
                "status": "blocked",
                "reason": "Output validation failed",
                "issues": output_issues
            }

        return {
            "status": "allowed",
            "response": llm_response
        }

    def process_user_input(self, user_input):
        """
        Process user input by validating it and querying the LLM.

        Returns:
            dict: A dictionary with validation results and the response from LLM.
        """
        messages = [{"role": "user", "content": user_input}]

        # Analyze sentiment + context before LLM call
        sentiment, context_info = self.guardrail.analyze_context_and_intent(user_input)

        # Validate input and query LLM
        validation_results = self.validate(messages)

        if validation_results["status"] == "allowed":
            return {
                "input_validation": "Valid",
                "sentiment": sentiment,
                "context_summary": context_info,
                "response_validation": [],
                "llm_response": validation_results["response"]
            }
        else:
            return {
                "input_validation": "Invalid",
                "sentiment": sentiment,
                "context_summary": context_info,
                "response_validation": validation_results.get("issues", []),
                "llm_response": None
            }