from gatekeeper import Gatekeeper
from guardrail import LLMGuardrail
import logging

# logger 
logging.basicConfig(level=logging.ERROR)  
logger = logging.getLogger(__name__)

def main():
    print("🛡️ Welcome to the LLM Guardrail System\n")

    guardrail = LLMGuardrail()
    gatekeeper = Gatekeeper(guardrail)

    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting...")
                break

            if not user_input:
                print("⚠️ Empty input. Please try again.")
                continue

            # Process input through the gatekeeper
            validation_results = gatekeeper.process_user_input(user_input)

            print("\nValidation Summary:")
            print(f"  Input: {validation_results.get('input_validation')}")
            print(f"  Sentiment: {validation_results.get('sentiment')}")
            
            context_summary = validation_results.get('context_summary', {})
            print(f"  Context + Intent:")
            print(f"    - Intent: {context_summary.get('intent', 'N/A')}")
            print(f"    - Context: {context_summary.get('context', 'N/A')}\n")

            violations = validation_results.get("response_validation", [])

            if not violations:
                print(f"✅ LLM Response:\n  {validation_results.get('llm_response')}")
            else:
                print("⛔ Response Blocked by Guardrails:")
                for reason in violations:
                    print(f"  - {reason}")

            print("\n" + "-"*60 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Unhandled error in main loop: {e}")
        print("⚠️ An unexpected error occurred.")

if __name__ == "__main__":
    main()