import json
import os
from datetime import datetime, UTC
import openai

# Initialize the OpenAI client pointing to Poe's gateway
client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY", ""),  # Ensure your POE_API_KEY is set in your environment variables
    base_url="https://api.poe.com/v1",
)

research_questions = [ #add questions here
]


# List the specific model identifiers you want to evaluate
# (Make sure these match the exact string identifiers listed in your Poe API profile)
target_models = [
    "gpt-3.5-turbo",
    "GPT-4-Turbo", # cutoff December 2023 but only slightly knows about october 7th 
    "gpt-4.1",
    "gpt-5.5",
]

# Replace the old dictionary entry with this:
compiled_results = {
    "metadata": {
        "timestamp": datetime.now(UTC).isoformat(),
        "prompts": research_questions,
    },
    "responses": {},
}

print(f"Starting data collection across {len(target_models)} models and {len(research_questions)} questions...")

for model in target_models:
    print(f"Querying {model}...")
    compiled_results["responses"][model] = []
    for question in research_questions:
        print(f"  Querying question: {question[:50]}...")
        try:
            # Switched to chat.completions.create for universal model compatibility
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
            )

            # Standard extraction format for chat completions
            output_text = response.choices[0].message.content

            # Map it to your results compilation
            compiled_results["responses"][model].append({
                "question": question,
                "status": "success",
                "content": output_text,
            })
            print(f"  Successfully retrieved response for question.")

        except Exception as e:
            print(f"  Error querying {model} for question: {str(e)}")
            compiled_results["responses"][model].append({
                "question": question,
                "status": "error",
                "error_message": str(e),
            })

# --- Hardcoded Output Storage ---
output_directory = "./research_data"
output_filename = "llm_bias_lexical_results.json"
output_path = os.path.join(output_directory, output_filename)

# Ensure the local directory exists
os.makedirs(output_directory, exist_ok=True)

# Write the compiled results out to a structured JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(compiled_results, f, indent=4, ensure_ascii=False)

print("\nData collection complete.")
print(f"Results successfully saved to: {os.path.abspath(output_path)}")