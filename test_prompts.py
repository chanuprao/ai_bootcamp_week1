import os
import openai
from typing import List, Dict, Tuple
import time

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def test_prompt(model: str, prompt: str, description: str) -> Tuple[str, float]:
    """
    Test a specific prompt with a given model and return the response and time taken.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=1000
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        answer = response.choices[0].message.content
        print(f"Response: {answer}")
        print(f"Time taken: {response_time:.2f} seconds")
        
        return answer, response_time
        
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}", 0

def analyze_results(results: List[Dict]) -> None:
    """
    Analyze and compare the results from different models and prompts.
    """
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"Prompt Type: {result['description']}")
        print(f"Response Time: {result['time']:.2f}s")
        print(f"Answer: {result['answer'][:200]}...")
        
        # Simple accuracy check - look for countries with 3+ letter repetitions
        answer_lower = result['answer'].lower()
        if any(country in answer_lower for country in ['guinea-bissau', 'madagascar', 'panama', 'bahamas']):
            print("✅ Likely correct - mentioned countries with 3 letter repetitions")
        elif 'canada' in answer_lower and '2' in result['answer']:
            print("⚠️  Partially correct - Canada has 2 'a's, but not the maximum")
        else:
            print("❌ Likely incorrect or incomplete")

def main():
    """
    Main function to test different prompts and models.
    """
    # Define the prompts to test
    prompts = [
        {
            "description": "Basic Prompt",
            "text": "What country has the same letter repeated the most in its name?"
        },
        {
            "description": "Improved Prompt",
            "text": """Please analyze all country names systematically to find which country has the same letter repeated the most times in its name. 

Follow these steps:
1. Consider all internationally recognized countries
2. Count each letter repetition in each country name (case-insensitive)
3. List the top 5 countries with the most letter repetitions
4. Show your counting process for the top result
5. Provide the final answer with the exact count

For example, "Canada" has 'a' repeated 2 times, "Madagascar" has 'a' repeated 3 times."""
        },
        {
            "description": "Optimized Prompt",
            "text": """You are a precise analyst. Your task is to find which country name has the highest frequency of any single letter.

Instructions:
1. Consider all 195 UN-recognized sovereign countries
2. For each country name, count the frequency of each letter (A-Z, case-insensitive)
3. Identify the maximum frequency of any single letter in each country name
4. Rank countries by their maximum letter frequency
5. Show your detailed analysis for the top 3 results
6. Provide the final answer with exact letter counts

Example analysis format:
- "Canada": a=2, c=1, d=1, n=1 (max frequency: 2)
- "Madagascar": a=3, m=1, d=1, g=1, s=1, c=1, r=1 (max frequency: 3)

Be thorough and systematic in your counting."""
        }
    ]
    
    # Define models to test (adjust based on your API access)
    models = [
        "gpt-3.5-turbo",
        "gpt-4o-mini",  # More cost-effective than full GPT-4o
        "gpt-4o"  # If you have access
    ]
    
    results = []
    
    print("Testing OpenAI Models for Country Name Letter Repetition Question")
    print("Expected correct answers include: Guinea-Bissau, Madagascar, Panama, Bahamas (all with 3 'a's)")
    
    # Test each model with each prompt
    for model in models:
        for prompt in prompts:
            try:
                answer, response_time = test_prompt(
                    model=model,
                    prompt=prompt["text"],
                    description=prompt["description"]
                )
                
                results.append({
                    "model": model,
                    "description": prompt["description"],
                    "answer": answer,
                    "time": response_time
                })
                
                # Add delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                print(f"Failed to test {model} with {prompt['description']}: {e}")
                continue
    
    # Analyze results
    analyze_results(results)
    
    # Save results to file
    with open("test_results.txt", "w") as f:
        f.write("OpenAI Model Testing Results\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Prompt: {result['description']}\n")
            f.write(f"Time: {result['time']:.2f}s\n")
            f.write(f"Answer: {result['answer']}\n")
            f.write("-" * 30 + "\n\n")
    
    print(f"\nResults saved to test_results.txt")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Please set your OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    main() 