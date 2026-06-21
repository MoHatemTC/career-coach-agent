import os
import asyncio
from dotenv import load_dotenv
from litellm import acompletion

# Load the environment configurations
load_dotenv()

async def test_connection():
    print("Testing primitive connection to LiteLLM Proxy...")
    
    # Retrieve proxy configs
    api_base = os.getenv("LITELLM_BASE_URL")
    api_key = os.getenv("LITELLM_API_KEY")
    model_name = os.getenv("DEFAULT_MODEL", "azure/FW-Kimi-K2.6")
    
    print(f"Proxy URL: {api_base}")
    print(f"API Key length: {len(api_key) if api_key else 0} characters")
    print(f"Model configured: {model_name}")
    
    messages = [
        {"role": "user", "content": "Reply with exactly the word 'PING'."}
    ]
    
    try:
        # NOTE: Make sure the model string matches the alias setup on your proxy!
        response = await acompletion(
            model=model_name, 
            messages=messages,
            api_base=api_base,
            api_key=api_key
        )
        print("\n--- SUCCESS ---")
        print("Model Response:", response.choices[0].message.content)
        
    except Exception as e:
        print("\n--- FAILED ---")
        print("Error details:", str(e))

if __name__ == "__main__":
    asyncio.run(test_connection())
