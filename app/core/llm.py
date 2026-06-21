from typing import Any, Dict, List
import os
from litellm import acompletion

class LLMServiceRegistry:
    """
    Shared LLM service registry for proxying inference calls.
    Provides a standardized interface for interacting with various LLMs.
    """
    
    @classmethod
    async def generate_json(cls, model_name: str, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a JSON response from the specified LLM.
        
        Args:
            model_name: The name of the model to use.
            messages: The list of messages comprising the prompt context.
            **kwargs: Additional parameters for the litellm acompletion call.
            
        Returns:
            The string content of the response, expected to be parseable JSON.
        """
        response = await acompletion(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            api_base=os.getenv("LITELLM_BASE_URL"),
            api_key=os.getenv("LITELLM_API_KEY"),
            **kwargs
        )
        return response.choices[0].message.content

    @classmethod
    async def generate_embedding(cls, text: str) -> List[float]:
        """
        Generate a 1536-dimensional embedding vector for the provided text.
        """
        from litellm import aembedding
        response = await aembedding(
            model="text-embedding-ada-002",
            input=text,
            api_base=os.getenv("LITELLM_BASE_URL"),
            api_key=os.getenv("LITELLM_API_KEY")
        )
        return response.data[0]["embedding"]
