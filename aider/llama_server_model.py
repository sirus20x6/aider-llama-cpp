import json
import requests
from typing import Dict, List, Optional, Union, Generator

class LlamaServerModel:
    """Client for llama.cpp server backend."""
    
    def __init__(self, server_url: str = "http://localhost:8080", **kwargs):
        """Initialize the llama.cpp server client.
        
        Args:
            server_url: URL of the running llama.cpp server
            **kwargs: Additional arguments are ignored since server is pre-configured
        """
        self.base_url = server_url.rstrip('/')
        self.streaming = True  # Always support streaming
        self.info = self._get_model_info()
        self.name = f"llama-cpp/{self.info.get('model', 'unknown')}"
        
    def _get_model_info(self) -> Dict:
        """Get model info from server props endpoint."""
        try:
            resp = requests.get(f"{self.base_url}/props")
            resp.raise_for_status()
            data = resp.json()
            
            # Combine relevant fields from props
            settings = data.get('default_generation_settings', {})
            return {
                'max_input_tokens': settings.get('n_ctx', 2048), 
                'max_output_tokens': settings.get('n_ctx', 2048),
                'supports_vision': False,  # Could detect from server capabilities 
                'litellm_provider': 'llama_server',
                'model': settings.get('model', 'unknown'),
                'mode': 'chat'
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get model info from server: {str(e)}")

    def completion(self,
                  messages: List[Dict[str, str]],
                  stream: bool = True,
                  temperature: float = 0.7,
                  functions: Optional[List] = None,
                  **kwargs) -> Union[Dict, Generator]:
        """Generate completion via llama.cpp server.
        
        Args:
            messages: List of message dicts with role and content 
            stream: Whether to stream the response
            temperature: Temperature parameter for sampling
            functions: Function specifications (not supported)
            **kwargs: Additional args are ignored since server is pre-configured
            
        Returns:
            Either a completion dict or a generator yielding completion chunks
        """
        if functions:
            raise ValueError("Function calling not supported by llama.cpp server")

        # Convert messages to prompt using ChatML format
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "ASSISTANT: "

        # Build request payload - only use minimal params
        payload = {
            "prompt": prompt,
            "stream": stream,
            "temperature": temperature
        }

        try:
            if stream:
                return self._stream_completion(payload)
            else:
                return self._blocking_completion(payload) 
        except Exception as e:
            raise RuntimeError(f"Llama.cpp server completion failed: {str(e)}")

    def _stream_completion(self, payload: Dict) -> Generator:
        """Stream completion from server."""
        response = requests.post(
            f"{self.base_url}/completion",
            json=payload, 
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield {
                    "choices": [{
                        "delta": {
                            "content": chunk.get("content", "")
                        }
                    }],
                    "stop": chunk.get("stop", False)
                }

    def _blocking_completion(self, payload: Dict) -> Dict:
        """Get blocking completion from server."""
        response = requests.post(
            f"{self.base_url}/completion",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        return {
            "choices": [{
                "message": {
                    "content": data.get("content", "")
                }
            }]
        }

    def token_count(self, text_or_messages) -> int:
        """Count tokens via server's tokenize endpoint.
        
        Args:
            text_or_messages: Text string or list of message dicts
            
        Returns:
            Number of tokens 
        """
        if isinstance(text_or_messages, str):
            text = text_or_messages
        else:
            # Combine messages into text
            text = ""
            for msg in text_or_messages:
                text += f"{msg['role']}: {msg['content']}\n"

        try:
            response = requests.post(
                f"{self.base_url}/tokenize",
                json={"content": text}
            )
            response.raise_for_status()
            return len(response.json()["tokens"])
        except Exception as e:
            raise RuntimeError(f"Token counting failed: {str(e)}")