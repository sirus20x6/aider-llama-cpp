import hashlib
import json
import time

from aider.dump import dump  # noqa: F401
from aider.exceptions import LiteLLMExceptions
from aider.llm import litellm

try:
    import requests
except ImportError:
    requests = None
    
    
from types import SimpleNamespace


# from diskcache import Cache


CACHE_PATH = "~/.aider.send.cache.v1"
CACHE = None
# CACHE = Cache(CACHE_PATH)

RETRY_TIMEOUT = 60


def send_completion(
    model_name,
    messages,
    functions,
    stream,
    temperature=0,
    extra_params=None,
):
    # Handle llama-cpp server backend
    if model_name.startswith("llama-cpp/"):
        if not requests:
            raise ImportError(
                "Requests library not installed. Install with: pip install requests"
            )

        # Convert messages to prompt using ChatML format
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "ASSISTANT: "

        # Build request payload
        payload = {
            "prompt": prompt,
            "stream": stream,
            "temperature": temperature
        }

        try:
            url = "http://localhost:8080/completion"
            if stream:
                response = requests.post(url, json=payload, stream=True)
                response.raise_for_status()
                def generate():
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            line = line.decode('utf-8')
                            # Server uses SSE format with "data: " prefix
                            if not line.startswith("data: "):
                                continue
                            # Remove the "data: " prefix
                            line = line[6:]
                            chunk = json.loads(line)
                            
                            # Convert dict to object with attributes that aider expects
                            response_obj = SimpleNamespace()
                            response_obj.choices = [SimpleNamespace()]
                            response_obj.choices[0].delta = SimpleNamespace()
                            response_obj.choices[0].delta.content = chunk.get("content", "")
                            response_obj.choices[0].finish_reason = "stop" if chunk.get("stop", False) else None
                            
                            yield response_obj

                        except Exception as e:
                            print(f"Error processing response line: {e}")
                            continue
                            
                return hashlib.sha1(b"llama_server"), generate()
            else:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Convert dict to object with attributes that aider expects
                response_obj = SimpleNamespace()
                response_obj.choices = [SimpleNamespace()]
                response_obj.choices[0].message = SimpleNamespace()
                response_obj.choices[0].message.content = data.get("content", "")
                response_obj.choices[0].finish_reason = "stop" if data.get("stop", False) else None
                
                return hashlib.sha1(b"llama_server"), response_obj

        except Exception as e:
            raise RuntimeError(f"Llama.cpp server completion failed: {str(e)}")

    # Handle litellm completion
    kwargs = dict(
        model=model_name,
        messages=messages,
        stream=stream,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    if functions is not None:
        function = functions[0]
        kwargs["tools"] = [dict(type="function", function=function)]
        kwargs["tool_choice"] = {"type": "function", "function": {"name": function["name"]}}

    if extra_params is not None:
        kwargs.update(extra_params)

    key = json.dumps(kwargs, sort_keys=True).encode()
    hash_object = hashlib.sha1(key)

    if not stream and CACHE is not None and key in CACHE:
        return hash_object, CACHE[key]

    res = litellm.completion(**kwargs)

    if not stream and CACHE is not None:
        CACHE[key] = res

    return hash_object, res


def simple_send_with_retries(model_name, messages, extra_params=None):
    litellm_ex = LiteLLMExceptions()

    retry_delay = 0.125
    while True:
        try:
            kwargs = {
                "model_name": model_name,
                "messages": messages,
                "functions": None,
                "stream": False,
                "extra_params": extra_params,
            }

            _hash, response = send_completion(**kwargs)
            if not response or not hasattr(response, "choices") or not response.choices:
                return None
            return response.choices[0].message.content
        except litellm_ex.exceptions_tuple() as err:
            ex_info = litellm_ex.get_ex_info(err)

            print(str(err))
            if ex_info.description:
                print(ex_info.description)

            should_retry = ex_info.retry
            if should_retry:
                retry_delay *= 2
                if retry_delay > RETRY_TIMEOUT:
                    should_retry = False

            if not should_retry:
                return None

            print(f"Retrying in {retry_delay:.1f} seconds...")
            time.sleep(retry_delay)
            continue
        except AttributeError:
            return None
