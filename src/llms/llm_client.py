from .gpt_client import GPTClient
from .gpt_batch_client import GPTBatchClient
from .claude_client import ClaudeClient
from .claude_batch_client import ClaudeBatchClient
from .gemini_client import GeminiClient
from .gemini_batch_client import GeminiBatchClient
from .llama_client import LlamaClient
from .qwen_client import QwenClient
from .solar_client import SolarClient
from .mistral_client import MistralClient
from .deepseek_client import DeepSeekClient
from .t5_client import T5Client


def get_llm_client(model_name, api_key, **kwargs):
    name = model_name.lower()

    if "gpt" in name:
        return GPTClient(api_key, **kwargs)
    elif "claude"in name:
        return ClaudeClient(api_key, **kwargs)
    elif "gemini" in name:
        return GeminiClient(api_key, **kwargs)
    elif name == "solar":
        return SolarClient(api_key, **kwargs)
    elif name == "llama":
        return LlamaClient(**kwargs)
    elif name == "qwen":
        return QwenClient(**kwargs)
    elif name == "mistral":
        return MistralClient(**kwargs)
    elif name == "deepseek":
        return DeepSeekClient(**kwargs)
    elif name == "t5":
        return T5Client(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_llm_batch_client(model_name, api_key, **kwargs):
    name = model_name.lower()

    if "gpt" in name:
        return GPTBatchClient(api_key, **kwargs)
    elif "claude" in name:
        return ClaudeBatchClient(api_key, **kwargs)
    elif "gemini" in name:
        return GeminiBatchClient(api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")