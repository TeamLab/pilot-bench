import asyncio
import json
import time
import pdb
import random

import anthropic
from anthropic import AI_PROMPT, HUMAN_PROMPT
from anthropic._exceptions import OverloadedError, RateLimitError
from jsonschema import ValidationError, validate


class ClaudeClient:
    def __init__(self, api_key, model, temperature, max_tokens, functions, timeout):
        self.client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = functions


    async def _call(self, prompt):
        return await self.client.messages.create(
            model=self.model,
            system=prompt["system"],
            messages=[
                {"role": "user", "content": prompt["user"]}
            ],
            tools=self.tools,
            tool_choice={"type": "tool", "name": self.tools[0]["name"]},
            temperature=self.temperature,
            max_tokens=self.max_tokens
            )
    

    def validate_with_schema(self, result):
        try:
            validate(instance=result, schema=self.tools[0].get("input_schema"))
        except ValidationError as e:
            raise ValidationError(f"Claude predict_subdecision validation error: {e.message}") from e
        
        return result
    

    async def generate_valid_json(self, prompt):
        retry_count = 0
        while True:
            try:
                response = await self._call(prompt)
                result_json = response.content[0].input
                valid_result = self.validate_with_schema(result_json)

                input_token = getattr(response.usage, "input_tokens", 0)
                c_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                c_read   = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                cached_token = c_create + c_read
                output_token = getattr(response.usage, "output_tokens", 0)
                reasoning_token = 0
                return valid_result, input_token, cached_token, output_token, reasoning_token
            
            except Exception as e:
                wait = (2 ** (retry_count - 1)) + random.random()
                print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({retry_count}/5)...")
                await asyncio.sleep(wait)
