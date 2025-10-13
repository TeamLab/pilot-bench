import asyncio
import json
import pdb
import random
import time

from google import genai
from google.genai import types
from jsonschema import ValidationError, validate


class GeminiClient:
    def __init__(self, api_key, model, temperature, functions):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.functions = functions


    async def _call(self, prompt):
        return await self.client.aio.models.generate_content(
            model = self.model,
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt["user"])]
                )
            ],
            config = types.GenerateContentConfig(
                system_instruction=(
                    prompt["system"]
                ),
                response_mime_type="application/json",
                response_schema=self.functions
            )
        )

    
    def validate_with_schema(self, result):
        try:
            validate(instance=result, schema=self.functions)
        except ValidationError as e:
            raise ValidationError(f"Gemini predict_subdecision validation error: {e.message}") from e
        
        return result


    async def generate_valid_json(self, prompt):
        retry_count = 0
        while True:
            try:
                response = await self._call(prompt)
                result = response.text
                result_json = json.loads(result)
                valid_result = self.validate_with_schema(result_json)

                input_token = int((getattr(getattr(response, "usage_metadata", None), "prompt_token_count", 0) or 0))
                cached_token = int((getattr(getattr(response, "usage_metadata", None), "cached_content_token_count", 0) or 0))
                candidates_token = int((getattr(getattr(response, "usage_metadata", None), "candidates_token_count", 0) or 0))
                thought_token = int((getattr(getattr(response, "usage_metadata", None), "thoughts_token_count", 0) or 0))

                return valid_result, input_token, cached_token, candidates_token, thought_token
            
            except Exception as e:
                wait = (2 ** (retry_count - 1)) + random.random()
                print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({retry_count}/5)...")
                await asyncio.sleep(wait)
