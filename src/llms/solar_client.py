import asyncio
import json
import pdb
import random
import time

from jsonschema import ValidationError, validate
from openai import AsyncOpenAI, OpenAI, RateLimitError


class SolarClient:
    def __init__(self, api_key, model, reasoning_effort, temperature, functions, function_call):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.upstage.ai/v1"
        )
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.response_format = functions
        self._schema = self.response_format["json_schema"]["schema"]


    async def _call(self, prompt):
        if self.model.endswith("pro"):
            return await self.client.chat.completions.create(
                model= self.model,
                messages= [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                ],
                stream=False,
                temperature=self.temperature,
                response_format=self.response_format
            )
        elif self.model.endswith("pro2"):
            return await self.client.chat.completions.create(
                model= self.model,
                messages= [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                ],
                reasoning_effort=self.reasoning_effort, # low, medium, high
                stream=False,
                temperature=self.temperature,
                response_format=self.response_format
            )


    def validate_with_schema(self, result: dict):
        if self._schema is None:
            return result
        validate(instance=result, schema=self._schema)
        return result


    async def generate_valid_json(self, prompt: str) -> dict:
        retry_count = 0
        while True:
            try:
                response = await self._call(prompt)

                result = response.choices[0].message.content

                result_json = json.loads(result)
                valid_result = self.validate_with_schema(result_json)

                input_token = getattr(getattr(response, "usage", None), "prompt_tokens", 0)
                cached_token = getattr(getattr(getattr(response, "usage", None), "prompt_tokens_details", None), "cached_tokens", 0)
                output_token = getattr(getattr(response, "usage", None), "completion_tokens", 0)
                reasoning_token = getattr(getattr(getattr(response, "usage", None), "completion_tokens_details", None), "reasoning_tokens", 0)

                return valid_result, input_token, cached_token, output_token, reasoning_token

            except (RateLimitError, ValidationError) as e:
                retry_count += 1

                if retry_count >= 5:
                    wait = 3600
                    retry_count = 0
                    print("f[RateLimit] Retries exceeded 5 times. Please wait an hour.")
                else:
                    wait = (2 ** (retry_count - 1)) + random.random()
                    print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({retry_count}/5)...")
                    await asyncio.sleep(wait)
