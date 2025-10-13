import asyncio
import json
import pdb
import random
import time

from jsonschema import ValidationError, validate
from openai import AsyncOpenAI, OpenAI, RateLimitError


class GPTClient:
    def __init__(self, api_key, model, temperature, functions, function_call):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.functions = functions
        self.function_call = function_call


    async def _call(self, prompt):
        if self.model.startswith("gpt"):
            if self.model.startswith("gpt-5"):
                return await self.client.chat.completions.create(
                    model = self.model,
                    messages= [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    reasoning_effort=self.temperature,
                    functions=self.functions,
                    function_call=self.function_call
                    )
            else:   
                return await self.client.chat.completions.create(
                    model = self.model,
                    messages= [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    temperature=self.temperature,
                    functions=self.functions,
                    function_call=self.function_call
                    )
        elif self.model.startswith("o"):
            return await self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ],
                functions=self.functions,
                function_call=self.function_call
                )


    def validate_with_schema(self, result: dict):
        target_name = self.function_call["name"]
        try:
            schema = next(fn["parameters"] for fn in self.functions if fn["name"] == target_name)
        except StopIteration:
            raise ValueError(f"Function '{target_name}' not found in functions list")

        ### 유효하지 않는 경우, ValidationError 발생
        validate(instance=result, schema=schema)
        return result


    async def generate_valid_json(self, prompt: str) -> dict:
        retry_count = 0
        while True:
            try:
                response = await self._call(prompt)
                result = response.choices[0].message.function_call.arguments
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
