import asyncio
import json
import pdb
import random
import time

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


class MistralClient:
    def __init__(self, **model_config):
        self.model_name = model_config.pop("model")
        self.load_model(model_config=model_config)

    def load_model(
        self, device_map: str = "cuda", model_config: dict = {}, cache_dir: str = None
    ):
        config = AutoConfig.from_pretrained(self.model_name, **model_config)

        torch_dtype = model_config.pop("torch_dtype", None)
        if torch_dtype and isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **model_config)
        if "cuda" in device_map or "cpu" in device_map:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                config=config,
                **({ "cache_dir": cache_dir } if cache_dir else {}),
            )

        self.tokenizer = tokenizer
        self.model = model

    async def _call(self, prompt):
        messages = [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        response = self.model.generate(**inputs,
                                      max_new_tokens=100,
                                      pad_token_id=self.tokenizer.eos_token_id)

        generated_tokens = response[0][inputs["input_ids"].shape[-1]:]
        content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        parts = content.split("\n\n", 1)
        answer = parts[0].strip()
        explanation = parts[1].strip() if len(parts) > 1 else None

        ## Calculate Tokens
        input_token = int(inputs["attention_mask"][0].sum().item())
        reasoning_token, cached_token = 0, 0
        output_token = len(generated_tokens)

        return answer, input_token, cached_token, output_token, reasoning_token

    async def generate_valid_json(self, prompt: str) -> dict:
        retry_count = 0
        while True:
            try:
                return await self._call(prompt)
            except Exception as e:
                retry_count += 1

                if retry_count >= 50:
                    wait = 3600
                    retry_count = 0
                    print("f[RateLimit] Retries exceeded 5 times. Please wait an hour.")
                else:
                    wait = (2 ** (retry_count - 1)) + random.random()
                    print(f"[WARN] {type(e).__name__}, Retry after {wait:.1f}s ({retry_count}/5)...")
                await asyncio.sleep(wait)
