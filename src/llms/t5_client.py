import asyncio
import json
import pdb
import random
import time

from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import transformers
import torch


class T5Client:
    def __init__(self, **model_config):
        self.model_name = model_config.pop("model")
        self.load_model(model_config=model_config)

    def load_model(
        self, device_map: str = "cuda", model_config: dict = {}, cache_dir: str = None
    ):
        torch_dtype = model_config.pop("torch_dtype", None)
        if torch_dtype and isinstance(torch_dtype, str) and torch_dtype != "auto":
            torch_dtype = getattr(torch, torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **model_config
        )
        if "cuda" in device_map or "cpu" in device_map:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **model_config,
                trust_remote_code=True,
                **({ "cache_dir": cache_dir } if cache_dir else {}),
            )

        self.tokenizer = tokenizer
        self.model = model

    async def _call(self, prompt):
        if self.model_name.lower().startswith("google/flan"):
            input_text = prompt["user"]

            input_ids = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.model.device)
        elif self.model_name.lower().startswith("google/t5gemma"):
            messages = [
                # {"role": "system", "content": prompt["system"]}, # system role 지원 X
                {"role": "user", "content": prompt["user"]},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True
            ).to(self.model.device)

        generated_tokens = self.model.generate(**input_ids, max_new_tokens=100)
        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        ## Calculate Tokens
        input_token = int(input_ids["attention_mask"][0].sum().item())
        reasoning_token, cached_token = 0, 0
        output_token = len(generated_tokens[0])

        return response, input_token, cached_token, output_token, reasoning_token

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
