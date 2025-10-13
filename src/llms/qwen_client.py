import asyncio
import json
import pdb
import random
import time

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


class QwenClient:
    def __init__(self, **model_config):
        self.model_name = model_config.pop("model")
        self.load_model(model_config=model_config)

    def load_model(
        self, device_map: str = "cuda", model_config: dict = {}, cache_dir: str = None
    ):
        config = AutoConfig.from_pretrained(self.model_name, **model_config)
        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )
        torch_dtype = model_config.pop("torch_dtype", None)
        if torch_dtype == "auto":
            torch_dtype = None
        elif isinstance(torch_dtype, str):
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

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(**inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        ### Calculate Tokens
        output_token = len(output_ids[index:])
        reasoning_token = len(output_ids[:index])
        cached_token = 0
        input_token = int(inputs["attention_mask"][0].sum().item())

        return content, input_token, cached_token, output_token, reasoning_token

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
