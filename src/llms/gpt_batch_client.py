import json, datetime, os
from pathlib import Path

import pdb
import time

from openai import OpenAI
from jsonschema import ValidationError, validate
import logging

class GPTBatchClient:
    def __init__(self, api_key: str, model: str, functions: list, function_call: dict, window="24h", temperature: float | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model  = model
        self.window = window
        self.temperature = temperature
        self.functions = functions
        self.function_call = function_call

    def _call(self, batch_path):
        batch_input_files = self.client.files.create(
            file=open(batch_path, "rb"),
            purpose="batch"
        )

        res = self.client.batches.create(
            input_file_id=batch_input_files.id,
            endpoint="/v1/chat/completions",
            completion_window=self.window,
        )
        print(f"[Batch] created id={res.id}")
        return res.id

    def __call__(self, batch_path):
        return self._call(batch_path)

    def make_request_line(self, prompt, custom_id) -> dict:
        body = {
            "model":        self.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]},
            ],
            "tools":        self.functions,
            "tool_choice":  self.function_call,
        }

        supports_temp = (
            self.temperature is not None
            and not self.model.lower().startswith(("o"))
        )

        if supports_temp:
            body["temperature"] = self.temperature

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

    def validate_with_schema(self, result: dict):
        target_name = self.function_call["function"]["name"]
        try:
            schema = next(fn["function"]["parameters"] for fn in self.functions if fn["function"]["name"] == target_name)
        except StopIteration:
            raise ValueError(f"Function '{target_name}' not found in functions list")

        validate(instance=result, schema=schema)
        return result

    def generate_valid_json(self, r_id):
        while True:
            info = self.client.batches.retrieve(r_id)
            status = info.status

            if status == "completed":
                break
            if status == "expired":
                if not info.output_file_id:
                    raise RuntimeError(
                        f"[Batch] {r_id} expired but no output_file_id was provided; "
                        "the batch produced no downloadable result."
                    )
                break

            if status in ("failed", "cancelled"):
                raise RuntimeError(f"[Batch] {r_id} ended with status={status}")
            time.sleep(5)

        if info.output_file_id:
            raw = self.client.files.retrieve_content(info.output_file_id)
        else:
            return {}

        validated: Dict[str, dict] = {}
        dropped:   list[str] = []

        for ln in raw.splitlines():
            rec   = json.loads(ln)
            document_id   = rec["custom_id"]

            try:
                result = rec["response"]["body"]["choices"][0]["message"] \
                              ["tool_calls"][0]["function"]["arguments"]
                result_json = json.loads(result)
            except (KeyError, json.JSONDecodeError):
                dropped.append(document_id); continue

            try:
                self.validate_with_schema(result_json)

                usage = rec["response"]["body"]["usage"]

                input_token = usage.get("prompt_tokens", 0)
                cached_token = usage.get("prompt_tokens_details", None).get("cached_tokens", 0)
                output_token = usage.get("completion_tokens", 0)
                reasoning_token = usage.get("completion_tokens_details", None).get("reasoning_tokens", 0)

                validated[document_id] = {
                    "result":result_json,
                    "input_token":input_token,
                    "cached_token":cached_token,
                    "output_token":output_token,
                    "reasoning_token":reasoning_token
                }
            except ValidationError:
                dropped.append(document_id)

        if dropped:
            print(f"[WARN] {len(dropped)} responses invalid → skipped (e.g. {dropped[:2]}...)")

        return validated