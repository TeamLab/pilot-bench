import asyncio
import json
import time
import pdb
import random

import anthropic
from anthropic import AI_PROMPT, HUMAN_PROMPT
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from anthropic._exceptions import OverloadedError, RateLimitError
from jsonschema import ValidationError, validate


class ClaudeBatchClient:
    def __init__(self, api_key, model, temperature, max_tokens, functions, timeout):
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = functions


    def make_request_line(self, prompt, custom_id) -> dict:
        return Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
                tools=self.tools,
                tool_choice={"type": "tool", "name": self.tools[0]["name"]},
            ),
        )

    def _call(self, files):
        res = self.client.messages.batches.create(
            requests=files
        )
        print(f"[Batch] created id={res.id}")
        return res.id

    def __call__(self, requests_files):
        return self._call(requests_files)

    def validate_with_schema(self, result):
        try:
            validate(instance=result, schema=self.tools[0].get("input_schema"))
        except ValidationError as e:
            raise ValidationError(f"Claude predict_subdecision validation error: {e.message}") from e

        return result

    def generate_valid_json(self, r_id):
        while True:
            batch = self.client.messages.batches.retrieve(r_id)

            status = batch.processing_status
            if status == "ended":
                break

            if status == "canceling":
                raise RuntimeError(f"[Batch] {r_id} ended with status={status}")
            time.sleep(5)

        raw = self.client.messages.batches.results(r_id)

        validated: Dict[str, dict] = {}
        dropped:   list[str] = []

        for ln in raw:
            document_id = ln.custom_id
            rtype = ln.result.type

            if rtype == "succeeded":
                msg = ln.result.message

                try:
                    if msg.content[0].type == "tool_use":
                        result = msg.content[0].input
                    elif msg.content[0].type == "text":
                        result = msg.content[0].text

                    if isinstance(result, dict):
                        result_json = result
                    elif isinstance(result, str):
                        result_json = json.loads(result)

                except (KeyError, json.JSONDecodeError):
                    dropped.append(document_id)
                    continue
            else:
                dropped.append(document_id)
                continue

            try:
                self.validate_with_schema(result_json)

                usage = msg.usage

                input_token = getattr(usage, "input_tokens", 0)
                c_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
                c_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                cached_token = c_create + c_read
                output_token = getattr(usage, "output_tokens", 0)
                reasoning_token = 0

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