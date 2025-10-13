import asyncio
import json
import pdb
import random
import time

from google import genai
from google.genai import types
from jsonschema import ValidationError, validate

import os
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)


class GeminiBatchClient:
    def __init__(self, api_key, model, temperature, functions, schema):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.functions = functions
        self.schema = schema


    def _call(self, path):
        uploaded = self.client.files.upload(
            file=str(path),
            config=types.
            UploadFileConfig(mime_type="application/jsonl")
        )

        job = self.client.batches.create(
            model=self.model,
            src=uploaded.name
        )
        print(f"[Batch] created id={job.name}")

        return job.name

    def __call__(self, batch_path):
        return self._call(batch_path)

    def make_request_line(self, prompt, custom_id) -> dict:
        return {
            "key":custom_id,
            "request":{
                "contents":[{
                    "role":"user",
                    "parts":[{"text": prompt["user"]}]
                }],
                "systemInstruction":{"parts":[{"text": prompt["system"]}]},
                "generationConfig":{
                    "responseMimeType":"application/json",
                    "temperature": self.temperature,
                    "responseSchema": self.functions
                }
            }
        }

    def validate_with_schema(self, result):
        try:
            validate(instance=result, schema=self.schema)
        except ValidationError as e:
            raise ValidationError(f"Gemini predict_subdecision validation error: {e.message}") from e

        return result

    def generate_valid_json(self, jname):
        while True:
            info = self.client.batches.get(name=jname)
            status = info.state

            if status == "JOB_STATE_SUCCEEDED":
                break
            if status == "JOB_STATE_EXPIRED":
                raise RuntimeError(
                    f"[Batch] {jname} expired but no output_file_id was provided; "
                    "the batch produced no downloadable result."
                )
                break

            if status in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
                raise RuntimeError(f"[Batch] {jname} ended with status={status}")
            time.sleep(5)

        raw = self.client.files.download(file=info.dest.file_name)
        text = raw.decode("utf-8")

        validated: Dict[str, dict] = {}
        dropped:   list[str] = []

        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)

            document_id   = rec["key"]

            try:
                result = rec["response"]["candidates"][0]["content"]["parts"][0]["text"]
                result_json = json.loads(result)
            except (KeyError, json.JSONDecodeError):
                dropped.append(document_id); continue

            try:
                self.validate_with_schema(result_json)

                usage = rec["response"]["usageMetadata"]

                input_token = usage.get("promptTokenCount", 0)
                cached_token = usage.get("cachedContentTokenCount", None)
                output_token = usage.get("candidatesTokenCount", 0)
                reasoning_token = usage.get("thoughtsTokenCount", None)

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