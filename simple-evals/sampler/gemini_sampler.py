import os
import time
from google import genai
from google.genai import types
from ..types import SamplerBase, SamplerResponse, MessageList


class GeminiCompletionSampler(SamplerBase):
    def __init__(
            self,
            model: str = "gemini-1.5-pro-002",
            system_message: str | None = None,
            temperature: float = 0.0,
            max_tokens: int = 4096,
            project: str | None = None,
            location: str = "us-central1",
            response_format: str | None = None,
            response_schema: dict | None = None,
    ):
        self.model_name = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.response_schema = response_schema

        # --- STORE CREDENTIALS (Don't create client yet) ---
        self.project_id = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.api_key = os.environ.get("GOOGLE_API_KEY")

        if not (self.project_id or self.api_key):
            raise ValueError("Missing credentials (GOOGLE_CLOUD_PROJECT or GOOGLE_API_KEY).")

    def _get_client(self):
        """Create a fresh client for the current thread to avoid socket conflicts."""
        if self.project_id:
            return genai.Client(vertexai=True, project=self.project_id, location=self.location)
        else:
            return genai.Client(api_key=self.api_key)

    def _convert_messages(self, messages: MessageList):
        gemini_contents = []
        system_instruction = self.system_message
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
            elif role == "assistant":
                gemini_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))
        return system_instruction, gemini_contents

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # --- INITIALIZE CLIENT HERE (THREAD-SAFE) ---
        client = self._get_client()

        system_instruction, contents = self._convert_messages(message_list)

        mime_type = "text/plain"
        if self.response_schema:
            mime_type = "application/json"
        elif self.response_format == "json_object":
            mime_type = "application/json"

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            candidate_count=1,
            system_instruction=system_instruction,
            response_mime_type=mime_type,
            response_schema=self.response_schema,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )

        max_retries = 10
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )

                text = response.text if response.text else ""

                usage_dict = None
                if response.usage_metadata:
                    usage_dict = {
                        "prompt_tokens": response.usage_metadata.prompt_token_count,
                        "completion_tokens": response.usage_metadata.candidates_token_count,
                        "total_tokens": response.usage_metadata.total_token_count
                    }

                return SamplerResponse(
                    response_text=text,
                    response_metadata={"usage": usage_dict, "finish_reason": "stop"},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                error_str = str(e)
                # Catch common temporary errors including "Bad file descriptor" (OSError 9)
                if "429" in error_str or "503" in error_str or "500" in error_str or "Errno 9" in error_str:
                    sleep_time = base_delay * (1.5 ** attempt)
                    time.sleep(sleep_time)
                    continue

                print(f"Gemini Fatal Error: {e}")
                raise e

        raise Exception(f"Gemini failed after {max_retries} retries")