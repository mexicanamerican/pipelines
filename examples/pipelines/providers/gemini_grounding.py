"""
title: Google GenAI Manifold Pipeline with Grounding
author: Marc Lopez (refactor by justinh-rahb)
date: 2024-06-06
version: 1.3
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI with grounding capabilities.
requirements: google-generativeai
environment_variables: GOOGLE_API_KEY
"""

import os
import asyncio
from typing import List, Union, Iterator, Dict
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    Tool,
    DynamicRetrievalConfig,
    HarmCategory,
    HarmBlockThreshold,
)


class Pipeline:
    """Google GenAI pipeline with grounding"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_API_KEY: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)
        GROUNDING_ENABLED: bool = Field(default=True)  # New valve for enabling/disabling grounding
        DYNAMIC_THRESHOLD: float = Field(default=0.3)  # New valve for dynamic threshold

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai"
        self.name = "Google: "

        self.valves = self.Valves(**{
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "USE_PERMISSIVE_SAFETY": False,
            "GROUNDING_ENABLED": True,
            "DYNAMIC_THRESHOLD": 0.3
        })
        self.pipelines = []

        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""

        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""

        print(f"on_valves_updated:{__name__}")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""

        if self.valves.GOOGLE_API_KEY:
            try:
                models = genai.list_models()
                self.pipelines = [
                    {
                        "id": model.name[7:],  # the "models/" part messeses up the URL
                        "name": model.display_name,
                    }
                    for model in models
                    if "generateContent" in model.supported_generation_methods
                    if model.name[:7] == "models/"
                ]
            except Exception:
                self.pipelines = [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Google, please update the API Key in the valves.",
                    }
                ]
        else:
            self.pipelines = []

    def pipe(
        self, user_message: str, model_id: str, messages: List[Dict], body: Dict
    ) -> Union[str, Iterator]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"

        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            
            contents = []
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        parts = []
                        for content in message["content"]:
                            if content["type"] == "text":
                                parts.append({"text": content["text"]})
                            elif content["type"] == "image_url":
                                image_url = content["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    image_data = image_url.split(",")[1]
                                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})
                                else:
                                    parts.append({"image_url": image_url})
                        contents.append({"role": message["role"], "parts": parts})
                    else:
                        contents.append({
                            "role": "user" if message["role"] == "user" else "model",
                            "parts": [{"text": message["content"]}]
                        })
            
            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            tools = []
            if self.valves.GROUNDING_ENABLED:
                tools.append(
                    Tool(
                        google_search_retrieval={
                            "dynamic_retrieval_config": {
                                "mode": "MODE_DYNAMIC",
                                "dynamic_threshold": self.valves.DYNAMIC_THRESHOLD,
                            }
                        }
                    )
                )

            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(
                    model_name=model_id,
                    system_instruction=system_message,
                    generation_config=generation_config,
                    tools=tools,
                )
            else:
                if system_message:
                    contents.insert(0, {"role": "user", "parts": [{"text": f"System: {system_message}"}]})
                
                model = genai.GenerativeModel(
                    model_name=model_id,
                    generation_config=generation_config,
                    tools=tools,
                )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            response = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=body.get("stream", False),
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return self.format_response(response)

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def format_response(self, response):
        """Format the response to include grounding sources and Google Search Suggestions."""
        formatted_response = {"response": "", "grounding_sources": [], "search_suggestions": []}

        for candidate in response.candidates:
            formatted_response["response"] += candidate.content.parts[0].text

            if hasattr(candidate, "grounding_metadata"):
                grounding_metadata = candidate.grounding_metadata
                for chunk in grounding_metadata.grounding_chunks:
                    formatted_response["grounding_sources"].append(chunk.web.uri)
                
                for query in grounding_metadata.web_search_queries:
                    formatted_response["search_suggestions"].append(query)

        return formatted_response


async def main():
    # Set the GOOGLE_API_KEY environment variable
    os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"

    pipeline = Pipeline()
    await pipeline.on_startup()

    # Example usage
    user_message = "Who won Wimbledon this year?"
    model_id = "gemini-1.5-pro-002"
    messages = [{"role": "user", "content": user_message}]
    body = {"stream": False}

    response = pipeline.pipe(user_message, model_id, messages, body)
    print(response)

    await pipeline.on_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
