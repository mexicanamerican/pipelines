"""
title: Google GenAI Manifold Pipeline with Grounding for Open-WebUI
author: Marc Lopez (refactored by justinh-rahb)
date: 2024-06-06
version: 1.4
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI with grounding capabilities.
requirements: google-generativeai, pydantic
environment_variables: GOOGLE_API_KEY
"""

import os
import asyncio
from typing import List, Union, Iterator, Dict, Optional
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    Tool,
    DynamicRetrievalConfig,
    HarmCategory,
    HarmBlockThreshold,
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    """Google GenAI pipeline with grounding"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)
        GROUNDING_ENABLED: bool = Field(default=True)  # New valve for enabling/disabling grounding
        DYNAMIC_THRESHOLD: float = Field(default=0.3)  # New valve for dynamic threshold

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai"
        self.name = "Google GenAI"

        self.valves = self.Valves()
        self.pipelines = []

        if not self.valves.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set in environment variables.")
            raise ValueError("GOOGLE_API_KEY is required.")

        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        logger.info("Pipeline startup initiated.")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()
        logger.info("Pipeline startup completed.")

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        logger.info("Pipeline shutdown initiated.")
        # Perform any necessary cleanup here
        logger.info("Pipeline shutdown completed.")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        logger.info("Valves updated. Reconfiguring pipeline.")
        if not self.valves.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set in environment variables.")
            raise ValueError("GOOGLE_API_KEY is required.")

        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()
        logger.info("Pipeline reconfiguration completed.")

    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""

        try:
            models = genai.list_models()
            self.pipelines = [
                {
                    "id": model.name.split("/")[-1],  # Extract model ID without prefix
                    "name": model.display_name,
                }
                for model in models
                if "generateContent" in model.supported_generation_methods and model.name.startswith("models/")
            ]

            if not self.pipelines:
                self.pipelines = [
                    {
                        "id": "no_models_available",
                        "name": "No available models found. Please check your API key and permissions.",
                    }
                ]
                logger.warning("No available models fetched from Google GenAI.")
            else:
                logger.info(f"Fetched {len(self.pipelines)} models from Google GenAI.")

        except Exception as e:
            logger.exception("Failed to fetch models from Google GenAI.")
            self.pipelines = [
                {
                    "id": "error",
                    "name": "Could not fetch models from Google. Please update the API Key in the valves.",
                }
            ]

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict],
        body: Dict
    ) -> Union[str, Iterator[str], Dict]:
        if not self.valves.GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set.")
            return "Error: GOOGLE_API_KEY is not set"

        try:
            # Ensure the model_id is correctly formatted
            if model_id.startswith("google_genai."):
                model_id = model_id[len("google_genai."):]

            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                logger.error(f"Invalid model name format: {model_id}")
                return f"Error: Invalid model name format: {model_id}"

            logger.info(f"Pipe function called for model: {model_id}")
            stream_mode = body.get("stream", False)
            logger.info(f"Stream mode: {stream_mode}")

            # Extract system message if present
            system_message = next(
                (msg["content"] for msg in messages if msg.get("role") == "system"),
                None
            )

            # Prepare contents by processing messages
            contents = self.prepare_contents(messages)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            tools = self.prepare_tools()

            # Instantiate the GenerativeModel
            model = genai.GenerativeModel(
                model_name=model_id,
                generation_config=generation_config,
                tools=tools,
            )

            if "gemini-1.5" in model_id and system_message:
                model.system_instruction = system_message
                logger.debug("System instruction set for the model.")

            # Set safety settings
            safety_settings = self.get_safety_settings(body)

            # Generate content
            response = model.generate_content(
                prompt=contents,
                safety_settings=safety_settings,
                stream=stream_mode,
            )

            if stream_mode:
                logger.info("Streaming response initiated.")
                return self.stream_response(response)
            else:
                logger.info("Formatting response.")
                return self.format_response(response)

        except Exception as e:
            logger.exception("Error generating content.")
            return f"An error occurred: {str(e)}"

    def prepare_contents(self, messages: List[Dict]) -> List[Dict]:
        """Process messages into the format required by GenAI."""
        contents = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if role not in ["user", "assistant", "system"]:
                logger.warning(f"Unknown role '{role}' in message. Skipping.")
                continue

            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            })
                        else:
                            parts.append({"image_url": image_url})
                    else:
                        logger.warning(f"Unknown content type '{part.get('type')}'. Skipping.")
                contents.append({"role": role, "parts": parts})
            elif isinstance(content, str):
                contents.append({
                    "role": role,
                    "parts": [{"text": content}]
                })
            else:
                logger.warning(f"Unsupported content format in message: {message}. Skipping.")
        logger.debug(f"Prepared contents for generation: {contents}")
        return contents

    def prepare_tools(self) -> List[Tool]:
        """Prepare tools for the GenerativeModel based on grounding settings."""
        tools = []
        if self.valves.GROUNDING_ENABLED:
            grounding_tool = Tool(
                google_search_retrieval=DynamicRetrievalConfig(
                    mode="MODE_DYNAMIC",
                    dynamic_threshold=self.valves.DYNAMIC_THRESHOLD,
                )
            )
            tools.append(grounding_tool)
            logger.debug("Grounding tool added to the model tools.")
        return tools

    def get_safety_settings(self, body: Dict) -> Optional[Dict]:
        """Determine safety settings based on valves and request body."""
        if self.valves.USE_PERMISSIVE_SAFETY:
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            logger.debug("Permissive safety settings applied.")
        else:
            safety_settings = body.get("safety_settings")
            logger.debug(f"Safety settings from body: {safety_settings}")
        return safety_settings

    def stream_response(self, response: Iterator) -> Iterator[str]:
        """Yield streamed response chunks."""
        try:
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("Error while streaming response.")
            yield f"An error occurred during streaming: {str(e)}"

    def format_response(self, response) -> Dict:
        """Format the response to include grounding sources and Google Search Suggestions."""
        formatted_response = {
            "response": "",
            "grounding_sources": [],
            "search_suggestions": []
        }

        try:
            for candidate in response.candidates:
                # Concatenate all text parts from the candidate's content
                for part in candidate.content.parts:
                    if "text" in part:
                        formatted_response["response"] += part["text"]

                # Extract grounding sources if available
                grounding_metadata = getattr(candidate, "grounding_metadata", None)
                if grounding_metadata:
                    for chunk in grounding_metadata.grounding_chunks:
                        formatted_response["grounding_sources"].append(chunk.web.uri)
                    for query in grounding_metadata.web_search_queries:
                        formatted_response["search_suggestions"].append(query)

            logger.debug(f"Formatted response: {formatted_response}")
            return formatted_response

        except Exception as e:
            logger.exception("Error formatting the response.")
            return {"response": f"An error occurred while formatting the response: {str(e)}"}

async def main():
    # Ensure the GOOGLE_API_KEY environment variable is set
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual API key

    try:
        pipeline = Pipeline()
    except ValueError as ve:
        logger.error(f"Pipeline initialization failed: {ve}")
        return

    await pipeline.on_startup()

    # Example usage
    user_message = "Who won Wimbledon this year?"
    model_id = "gemini-1.5-pro-002"
    messages = [{"role": "user", "content": user_message}]
    body = {
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 1500,
        "stop": ["\n"],
        # "safety_settings": { ... }  # Optional: Define if not using permissive safety
    }

    response = pipeline.pipe(user_message, model_id, messages, body)
    
    if isinstance(response, Iterator):
        # Handle streaming response
        async for chunk in response:
            print(chunk, end="")
    elif isinstance(response, dict):
        # Handle formatted response
        print("Response:", response.get("response", ""))
        print("Grounding Sources:", response.get("grounding_sources", []))
        print("Search Suggestions:", response.get("search_suggestions", []))
    else:
        # Handle plain string response (e.g., error messages)
        print(response)

    await pipeline.on_shutdown()

if __name__ == "__main__":
    asyncio.run(main())
