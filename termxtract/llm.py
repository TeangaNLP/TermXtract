import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Protocol

THINKING = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
"""The reasoning that some models write before their answer."""


class LLMClient(Protocol):
    """A large language model that answers a prompt, such as `OpenAIClient`."""

    def complete(self, prompt: str) -> str:
        """
        Answer a prompt.

        Args:
            prompt (str): The prompt, sent as a single user message.

        Returns:
            str: The model's answer.
        """
        ...


class OpenAIClient:
    """
    A model served by the OpenAI API or any server compatible with it, such as Ollama,
    vLLM or llama.cpp.

    Answers are generated with temperature 0, so that they can be reproduced, and can be
    cached on disk, so that the same prompt is only sent once.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        max_tokens: int = 512,
        **options: Any,
    ):
        """
        Connect to the model.

        Args:
            model (str): The name of the model, such as "gpt-4o-mini" or, for Ollama,
                "qwen2.5:7b".
            base_url (Optional[str]): The URL of the API, such as
                "http://localhost:11434/v1" for Ollama, or None for the
                `OPENAI_BASE_URL` environment variable or, failing that, OpenAI's.
            api_key (Optional[str]): The API key, or None for the `OPENAI_API_KEY`
                environment variable. Local servers need none.
            cache_dir (Optional[Union[str, PathLike]]): A directory to keep each answer
                in, keyed by the model, the prompt and the options, or None to cache
                nothing.
            max_tokens (int): The maximum length of an answer, in tokens.
            **options: Further parameters of the chat completions request, such as
                `seed` or `reasoning_effort`.

        Raises:
            ImportError: If the optional 'llm' dependencies are missing.
        """
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAIClient requires the optional 'llm' dependencies. Install them "
                "with: pip install 'termxtract[llm]'"
            ) from e
        self.model = model
        self.options = {"temperature": 0, "max_tokens": max_tokens, **options}
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if base_url is not None and api_key is None:
            # The library needs a key, which a local server ignores
            api_key = os.environ.get("OPENAI_API_KEY", "none")
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def complete(self, prompt: str) -> str:
        """
        Answer a prompt, from the cache if it has been answered before.

        Args:
            prompt (str): The prompt, sent as a single user message.

        Returns:
            str: The model's answer, without any reasoning in `<think>` tags.
        """
        path = None
        if self.cache_dir is not None:
            key = json.dumps([self.model, prompt, self.options], sort_keys=True)
            name = hashlib.sha256(key.encode()).hexdigest()
            path = self.cache_dir / name[:2] / f"{name}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))["answer"]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.options,
        )
        answer = THINKING.sub("", response.choices[0].message.content or "").strip()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write the whole file before it takes the name, so a run stopped while
            # writing leaves no partial answer, and requests sent at once do not clash
            partial = path.with_suffix(f".{os.getpid()}-{threading.get_ident()}.tmp")
            partial.write_text(
                json.dumps({"prompt": prompt, "answer": answer}), encoding="utf-8"
            )
            partial.replace(path)
        return answer
