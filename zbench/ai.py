import asyncio
import json
import os
from collections.abc import Callable
from typing import Literal

import anthropic
import openai
import redis.exceptions
import tiktoken
from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, NotGiven
from anthropic.types import MessageParam
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openlimit.rate_limiters import RateLimiter
from openlimit.redis_rate_limiters import RateLimiterWithRedis 
from pydantic import BaseModel, ValidationError, computed_field

REDIS_URL = None

class AIModel(BaseModel):
    company: Literal["openai", "google", "anthropic"]
    model: str

    @computed_field
    @property
    def ratelimit_tpm(self) -> float:
        match self.company:
            case "openai":
                # Tier 5
                match self.model:
                    case _ if self.model.startswith("gpt-4o-mini"):
                        return 150_000_000
                    case _ if self.model.startswith("gpt-4o"):
                        return 30_000_000
                    case "gpt-4-turbo":
                        return 2_000_000
                    case _:
                        return 1_000_000
            case "google":
                # Tier 2
                return 5_000_000
            case "anthropic":
                # Tier 4
                return 80_000

    @computed_field
    @property
    def ratelimit_rpm(self) -> float:
        match self.company:
            case "openai":
                # Tier 5
                match self.model:
                    case _ if self.model.startswith("gpt-4o-mini"):
                        return 30_000
                    case _:
                        return 10_000
            case "google":
                # Tier 2
                return 1_000
            case "anthropic":
                # Tier 4
                return 4_000

class AIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

RATE_LIMIT_RATIO = 0.95

class AIConnection:
    openai_client: AsyncOpenAI
    anthropic_client: AsyncAnthropic
    sync_anthropic_client: Anthropic
    google_client: AsyncOpenAI

    # Mapping from (company, model) to RateLimiter
    rate_limiters: dict[str, RateLimiter | RateLimiterWithRedis]
    backoff_semaphores: dict[str, asyncio.Semaphore]
    redis_semaphores: dict[str, asyncio.Semaphore]

    def __init__(self) -> None:
        self.openai_client = AsyncOpenAI()
        self.google_client = AsyncOpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/",
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.anthropic_client = AsyncAnthropic()
        self.sync_anthropic_client = Anthropic()

        self.rate_limiters = {}
        self.backoff_semaphores = {}
        self.redis_semaphores = {}

    async def ai_wait_ratelimit(
        self,
        model: AIModel,
        num_tokens: int,
        backoff: float | None = None,
    ) -> None:
        key = f"{model.__class__}:{model.company}:{model.model}"
        if key not in self.rate_limiters:
            if REDIS_URL is None:
                self.rate_limiters[key] = RateLimiter(
                    request_limit=model.ratelimit_rpm * RATE_LIMIT_RATIO,
                    token_limit=model.ratelimit_tpm * RATE_LIMIT_RATIO,
                    token_counter=None,
                    bucket_size_in_seconds=15,
                )
            else:
                self.rate_limiters[key] = RateLimiterWithRedis(
                    request_limit=model.ratelimit_rpm * RATE_LIMIT_RATIO,
                    token_limit=model.ratelimit_tpm * RATE_LIMIT_RATIO,
                    token_counter=None,
                    bucket_size_in_seconds=15,
                    redis_url=REDIS_URL,
                    bucket_key=key,
                )
            self.backoff_semaphores[key] = asyncio.Semaphore(1)
            # Prevent too many redis connections.
            self.redis_semaphores[key] = asyncio.Semaphore(100)
        if backoff is not None:
            async with self.backoff_semaphores[key]:
                await asyncio.sleep(backoff)

        for _redis_retry in range(10):
            try:
                async with self.redis_semaphores[key]:
                    await self.rate_limiters[key].wait_for_capacity(num_tokens)  # pyright: ignore[reportUnknownMemberType]
                break
            except redis.exceptions.LockError:
                logger.warning("redis.exceptions.LockError")
                await asyncio.sleep(0.05)
                continue
            except (ConnectionResetError, redis.exceptions.ConnectionError):
                logger.exception("Redis Exception")
                await asyncio.sleep(0.05)
                continue

# NOTE: API Clients cannot be called from multiple event loops,
# So every asyncio event loop needs its own API connection
ai_connections: dict[asyncio.AbstractEventLoop, AIConnection] = {}

def get_ai_connection() -> AIConnection:
    event_loop = asyncio.get_event_loop()
    if event_loop not in ai_connections:
        ai_connections[event_loop] = AIConnection()
    return ai_connections[event_loop]

class AIError(Exception):
    """A class for AI Task Errors"""

class AIValueError(AIError, ValueError):
    """A class for AI Value Errors"""

class AITimeoutError(AIError, TimeoutError):
    """A class for AI Task Timeout Errors"""

class AIRuntimeError(AIError, RuntimeError):
    """A class for AI Task Timeout Errors"""

def ai_num_tokens(model: AIModel, s: str) -> int:
    if model.company == "anthropic":
        # Doesn't actually connect to the network
        return (
            get_ai_connection()
            .sync_anthropic_client.messages.count_tokens(
                model=model.model,
                system="",
                messages=[
                    {
                        "role": "user",
                        "content": s,
                    }
                ],
            )
            .input_tokens
        )
    elif model.company == "openai":
        if model.model.startswith("gpt-4.1"):
            model_str = "gpt-4"
        else:
            model_str = model.model
        encoding = tiktoken.encoding_for_model(model_str)
        num_tokens = len(encoding.encode(s))
        return num_tokens
    return int(len(s) / 4)

async def ai_call[T: str | BaseModel](
    model: AIModel,
    messages: list[AIMessage],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    # When using anthropic, the first message must be from the user.
    # If the first message is not a User, this message will be prepended to the messages.
    anthropic_initial_message: str | None = "<START>",
    # If two messages of the same role are given to anthropic, they must be concatenated.
    # This is the delimiter between concatenated.
    anthropic_combine_delimiter: str = "\n",
    # Throw an AITimeoutError after this many retries fail
    num_ratelimit_retries: int = 10,
    # Backoff function (Receives index of attempt)
    backoff_algo: Callable[[int], float] = lambda i: min(2**i, 5),
    # The output type for the ai_call. Valid options are a pydantic BaseModel or a str. Using a BaseModel will use the Structured Output API.
    response_format: type[T] = str,
) -> T:

    num_tokens_input: int = sum(
        [ai_num_tokens(model, message.content) for message in messages]
    )

    return_value: T | None = None
    match model.company:
        case "openai" | "google":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )

                    def ai_message_to_openai_message_param(
                        message: AIMessage,
                    ) -> ChatCompletionMessageParam:
                        if message.role == "system":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "user":  # noqa: SIM114
                            return {"role": message.role, "content": message.content}
                        elif message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        raise NotImplementedError("Unreachable Code")

                    client = {
                        "openai": get_ai_connection().openai_client,
                        "google": get_ai_connection().google_client,
                    }[model.company]
                    if client is None:
                        raise AIValueError(f"{model.company!r} client not configured")
                    if response_format is str:
                        response = await client.chat.completions.create(
                            model=model.model,
                            messages=[
                                ai_message_to_openai_message_param(message)
                                for message in messages
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        response_content = response.choices[0].message.content
                        assert response_content is not None
                        assert isinstance(response_content, response_format)
                        return_value = response_content
                    else:
                        response = await client.beta.chat.completions.parse(
                            model=model.model,
                            messages=[
                                ai_message_to_openai_message_param(message)
                                for message in messages
                            ],
                            temperature=0,
                            response_format=response_format,
                        )
                        response_parsed = response.choices[0].message.parsed
                        assert response_parsed is not None
                        assert isinstance(response_parsed, response_format)
                        return_value = response_parsed
                    break
                except openai.RateLimitError:
                    logger.warning("OpenAI RateLimitError")
            if return_value is None:
                raise AITimeoutError("Cannot overcome OpenAI RateLimitError")

        case "anthropic":
            for i in range(num_ratelimit_retries):
                try:
                    await get_ai_connection().ai_wait_ratelimit(
                        model, num_tokens_input, backoff_algo(i - 1) if i > 0 else None
                    )

                    def ai_message_to_anthropic_message_param(
                        message: AIMessage,
                    ) -> MessageParam:
                        if message.role == "user" or message.role == "assistant":
                            return {"role": message.role, "content": message.content}
                        elif message.role == "system":
                            raise AIValueError(
                                "system not allowed in anthropic message param"
                            )
                        raise NotImplementedError("Unreachable Code")

                    # Extract system message if it exists
                    system: str | NotGiven = NOT_GIVEN
                    if len(messages) > 0 and messages[0].role == "system":
                        system = messages[0].content
                        messages = messages[1:]
                    if issubclass(response_format, BaseModel):
                        if not isinstance(system, str):
                            system = ""
                        system = (
                            f"Please respond with a JSON object adhering to the provided JSON schema. Don't provide any extra fields, and don't respond with anything other than the JSON object.\n\n{json.dumps(response_format.model_json_schema())}"
                            + system
                        )
                        messages.append(
                            AIMessage(
                                role="assistant",
                                content="```json",
                            )
                        )

                    # Insert initial message if necessary
                    if (
                        anthropic_initial_message is not None
                        and len(messages) > 0
                        and messages[0].role != "user"
                    ):
                        messages = [
                            AIMessage(role="user", content=anthropic_initial_message)
                        ] + messages
                    # Combined messages (By combining consecutive messages of the same role)
                    combined_messages: list[AIMessage] = []
                    for message in messages:
                        if (
                            len(combined_messages) == 0
                            or combined_messages[-1].role != message.role
                        ):
                            combined_messages.append(message)
                        else:
                            # Copy before edit
                            combined_messages[-1] = combined_messages[-1].model_copy(
                                deep=True
                            )
                            # Merge consecutive messages with the same role
                            combined_messages[-1].content += (
                                anthropic_combine_delimiter + message.content
                            )
                    # Get the response
                    response_message = (
                        await get_ai_connection().anthropic_client.messages.create(
                            model=model.model,
                            system=system,
                            messages=[
                                ai_message_to_anthropic_message_param(message)
                                for message in combined_messages
                            ],
                            temperature=0.0,
                            max_tokens=max_tokens,
                        )
                    )
                    assert isinstance(
                        response_message.content[0], anthropic.types.TextBlock
                    )
                    response_content = response_message.content[0].text
                    assert isinstance(response_content, str)
                    if response_format is str:
                        assert isinstance(response_content, response_format)
                        return_value = response_content  # pyright: ignore[reportAssignmentType]
                    else:
                        assert issubclass(response_format, BaseModel)
                        response_content = response_content.strip()
                        if response_content.startswith("```json"):
                            response_content = response_content[len("```json") :]
                        while True:
                            if response_content.endswith("```"):
                                response_content = response_content[: -len("```")]
                                response_content = response_content.strip()
                            else:
                                break
                        try:
                            return_value = response_format.model_validate_json(
                                response_content
                            )
                        except ValidationError as e:
                            print(f"Invalid: {response_content}")
                            raise AIRuntimeError(
                                "Failed to Validate Response Content."
                            ) from e
                    break
                except (anthropic.RateLimitError, anthropic.BadRequestError) as e:
                    logger.warning(f"Anthropic Error: {repr(e)}")
            if return_value is None:
                raise AITimeoutError("Cannot overcome Anthropic RateLimitError")

    return return_value