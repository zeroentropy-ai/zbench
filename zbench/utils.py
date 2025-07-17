import asyncio
import fcntl
import hashlib
import json
import math
import os
import sys
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any, Dict, List

import aiohttp
import numpy as np
from datasets import (  # pyright: ignore[reportMissingTypeStubs]
    Dataset,
    DatasetDict,
    load_from_disk,
)
from dotenv import load_dotenv
from numpy.typing import NDArray
from tqdm import tqdm
from pydantic import BaseModel

load_dotenv(override=True)


ROOT = f"{Path(__file__).resolve().parent.parent}"

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false

# NumPy Float N-Dimensional Array
type npf = NDArray[np.float64]


def format_globals(globals_dict: dict[str, Any]) -> str:
    serializable_globals: dict[str, Any] = {
        "COMMAND": sys.argv[0],
    }
    if len(sys.argv) > 1:
        serializable_globals["ARGS"] = sys.argv[1:]
    for k, v in globals_dict.items():
        if k.upper() != k or k in ["ROOT"]:
            continue
        try:
            json.dumps(v)
            serializable_globals[k] = v
        except (TypeError, OverflowError):
            pass
    return json.dumps(serializable_globals, indent=4)


def hash_str(input: str) -> str:
    return hashlib.sha256(input.encode()).hexdigest()[:32]


def hash_str_to_int(input: str) -> int:
    return int(hashlib.sha256(input.encode()).hexdigest(), base=16)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


client_connections: dict[asyncio.AbstractEventLoop, aiohttp.ClientSession] = {}


def get_client() -> aiohttp.ClientSession:
    event_loop = asyncio.get_event_loop()
    if event_loop not in client_connections:
        client_connections[event_loop] = aiohttp.ClientSession()
    return client_connections[event_loop]


async def wrap_sem[T](f: Awaitable[T], sem: asyncio.Semaphore) -> T:
    async with sem:
        return await f


def unzip[A, B](pairs: list[tuple[A, B]]) -> tuple[list[A], list[B]]:
    return tuple(map(list, zip(*pairs, strict=True))) if len(pairs) > 0 else ([], [])  # pyright: ignore[reportReturnType]


def unzip3[A, B, C](pairs: list[tuple[A, B, C]]) -> tuple[list[A], list[B], list[C]]:
    return (
        tuple(map(list, zip(*pairs, strict=True))) if len(pairs) > 0 else ([], [], [])
    )  # pyright: ignore[reportReturnType]


def unzip4[A, B, C, D](
    pairs: list[tuple[A, B, C, D]],
) -> tuple[list[A], list[B], list[C], list[D]]:
    return (
        tuple(map(list, zip(*pairs, strict=True)))
        if len(pairs) > 0
        else ([], [], [], [])
    )  # pyright: ignore[reportReturnType]


def avg(values: list[float]) -> float:
    if len(values) == 0:
        return float("nan")
    else:
        return sum(values) / len(values)


def argsort(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i])


def sorted_by_keys[T](
    values: list[T], keys: list[float], *, reverse: bool = False
) -> list[T]:
    """Sort values by corresponding scores in ascending order."""
    return [
        value
        for value, _ in sorted(
            zip(values, keys, strict=True), key=lambda x: x[1], reverse=reverse
        )
    ]


def clamp(value: float, minimum: float, maximum: float) -> float:
    assert minimum <= maximum
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def flatten[T](array_2d: list[list[T]]) -> list[T]:
    return [item for array in array_2d for item in array]


def unwrap[T](value: T | None) -> T:
    assert value is not None
    return value


def read_num_lines_pbar(file_path: str, *, display_name: str | None = None) -> int:
    if display_name is None:
        display_name = file_path
    num_lines = 0
    with open(file_path) as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {display_name}",
        ) as pbar:
            for line in f:
                num_lines += 1
                pbar.update(len(line))
                pbar.set_postfix(
                    {
                        "Lines": str(num_lines),
                    }
                )
    return num_lines


@contextmanager
def lock_file(f: IO[Any]) -> Generator[None, Any, None]:
    try:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield None
    finally:
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def load_custom_dataset(path: str) -> Dataset | DatasetDict:
    dataset_dict_path = f"{path}/dataset_dict.json"
    if os.path.exists(dataset_dict_path):
        with open(dataset_dict_path) as f:
            dataset_dict_splits = [
                str(elem) for elem in list(json.loads(f.read())["splits"])
            ]
            return DatasetDict(
                {
                    split_name: load_custom_dataset(f"{path}/{split_name}")
                    for split_name in dataset_dict_splits
                }
            )
    else:
        return load_from_disk(path)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
    
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def save_pydantic_json(path: str, data: BaseModel) -> None:
    with open(path, "w") as f:
        f.write(data.model_dump_json(indent=4))

def save_pydantic_jsonl(path: str, data: List[BaseModel]) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(item.model_dump_json() + "\n")

def append_to_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    with open(path, "a") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def elos_loss(w: npf, elos: npf) -> float:
    N = len(elos)
    elos_col = elos.reshape(-1, 1)
    elos_row = elos.reshape(1, -1)

    # Stable computation of log(exp(elos_i) + exp(elos_j))
    max_elos = np.maximum(elos_col, elos_row)
    log_pairwise_sums = max_elos + np.log(
        np.exp(elos_col - max_elos) + np.exp(elos_row - max_elos)
    )

    # Calculate elos_i - log(exp(elos_i) + exp(elos_j))
    log_diff = np.broadcast_to(elos_col, (N, N)) - log_pairwise_sums

    # We want to maximize the loglikelihood of the observed w with respect to elos
    loglikelihood = float(np.sum(w * log_diff))

    # Return the loss that we're trying to minimize
    return -loglikelihood


def calculate_elos(
    w: npf,
    *,
    # How close we must be to the log-likelihood loss
    epsilon: float = 1e-4,
    # If you have ELOs calculated from a similar W, then it will converge faster by initializing to the same ELOs
    initial_elos: npf | None = None,
    # Max iters before giving up
    max_iters: int = 1000,
) -> tuple[npf, list[float]]:
    # https://hackmd.io/@-Gjw1zWMSH6lMPRlziQFEw/B15B4Rsleg

    N = len(w)
    elos = initial_elos.copy() if initial_elos is not None else np.zeros(N)

    losses: list[float] = []
    for _iter in range(max_iters):
        # Create all pairwise differences elo_j - elo_i in a matrix
        # outer(ones, elos) - outer(elos, ones)
        D: npf = elos.reshape(1, N) - elos.reshape(N, 1)  # Shape: (N, N)
        # Calculate sigmoid matrix
        S: npf = 1.0 / (1.0 + np.exp(-D))  # S[i,j] = sigmoid(elo_j - elo_i)

        # Calculate the update terms
        numerator: npf = np.sum(w * S, axis=1)  # Shape: (N,)
        denominator: npf = np.sum(w.T * S.T, axis=1)  # Shape: (N,)
        # Apply update rule, using decreasing learning rate.
        learning_rate = float((1.0 + _iter) ** (-0.125))
        elos += (np.log(numerator) - np.log(denominator)) * learning_rate
        elos -= np.mean(elos)

        # Calculate loss for this iteration
        loss = elos_loss(w, elos)
        losses.append(loss)
        if len(losses) > 2 and abs(losses[-2] - losses[-1]) < epsilon:
            break
    if abs(losses[-2] - losses[-1]) > epsilon:
        print(f"ERROR! Not within epsilon after {len(losses)} iterations!")

    return elos, losses

def dcg(scores: list[float]) -> float:
    """Calculate DCG."""
    return np.sum(scores / np.log2(np.arange(len(scores)) + 2))

def ndcg(ground_truth: list[float], scores: list[float]) -> float:
    """Calculate NDCG."""
    assert len(ground_truth) == len(scores)
    indices = np.argsort(scores)[::-1]
    predicted_scores = np.array(ground_truth)[indices]
    ideal_scores = np.array(sorted(ground_truth, reverse=True))
    return dcg(predicted_scores) / dcg(ideal_scores)