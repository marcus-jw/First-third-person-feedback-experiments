import aiohttp
import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
import anthropic

def estimate_tokens(text: str) -> int:
    # This is a very rough estimate.
    return len(text) // 4  # Assuming average of 4 characters per token

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1

async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    api_key: str,
    max_requests_per_second: float,
    max_tokens_per_second: float,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()
    status_tracker = StatusTracker()
    next_request = None

    # initialize available capacity counts
    available_request_capacity = max_requests_per_second
    available_token_capacity = max_tokens_per_second
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            request_json = json.loads(next(requests))
                            estimated_tokens = estimate_tokens(json.dumps(request_json))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_estimate=estimated_tokens,
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_second * seconds_since_update,
                    max_requests_per_second,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_second * seconds_since_update,
                    max_tokens_per_second,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    if (available_request_capacity >= 1 and
                        available_token_capacity >= next_request.token_estimate):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request.token_estimate
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                client=client,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_estimate: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        client: anthropic.AsyncAnthropic,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the Anthropic API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            response = await client.messages.create(**self.request_json)
        except anthropic.RateLimitError as e:
            logging.warning(f"Request {self.task_id} hit rate limit: {e}")
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
            error = e
        except anthropic.APIError as e:
            logging.warning(f"Request {self.task_id} failed with API error: {e}")
            status_tracker.num_api_errors += 1
            error = e
        except Exception as e:
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(str(error))
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, self.result, self.metadata]
                    if self.metadata
                    else [self.request_json, self.result]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response.model_dump(), self.metadata]
                if self.metadata
                else [self.request_json, response.model_dump()]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")

def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

if __name__ == "__main__":
    train_test = "train"
    split = "harmless-base"
    # "harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"
    for split in ["helpful-base", "helpful-online", "helpful-rejection-sampled"]:
        for perspective in ["1_1", "3_1", "1_3", "3_3"]:
            requests_filepath = os.path.dirname(__file__) + f"/../../data/hh_labels/haiku_{split}_{train_test}_requests_{perspective}.jsonl"
            save_filepath = os.path.dirname(__file__) + f"/../../data/hh_labels/haiku_{split}_{train_test}_results_{perspective}.jsonl"
            api_key = os.getenv("ANTHROPIC_API_KEY")
            max_requests_per_second = int(3500/60)  # Changed from per minute to per second
            max_tokens_per_second = int(350_000/60)  # Changed from per minute to per second
            max_attempts = 5
            logging_level = logging.INFO

            # run script
            asyncio.run(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    api_key=api_key,
                    max_requests_per_second=float(max_requests_per_second),
                    max_tokens_per_second=float(max_tokens_per_second),
                    max_attempts=int(max_attempts),
                    logging_level=int(logging_level),
                )
            )