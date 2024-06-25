import asyncio
import json
from itertools import cycle
from openai import AsyncOpenAI
import time
from tqdm import tqdm

# Configure these values based on your OpenAI plan and requirements
MAX_REQUESTS_PER_MINUTE = 200
MAX_CONCURRENT_REQUESTS = 15

async def main():
    client = AsyncOpenAI()
    model = "gpt-4o"
    task_list = ['danger_refusal']
    pos_neg_key_label_dict = {
        "danger_refusal": ["refusal_answer", "dangerous_answer"],
        "impossible_task_refusal": ["refusal_answer", "incorrect_answer"],
        "personalisation": ["appropriate_answer", "inappropriate_answer"],
        "sycophancy": ["sycophantic_answer", "normal_answer"],
        "verbosity": ["normal_answer", "short_answer"],
    }

    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Create a rate limiter
    class RateLimiter:
        def __init__(self, max_calls, period):
            self.max_calls = max_calls
            self.period = period
            self.calls = []

        async def wait(self):
            now = time.time()
            # Remove old calls
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] - (now - self.period)
                await asyncio.sleep(sleep_time)
            self.calls.append(time.time())

    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE, 60)

    # Create a progress bar
    total_requests = len(task_list) * 80  # 80 requests per task
    progress_bar = tqdm(total=total_requests, desc="Processing requests")

    async def process_task(task):
        save_path = f"data/datasets/{task}.jsonl"
        with open(f"settings/prompts/topics/{task}.txt", 'r') as file:
            topics_line = file.read()
            topics = [topic.strip() for topic in topics_line.split(',')]
            topics_cycle = cycle(topics)

        with open(f"settings/prompts/generate_pov_themes_data/{task}.txt", "r") as file:
            file_content = file.read()

        tasks = []
        for i in range(80):
            topic = next(topics_cycle)
            vars_dict = {"topic": topic}
            prompt = file_content.format(**vars_dict)
            tasks.append(process_prompt(client, model, task, prompt, pos_neg_key_label_dict, semaphore, rate_limiter, progress_bar))

        results = await asyncio.gather(*tasks)

        with open(save_path, "w", encoding="utf-8") as save_file:
            for result in results:
                save_file.write(result)

    async def process_prompt(client, model, task, prompt, pos_neg_key_label_dict, semaphore, rate_limiter, progress_bar):
        async with semaphore:
            await rate_limiter.wait()
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a JSONL dataset generator. Do not answer with anything other than JSONL."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                )
                content = response.choices[0].message.content
                result = ""
                for split in content.split("\n"):
                    try:
                        tmp_dict = eval(split)
                        tmp_dict["positive_label"], tmp_dict["negative_label"] = pos_neg_key_label_dict[task]
                        result += json.dumps(tmp_dict) + "\n"
                    except:
                        pass
                progress_bar.update(1)
                return result
            except Exception as e:
                print(f"Error processing prompt: {e}")
                progress_bar.update(1)
                return ""

    await asyncio.gather(*[process_task(task) for task in task_list])
    progress_bar.close()

if __name__ == "__main__":
    asyncio.run(main())