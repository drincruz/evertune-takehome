import asyncio
import time
import math
import statistics
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from llm import Gemini

# Sample realistic prompts of varying lengths
SAMPLE_PROMPTS = [
    {
        "system": "You are a customer support triage assistant. Categorize the sentiment and intent of the customer request.",
        "question": "My order #84920 has not arrived yet. It was supposed to be delivered yesterday. Can you please check where it is and offer a refund if it is lost?"
    },
    {
        "system": "You are a technical document reviewer. Provide a 2-sentence summary of the main points.",
        "question": "Microservice architecture decomposes monolithic applications into independent services that communicate over lightweight protocols like HTTP or gRPC. While this improves scalability, deployment independence, and fault isolation, it introduces complexity in distributed data management, service discovery, and end-to-end tracing."
    },
    {
        "system": "You are a code refactoring advisor. List 3 key code quality improvements for Python code.",
        "question": "How can I improve the readability, maintainability, and performance of asynchronous Python code using asyncio and type hints?"
    },
    {
        "system": "You are an executive assistant. Draft a concise email reply.",
        "question": "Can we move our strategy meeting from Tuesday 2 PM to Thursday 10 AM? Let me know if that time works for your calendar."
    }
]

@dataclass
class RequestResult:
    success: bool
    latency: float
    input_tokens: int
    output_tokens: int
    error: str = ""
    status_code: str = ""

@dataclass
class ScenarioResult:
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_sec: float
    rps: float
    input_tokens_per_sec: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    latency_min: float
    latency_max: float
    errors_breakdown: Dict[str, int]

def calculate_percentile(data: List[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1

async def worker(gemini: Gemini, queue: asyncio.Queue, results: List[RequestResult]):
    while not queue.empty():
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        start_time = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                gemini.ask_generic_question(
                    system_prompt=item["system"],
                    question=item["question"],
                    temperature=0.7
                ),
                timeout=120.0
            )
            elapsed = time.perf_counter() - start_time
            results.append(RequestResult(
                success=True,
                latency=elapsed,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens
            ))
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                err_category = "HTTP 429 Resource Exhausted (Quota/Rate Limit)"
            elif "TimeoutError" in type(e).__name__ or "timeout" in err_str.lower():
                err_category = "Timeout Error (>120s)"
            else:
                err_category = f"Error: {type(e).__name__}"

            results.append(RequestResult(
                success=False,
                latency=elapsed,
                input_tokens=0,
                output_tokens=0,
                error=err_category
            ))
        finally:
            queue.task_done()

async def run_scenario(gemini: Gemini, concurrency: int, total_requests: int) -> ScenarioResult:
    queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
    
    for i in range(total_requests):
        prompt = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        queue.put_nowait(prompt)

    results: List[RequestResult] = []
    
    print(f"--- Running Load Test Scenario: Concurrency={concurrency}, Total Requests={total_requests} ---", flush=True)
    start_wall = time.perf_counter()
    
    workers = [
        asyncio.create_task(worker(gemini, queue, results))
        for _ in range(concurrency)
    ]
    
    await queue.join()
    for w in workers:
        w.cancel()
        
    duration = time.perf_counter() - start_wall
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    latencies = [r.latency for r in successful]
    total_input = sum(r.input_tokens for r in successful)
    total_output = sum(r.output_tokens for r in successful)
    
    errors_breakdown: Dict[str, int] = {}
    for r in failed:
        err_key = r.error
        errors_breakdown[err_key] = errors_breakdown.get(err_key, 0) + 1
        
    rps = len(successful) / duration if duration > 0 else 0
    in_tps = total_input / duration if duration > 0 else 0
    out_tps = total_output / duration if duration > 0 else 0
    tot_tps = (total_input + total_output) / duration if duration > 0 else 0
    
    res = ScenarioResult(
        concurrency=concurrency,
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_duration_sec=duration,
        rps=rps,
        input_tokens_per_sec=in_tps,
        output_tokens_per_sec=out_tps,
        total_tokens_per_sec=tot_tps,
        latency_p50=calculate_percentile(latencies, 50),
        latency_p90=calculate_percentile(latencies, 90),
        latency_p95=calculate_percentile(latencies, 95),
        latency_p99=calculate_percentile(latencies, 99),
        latency_mean=statistics.mean(latencies) if latencies else 0.0,
        latency_min=min(latencies) if latencies else 0.0,
        latency_max=max(latencies) if latencies else 0.0,
        errors_breakdown=errors_breakdown
    )
    
    print(f"Result: Concurrency={concurrency:3d} | Reqs={total_requests:3d} | Success={len(successful):3d} | Fail={len(failed):3d} | Dur={duration:6.2f}s | RPS={rps:6.2f} | p50={res.latency_p50:5.2f}s | p95={res.latency_p95:5.2f}s | p99={res.latency_p99:5.2f}s", flush=True)
    if failed:
        print(f"  Failures breakdown: {errors_breakdown}", flush=True)
    print(flush=True)
    return res

async def main():
    gemini = Gemini()
    concurrency_levels = [1, 5, 10, 20, 30, 50, 75, 100]
    all_results = []
    
    for c in concurrency_levels:
        reqs = max(20, c)
        res = await run_scenario(gemini=gemini, concurrency=c, total_requests=reqs)
        all_results.append(asdict(res))
        
        with open("load_test_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
            
        # 3 second cooldown between concurrency steps
        await asyncio.sleep(3)

    print("=== All Load Test Scenarios Complete ===", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
