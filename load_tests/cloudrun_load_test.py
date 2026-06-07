from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import aiohttp

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


async def send_request(session: aiohttp.ClientSession, url: str, image_path: Path) -> dict:
    start = time.perf_counter()

    data = aiohttp.FormData()
    data.add_field(
        "file",
        image_path.read_bytes(),
        filename=image_path.name,
        content_type="image/jpeg",
    )

    try:
        async with session.post(url, data=data, timeout=120) as response:
            text = await response.text()
            latency_ms = (time.perf_counter() - start) * 1000.0

            return {
                "status": response.status,
                "latency_ms": latency_ms,
                "ok": response.status == 200,
                "body_sample": text[:200],
            }

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "status": None,
            "latency_ms": latency_ms,
            "ok": False,
            "error": str(exc),
        }


async def run_load_test(
    url: str, images: list[Path], requests: int, concurrency: int
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:

        async def bounded_request(index: int) -> None:
            async with semaphore:
                image_path = images[index % len(images)]
                result = await send_request(session, url, image_path)
                results.append(result)

                if len(results) % 100 == 0:
                    print(f"Completed {len(results)}/{requests}")

        tasks = [bounded_request(i) for i in range(requests)]
        await asyncio.gather(*tasks)

    return results


def summarize(results: list[dict], total_seconds: float) -> dict:
    latencies = [row["latency_ms"] for row in results]
    success_count = sum(1 for row in results if row["ok"])
    error_count = len(results) - success_count

    return {
        "requests": len(results),
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / len(results),
        "total_seconds": total_seconds,
        "requests_per_second": len(results) / total_seconds,
        "estimated_requests_per_day": (len(results) / total_seconds) * 86400,
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1],
        "latency_max_ms": max(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    images = list_images(Path(args.image_dir))

    if not images:
        raise FileNotFoundError("No images found for load test.")

    url = args.service_url.rstrip("/") + "/predict"

    start = time.perf_counter()
    results = asyncio.run(
        run_load_test(
            url=url,
            images=images,
            requests=args.requests,
            concurrency=args.concurrency,
        )
    )
    total_seconds = time.perf_counter() - start

    summary = summarize(results, total_seconds)

    print("\nLOAD TEST SUMMARY")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
