import asyncio
import time

from council.client import SlidingWindowRateLimiter


async def test_allows_burst_up_to_limit():
    rl = SlidingWindowRateLimiter(limit=5, window=60.0)
    start = time.monotonic()
    for _ in range(5):
        await rl.acquire()
    assert time.monotonic() - start < 0.5


async def test_blocks_past_limit_until_window_slides():
    rl = SlidingWindowRateLimiter(limit=2, window=0.4)
    await rl.acquire()
    await rl.acquire()
    start = time.monotonic()
    await rl.acquire()  # must wait for the first timestamp to age out
    assert time.monotonic() - start >= 0.3


async def test_concurrent_acquires_never_exceed_limit():
    rl = SlidingWindowRateLimiter(limit=3, window=0.5)
    in_window: list[float] = []

    async def worker():
        await rl.acquire()
        in_window.append(time.monotonic())

    await asyncio.gather(*(worker() for _ in range(9)))
    in_window.sort()
    # in any 0.5s span, at most 3 acquisitions
    for i in range(len(in_window) - 3):
        assert in_window[i + 3] - in_window[i] >= 0.45
