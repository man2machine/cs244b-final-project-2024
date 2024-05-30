# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:04:21 2024

@author: Shahir
"""

import asyncio
from typing import Generic, TypeVar, Self, final

AsyncQueueItemT = TypeVar('AsyncQueueItemT')


@final
class AsyncQueueFullException(Exception):
    pass


@final
class AsyncQueueEmptyException(Exception):
    pass


@final
class AsyncQueue(Generic[AsyncQueueItemT]):
    max_size: int

    _queue: asyncio.Queue[AsyncQueueItemT]

    def __init__(
        self: Self,
        max_size: int = 0
    ) -> None:

        self.max_size = max_size
        self._queue = asyncio.Queue(self.max_size)

    def size(
        self: Self
    ) -> int:

        return self._queue.qsize()

    def remaining(
        self: Self
    ) -> int:

        return -1 if (self.max_size == 0) else (self.max_size - self.size())

    def is_empty(
        self: Self
    ) -> bool:

        return self._queue.empty()

    def is_full(
        self: Self
    ) -> bool:

        return self._queue.full()

    def put_nowait(
        self: Self,
        item: AsyncQueueItemT
    ) -> None:

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            raise AsyncQueueEmptyException()

    def get_nowait(
        self: Self
    ) -> AsyncQueueItemT:

        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            raise AsyncQueueEmptyException()

    async def put_async(
        self: Self,
        item: AsyncQueueItemT,
        timeout: float | None = None
    ) -> None:

        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")

        try:
            await asyncio.wait_for(self._queue.put(item), timeout)
        except asyncio.TimeoutError:
            raise AsyncQueueFullException()

    async def get_async(
        self: Self,
        timeout: float | None = None
    ) -> AsyncQueueItemT:

        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")

        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            raise AsyncQueueEmptyException()

    def put_nowait_batch(
        self: Self,
        items: list[AsyncQueueItemT]
    ) -> None:

        if self.max_size > 0 and len(items) + self.size() > self.max_size:
            raise AsyncQueueFullException(
                f"Cannot add {len(items)} items to queue of size {self.size()} and maxsize {self.max_size}."
            )
        for item in items:
            self._queue.put_nowait(item)

    def get_nowait_batch(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False
    ) -> list[AsyncQueueItemT]:

        if ((max_num_items is not None) + (num_items is not None) + all_items) != 1:
            raise ValueError("Invalid input")
        if (num_items is not None) and (num_items > self.size() or num_items < 0):
            raise AsyncQueueEmptyException(f"Cannot get {num_items} items from queue of size {self.size()}.")
        if max_num_items is not None:
            num_items = min(self.size(), max_num_items)
        if all_items:
            num_items = self.size()
        assert num_items is not None

        out = []
        try:
            for _ in range(num_items):
                out.append(self._queue.get_nowait())
        except AsyncQueueEmptyException:
            pass

        return out

    async def put_batch_async(
        self: Self,
        items: list[AsyncQueueItemT],
        timeout: float | None = None
    ) -> None:

        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")
        if self.max_size > 0 and len(items) + self.size() > self.max_size:
            raise AsyncQueueFullException(
                f"Cannot add {len(items)} items to queue of size "
                f"{self.size()} and maxsize {self.max_size}."
            )

        for item in items:
            await self.put_async(item, timeout)

    async def get_batch_async(
        self: Self,
        num_items: int | None = None,
        max_num_items: int | None = None,
        all_items: bool = False,
        min_num_items: int | None = None,
        timeout: float | None = None
    ) -> list[AsyncQueueItemT]:

        if ((max_num_items is not None) + (num_items is not None) + all_items) != 1:
            raise ValueError("Invalid input")
        if (num_items is not None) and (num_items > self.size() or num_items < 0):
            raise AsyncQueueEmptyException(f"Cannot get {num_items} items from queue of size {self.size()}.")
        if (timeout is not None) and (timeout < 0):
            raise ValueError("Invalid input")
        if max_num_items is not None:
            num_items = min(self.size(), max_num_items)
            if min_num_items is not None:
                num_items = max(min_num_items, num_items)
        if all_items:
            num_items = self.size()
            if min_num_items is not None:
                num_items = max(min_num_items, num_items)
        assert num_items is not None

        out = []
        try:
            for _ in range(num_items):
                out.append(await asyncio.wait_for(self._queue.get(), timeout))
        except asyncio.TimeoutError:
            pass

        return out
