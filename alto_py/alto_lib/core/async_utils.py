# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:43:03 2023

@author: Shahir
"""

import asyncio
from typing import Self, Any, final
from collections.abc import Coroutine


@final
class AsyncConcurrentProcessor:
    _background_tasks: set[asyncio.Task]
    _max_running_requests: int
    _request_sem: asyncio.Semaphore

    def __init__(
        self: Self,
        max_running_requests: int | None = None
    ) -> None:

        self._background_tasks = set()
        if max_running_requests is not None:
            assert max_running_requests > 0
            self._max_running_requests = max_running_requests
            self._request_sem = asyncio.Semaphore(self._max_running_requests)
        else:
            self._max_running_requests = 0

    def _task_done_callback(
        self: Self,
        task: asyncio.Task
    ) -> None:

        self._background_tasks.discard(task)
        if self._max_running_requests:
            self._request_sem.release()

    async def add_task(
        self: Self,
        coro: Coroutine[Any, Any, None]
    ) -> asyncio.Task:

        if self._max_running_requests:
            await self._request_sem.acquire()

        # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)  # must create strong reference
        task.add_done_callback(self._task_done_callback)
        
        return task

    def get_num_open_task_slots(
        self: Self
    ) -> int:

        if self._max_running_requests:
            return self._max_running_requests - len(self._background_tasks)
        else:
            return -1

    async def wait_for_open_task_slot(
        self: Self
    ) -> None:
        
        if self._max_running_requests:
            await self._request_sem.acquire()
            self._request_sem.release()
