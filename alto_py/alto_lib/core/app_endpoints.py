# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 06:41:57 2023

@author: Shahir
"""

import os
import abc
import time
import struct
import threading
import asyncio
import struct
import queue
from dataclasses import dataclass
from typing import Self, Any, final
from collections.abc import Callable, Coroutine, Hashable

import numpy as np

import ujson as json  # type: ignore[import-untyped]
from loguru import logger

from alto_lib.utils import system_time, get_rel_pkg_path
from alto_lib.core.communication import TCPClientSocket, TCPServerSocketAsync, IPv4Address
from alto_lib.core.manager import StageCommunicator
from alto_lib.core.async_utils import AsyncConcurrentProcessor

DEFAULT_LOG_DIR: str = get_rel_pkg_path("logs/")
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

DETAILED_LOGS: bool = False


@final
@dataclass(frozen=True, slots=True)
class AppRequest:
    user_request_id: int
    message: bytes


@final
@dataclass(frozen=True, slots=True)
class AppResponse:
    user_request_id: int
    message: bytes


@final
class AppClientNodeLogger:
    _log_fname: str
    _log_interval: float
    _last_log_write_time: float

    _request_send_times: dict[int, float]
    _response_recv_times: dict[int, float]
    _response_latencies: dict[int, float]
    _last_request_sent_time: float | None
    _last_response_recv_time: float | None
    _track_unfinished_requests: bool

    def __init__(
        self: Self,
        log_fname: str,
        log_interval: float = 2,
        track_unfinished_requests: bool = False,
    ) -> None:

        super().__init__()

        self._log_fname = log_fname
        self._log_interval = log_interval
        self._last_log_write_time = -1
        self._request_send_times = {}
        self._response_recv_times = {}
        self._response_latencies = {}
        self._last_request_sent_time = None
        self._last_response_recv_time = None
        self._track_unfinished_requests = track_unfinished_requests

    def get_log_fname(
        self: Self
    ) -> str:

        return self._log_fname

    def log_metadata(
        self: Self,
        name: str,
        request_send_address: IPv4Address,
        response_recv_address: IPv4Address
    ) -> None:

        data = {
            'name': name,
            'request_send_address': request_send_address,
            'response_recv_address': response_recv_address
        }

        buf = json.dumps(data)
        with open(self._log_fname, 'a+') as f:
            f.write(buf + "\n")

    def log_counts(
        self: Self,
        log_all: bool = False
    ) -> None:

        current_time = system_time()

        if self._track_unfinished_requests:
            unfinished_requests = set(self._request_send_times.keys()) - set(self._response_recv_times.keys())
            for user_request_id in unfinished_requests:
                self._response_latencies[user_request_id] = current_time - self._request_send_times[user_request_id]
        latencies = list(self._response_latencies.values())

        has_latencies = len(self._response_latencies) == 0
        avg_latency = None if has_latencies else float(np.mean(latencies))
        median_latency = None if has_latencies else float(np.median(latencies))
        p99_latency = None if has_latencies else float(np.percentile(latencies, 99))
        data: dict[str, Any] = {
            'time': current_time,
            'last_request_sent_time': self._last_request_sent_time,
            'last_response_recv_time': self._last_response_recv_time,
            'avg_latency': avg_latency,
            'median_latency': median_latency,
            'p99_latency': p99_latency,
            'num_requests_sent': len(self._request_send_times),
            'num_responses_recv': len(self._response_recv_times)
        }
        if log_all:
            data.update({
                'request_send_times': self._request_send_times,
                'response_recv_times': self._response_recv_times,
                'request_latencies': self._response_latencies
            })
        buf = json.dumps(data)
        with open(self._log_fname, 'a+') as f:
            f.write(buf + "\n")

    def update_request_send(
        self: Self,
        request: AppRequest
    ) -> None:

        request_time = system_time()
        self._request_send_times[request.user_request_id] = request_time
        self._last_request_sent_time = request_time

    def update_response_recv(
        self: Self,
        response: AppResponse
    ) -> None:

        response_time = system_time()
        self._response_recv_times[response.user_request_id] = response_time
        self._last_response_recv_time = response_time

        request_time = self._request_send_times.get(response.user_request_id, None)
        if request_time is not None:
            elapsed = response_time - request_time
            self._response_latencies[response.user_request_id] = elapsed
        else:
            raise ValueError("Recieved response with id that was never sent")

    def update_log(
        self: Self,
        force: bool = False,
        log_all: bool = False
    ) -> None:

        if force or (system_time() - self._last_log_write_time > self._log_interval):
            self._last_log_write_time = system_time()
            self.log_counts(log_all=log_all)


class AppClientNode:
    _CONTROL_START_MSG: bytes = b'start'
    _CONTROL_STOP_MSG: bytes = b'stop'

    name: str

    _request_send_address: IPv4Address
    _response_recv_address: IPv4Address
    _control_client: TCPClientSocket
    _request_send_client: TCPClientSocket
    _response_recv_client: TCPClientSocket

    _request_queue: queue.Queue[bytes]
    _sent_request_ids: set[int]
    _recv_response_ids: set[int]

    _log: AppClientNodeLogger
    _log_interval: float

    _send_length: float | None

    _ready: bool
    _app_started: bool
    _stop: bool

    _requests_finished_ev: threading.Event
    _requests_start_ev: threading.Event
    _responses_finished_ev: threading.Event
    _shutdown_ev: threading.Event

    _threads: list[threading.Thread]

    def __init__(
        self: Self,
        *,
        control_address: IPv4Address,
        request_send_address: IPv4Address,
        response_recv_address: IPv4Address,
        name: str,
        log_fname: str,
        log_interval: float = 1,
        send_length: float | None = None
    ) -> None:

        self.name = name

        self._request_send_address = request_send_address
        self._response_recv_address = response_recv_address
        self._control_client = TCPClientSocket(control_address)
        self._request_send_client = TCPClientSocket(request_send_address)
        self._response_recv_client = TCPClientSocket(response_recv_address)

        self._request_queue = queue.Queue()
        self._sent_request_ids = set()
        self._recv_response_ids = set()

        self._log = AppClientNodeLogger(log_fname, log_interval=log_interval)
        self._log_interval = log_interval

        self._send_length = send_length

        self._ready = False
        self._app_started = False
        self._stop = False

        self._requests_finished_ev = threading.Event()
        self._requests_start_ev = threading.Event()
        self._responses_finished_ev = threading.Event()
        self._shutdown_ev = threading.Event()

    def initialize(
        self: Self
    ) -> None:

        logger.debug("Initializing {} app client node".format(self.name))
        logger.info("Writing to log file {}".format(self._log.get_log_fname()))

        self._log.log_metadata(
            name=self.name,
            request_send_address=self._request_send_address,
            response_recv_address=self._response_recv_address
        )

        success = self._control_client.connect_wait_for_server()
        assert success
        success = self._request_send_client.connect_wait_for_server()
        assert success
        success = self._response_recv_client.connect_wait_for_server()
        assert success

        self._ready = True

    def set_requests_finished(
        self: Self
    ) -> None:

        self._requests_finished_ev.set()

    def stop(
        self: Self
    ) -> None:

        logger.debug("Stopping {} app client node".format(self.name))
        self._stop = True

    @abc.abstractmethod
    def _get_next_request(
        self: Self
    ) -> AppRequest:

        pass

    @abc.abstractmethod
    def _get_next_request_wait_time(
        self: Self
    ) -> float:

        pass

    @staticmethod
    def _construct_request_packet(
        request: AppRequest
    ) -> bytes:

        return struct.pack('I', request.user_request_id) + request.message

    @staticmethod
    def _parse_response_packet(
        buf: bytes
    ) -> AppResponse:

        return AppResponse(struct.unpack('I', buf[:4])[0], buf[4:])

    def _add_next_request_to_queue(
        self: Self
    ) -> None:

        request = self._get_next_request()
        buf = self._construct_request_packet(request)
        self._request_queue.put(buf)
        self._log.update_request_send(request)
        self._sent_request_ids.add(request.user_request_id)

    def _recv_next_response(
        self: Self
    ) -> None:

        buf = self._response_recv_client.receive()
        response = self._parse_response_packet(buf)
        self._recv_response_ids.add(response.user_request_id)

        self._log.update_response_recv(response)
        self._log.update_log()

    def _run_control_loop(
        self: Self
    ) -> None:

        logger.debug("Running {} app client node control loop".format(self.name))

        self._log.update_log(force=True)

        while True:
            if not self._app_started:
                if self._control_client.is_packet_waiting(timeout=10):
                    buf = self._control_client.receive()
                    if buf == self._CONTROL_START_MSG:
                        logger.debug("Starting client requests")
                        self._requests_start_ev.set()
                        self._app_started = True
            else:
                self._responses_finished_ev.wait()
                self._control_client.send(self._CONTROL_STOP_MSG)
                self._shutdown_ev.set()
                self.stop()

            if self._stop:
                break

        self._control_client.shutdown()
        self._log.update_log(force=True)

    def _run_request_generation_loop(
        self: Self
    ) -> None:

        logger.debug("Running {} app client node request generation loop".format(self.name))

        self._requests_start_ev.wait()

        start_time = system_time()

        while True:
            if not self._requests_finished_ev.is_set():
                loop_time = system_time()
                self._add_next_request_to_queue()
                time.sleep(max(self._get_next_request_wait_time() - (system_time() - loop_time), 0))

            else:
                self._responses_finished_ev.wait()

            self._log.update_log()

            if self._send_length is not None:
                if (system_time() - start_time) > self._send_length:
                    self.set_requests_finished()

            if self._stop:
                break

    def _run_request_send_loop(
        self: Self
    ) -> None:

        logger.debug("Running {} app client node request send loop".format(self.name))

        self._requests_start_ev.wait()

        while True:
            if not self._requests_finished_ev.is_set():
                buf = self._request_queue.get()
                self._request_send_client.send(buf)

            else:
                self._responses_finished_ev.wait()

            if self._stop:
                break

        self._shutdown_ev.wait()
        self._request_send_client.shutdown()

    def _run_response_recv_loop(
        self: Self
    ) -> None:

        logger.debug("Running {} app client node response recv loop".format(self.name))

        while True:
            if self._response_recv_client.is_packet_waiting(timeout=self._log_interval):
                self._recv_next_response()

                all_responses_recv = (
                    self._requests_finished_ev.is_set() and
                    (len(self._sent_request_ids) == len(self._recv_response_ids))
                )

                if all_responses_recv:
                    self._responses_finished_ev.set()

            self._log.update_log()

            if self._stop:
                break

        self._shutdown_ev.wait()
        self._response_recv_client.shutdown()

    def run(
        self: Self
    ) -> None:

        assert self._ready

        self._threads = [
            threading.Thread(target=self._run_control_loop),
            threading.Thread(target=self._run_request_generation_loop),
            threading.Thread(target=self._run_request_send_loop),
            threading.Thread(target=self._run_response_recv_loop)
        ]

        for t in self._threads:
            t.start()
        for t in self._threads:
            t.join()

        self._log.update_log(force=True, log_all=True)


@final
class AppControlServerNode:
    name: str

    _control_address: IPv4Address
    _control_server: TCPServerSocketAsync
    _client_addr: IPv4Address

    _stage_comm: StageCommunicator

    _poll_interval: float
    _instances_ready: bool
    _server_ready: bool
    _stop: bool

    def __init__(
        self: Self,
        *,
        control_address: IPv4Address,
        name: str,
        stage_comm: StageCommunicator,
        poll_interval: float = 10,
    ) -> None:

        self.name = name

        self._control_address = control_address
        self._control_server = TCPServerSocketAsync(control_address)

        self._stage_comm = stage_comm
        self._poll_interval = poll_interval

        self._instances_ready = False
        self._server_ready = False
        self._stop = False

    async def initialize(
        self: Self
    ) -> None:

        logger.debug("Initializing {} app control server node".format(self.name))

        await self._control_server.start()
        self._client_addr = self._control_server.get_client_addresses()[0]
        self._server_ready = True

    def stop(
        self: Self
    ) -> None:

        logger.debug("Stopping {} app control server node".format(self.name))

        self._stop = True

    async def run(
        self: Self
    ) -> None:

        assert self._server_ready
        logger.debug("Running {} app control server node".format(self.name))

        while True:
            if not self._instances_ready:
                logger.debug("Waiting until all instances are ready")
                await self._stage_comm.wait_all_instances_ready()
                logger.debug("All instances are ready!")

                await self._control_server.send(self._client_addr, AppClientNode._CONTROL_START_MSG)
                self._instances_ready = True

            buf = await self._control_server.receive(self._client_addr)
            if buf == AppClientNode._CONTROL_STOP_MSG:
                self._stage_comm.signal_global_stop()
                break

            if self._stop:
                break

        self._control_server.shutdown()


@final
class AppRequestHandlerServerNode:
    name: str

    _request_recv_address: IPv4Address
    _request_recv_server: TCPServerSocketAsync
    _client_addr: IPv4Address

    _request_recv_callback: Callable[[AppRequest], None] | Callable[[AppRequest], Coroutine[Any, Any, None]]
    _callback_is_coroutine: bool
    _task_runner: AsyncConcurrentProcessor
    _stage_comm: StageCommunicator

    _poll_interval: float
    _ready: bool
    _stop: bool

    def __init__(
        self: Self,
        *,
        request_recv_address: IPv4Address,
        name: str,
        request_recv_callback: Callable[[AppRequest], None] | Callable[[AppRequest], Coroutine[Any, Any, None]],
        stage_comm: StageCommunicator,
        poll_interval: float = 10
    ) -> None:

        self.name = name

        self._request_recv_address = request_recv_address
        self._request_recv_server = TCPServerSocketAsync(request_recv_address)

        self._request_recv_callback = request_recv_callback
        self._callback_is_coroutine = asyncio.iscoroutinefunction(self._request_recv_callback)
        if self._callback_is_coroutine:
            self._task_runner = AsyncConcurrentProcessor()
        self._stage_comm = stage_comm

        self._poll_interval = poll_interval
        self._ready = False
        self._stop = False

    async def initialize(
        self: Self
    ) -> None:

        logger.debug("Initializing {} app request handler server node".format(self.name))

        success = await self._request_recv_server.start()
        assert success
        self._client_addr = self._request_recv_server.get_client_addresses()[0]
        self._ready = True

    def stop(
        self: Self
    ) -> None:

        logger.debug("Stopping {} app request handler server node".format(self.name))

        self._stop = True

    @staticmethod
    def _parse_request_packet(
        buf: bytes
    ) -> AppRequest:

        return AppRequest(struct.unpack('I', buf[:4])[0], buf[4:])

    async def run(
        self: Self
    ) -> None:

        assert self._ready
        logger.debug("Running {} app request handler server node".format(self.name))

        while True:
            if await self._request_recv_server.is_client_packet_waiting(self._client_addr, timeout=self._poll_interval):
                buf = await self._request_recv_server.receive(self._client_addr)
                req = self._parse_request_packet(buf)
                if self._callback_is_coroutine:
                    coro = self._request_recv_callback(req)
                    assert coro is not None
                    await self._task_runner.add_task(coro)
                else:
                    self._request_recv_callback(req)
            else:
                logger.debug("Timeout, did not recieve request")

            if self._stage_comm.should_stop():
                self.stop()
            if self._stop:
                break

        self._request_recv_server.shutdown()


@final
class AppResponseHandlerServerNode:
    name: str

    _response_send_address: IPv4Address
    _response_send_server: TCPServerSocketAsync
    _client_addr: IPv4Address

    _responses_to_send: asyncio.Queue[AppResponse]
    _stage_comm: StageCommunicator

    _poll_interval: float
    _ready: bool
    _stop: bool

    def __init__(
        self: Self,
        *,
        response_send_address: IPv4Address,
        name: str,
        stage_comm: StageCommunicator,
        poll_interval: float = 10,
    ) -> None:

        self.name = name
        self._response_send_address = response_send_address

        self._response_send_server = TCPServerSocketAsync(response_send_address)

        self._responses_to_send = asyncio.Queue()
        self._stage_comm = stage_comm

        self._poll_interval = poll_interval
        self._ready = False
        self._stop = False

    async def initialize(
        self: Self
    ) -> None:

        logger.debug("Initializing {} app response handler server node".format(self.name))

        await self._response_send_server.start()
        self._client_addr = self._response_send_server.get_client_addresses()[0]
        self._ready = True

    def stop(
        self: Self
    ) -> None:

        logger.debug("Stopping {} app response handler server node".format(self.name))

        self._stop = True

    async def add_response(
        self: Self,
        response: AppResponse
    ) -> None:

        await self._responses_to_send.put(response)

    @staticmethod
    def _construct_response_packet(
        response: AppResponse
    ) -> bytes:

        return struct.pack('I', response.user_request_id) + response.message

    async def run(
        self: Self
    ) -> None:

        assert self._ready
        logger.debug("Running {} app response handler server node".format(self.name))

        while True:
            res = None
            try:
                res = await asyncio.wait_for(self._responses_to_send.get(), self._poll_interval)
            except asyncio.TimeoutError:
                pass
            if res is not None:

                buf = self._construct_response_packet(res)
                await self._response_send_server.send(self._client_addr, buf)

            if self._stage_comm.should_stop():
                self.stop()
            if self._stop:
                break

        self._response_send_server.shutdown()


class UniformSourceClientNode(AppClientNode):
    _rate: float
    _scale: float

    def __init__(
        self: Self,
        *,
        control_address: IPv4Address,
        request_send_address: IPv4Address,
        response_recv_address: IPv4Address,
        name: str,
        log_fname: str,
        rate: float = 1,
        log_interval: float = 1,
        send_length: float | None = None
    ) -> None:

        super().__init__(
            control_address=control_address,
            request_send_address=request_send_address,
            response_recv_address=response_recv_address,
            name=name,
            log_fname=log_fname,
            log_interval=log_interval,
            send_length=send_length
        )

        self._rate = rate
        self._scale = 1 / rate

    def _get_next_request_wait_time(
        self: Self
    ) -> float:

        return self._scale


class PossionSourceClientNode(AppClientNode):
    _rate: float
    _scale: float

    def __init__(
        self: Self,
        *,
        control_address: IPv4Address,
        request_send_address: IPv4Address,
        response_recv_address: IPv4Address,
        name: str,
        log_fname: str,
        rate: float = 1,
        log_interval: float = 1,
        send_length: float | None = None,
        rng: np.random.Generator | None = None
    ) -> None:

        super().__init__(
            control_address=control_address,
            request_send_address=request_send_address,
            response_recv_address=response_recv_address,
            name=name,
            log_fname=log_fname,
            log_interval=log_interval,
            send_length=send_length
        )

        self._rate = rate
        self._scale = 1 / rate

        self._rng = rng or np.random.default_rng()

    def _get_next_request_wait_time(
        self: Self
    ) -> float:

        return self._rng.exponential(self._scale)


@final
class InstanceComputeLatencyLogger:
    _instance_name: str

    _stage_in_times: dict[Hashable, float]
    _stage_out_min_times: dict[Hashable, float]
    _stage_out_max_times: dict[Hashable, float]
    _stage_out_num_times: dict[Hashable, int]
    _lock: threading.Lock

    _log_fname: str
    _log_interval: float
    _last_log_write_time: float

    def __init__(
        self: Self,
        instance_name: str,
        log_interval: float = 2,
    ) -> None:

        super().__init__()

        self._instance_name = instance_name

        self._stage_in_times = {}
        self._stage_out_min_times = {}
        self._stage_out_max_times = {}
        self._stage_out_num_times = {}
        self._lock = threading.Lock()

        os.makedirs(get_rel_pkg_path("logs/instances_compute/"), exist_ok=True)
        self._log_fname = get_rel_pkg_path(
            "logs/instances_compute/{}.json".format(self._instance_name)
        )
        self._log_interval = log_interval
        self._last_log_write_time = -1

    def _get_summary_stats(
        self: Self,
        vals: list[float] | list[int]
    ) -> dict[str, Any]:

        empty = len(vals) == 0
        return {
            'avg': None if empty else float(np.mean(vals)),
            'median': None if empty else float(np.median(vals)),
            'p99': None if empty else float(np.percentile(vals, 99)),
            'count': len(vals),
        }

    def log_metadata(
        self: Self
    ) -> None:

        if not DETAILED_LOGS:
            return

        data = {
            'instance_name': self._instance_name,
        }

        buf = json.dumps(data)
        with open(self._log_fname, 'a+') as f:
            f.write(buf + "\n")

    def log_counts(
        self: Self
    ) -> None:

        if not DETAILED_LOGS:
            return

        current_time = system_time()

        self._lock.acquire()
        min_latencies = [
            self._stage_out_min_times[key] - self._stage_in_times[key]
            for key in self._stage_out_min_times
        ]
        max_latencies = [
            self._stage_out_max_times[key] - self._stage_in_times[key]
            for key in self._stage_out_max_times
        ]
        num_outputs = list(self._stage_out_num_times.values())
        self._lock.release()

        data: dict[str, Any] = {
            'time': current_time,
            'stage_min_latency': self._get_summary_stats(min_latencies),
            'stage_max_latency': self._get_summary_stats(max_latencies),
            'stage_out_num_time': self._get_summary_stats(num_outputs)
        }
        buf = json.dumps(data)
        with open(self._log_fname, 'a+') as f:
            f.write(buf + "\n")

    def update_stage_queue_in(
        self: Self,
        key: Hashable
    ) -> None:

        if not DETAILED_LOGS:
            return

        current_time = system_time()
        self._lock.acquire()
        self._stage_in_times[key] = current_time
        self._lock.release()

    def update_stage_queue_out(
        self: Self,
        key: Hashable
    ) -> None:

        if not DETAILED_LOGS:
            return

        current_time = system_time()

        self._lock.acquire()
        self._stage_out_min_times[key] = min(
            current_time,
            self._stage_out_min_times.get(key, float('inf'))
        )
        self._stage_out_max_times[key] = max(
            current_time,
            self._stage_out_max_times.get(key, -float('inf'))
        )
        self._stage_out_num_times[key] = self._stage_out_num_times.get(key, 0) + 1
        self._lock.release()

    def update_log(
        self: Self,
        force: bool = False
    ) -> None:

        if not DETAILED_LOGS:
            return

        if force or (system_time() - self._last_log_write_time > self._log_interval):
            self._last_log_write_time = system_time()
            self.log_counts()


@final
class InstanceQueueLatencyLogger:
    _instance_name: str
    _item_type_name: str

    _message_latencies: list[float]

    _log_fname: str
    _log_interval: float
    _last_log_write_time: float

    def __init__(
        self: Self,
        instance_name: str,
        item_type_name: str,
        log_interval: float = 2,
    ) -> None:

        super().__init__()

        self._instance_name = instance_name
        self._item_type_name = item_type_name

        self._message_latencies = []

        os.makedirs(get_rel_pkg_path("logs/instances_queue/"), exist_ok=True)
        self._log_fname = get_rel_pkg_path(
            "logs/instances_queue/{}_{}.json".format(self._instance_name, item_type_name)
        )
        self._log_interval = log_interval
        self._last_log_write_time = -1

    def _get_summary_stats(
        self: Self,
        vals: list[float] | list[int]
    ) -> dict[str, Any]:

        empty = len(vals) == 0
        return {
            'avg': None if empty else float(np.mean(vals)),
            'median': None if empty else float(np.median(vals)),
            'p99': None if empty else float(np.percentile(vals, 99)),
            'count': len(vals),
        }

    def log_metadata(
        self: Self
    ) -> None:

        if not DETAILED_LOGS:
            return

        data = {
            'instance_name': self._instance_name,
            'item_type_name': self._item_type_name
        }

        buf = json.dumps(data)
        with open(self._log_fname, 'a+') as f:
            f.write(buf + "\n")

    def log_counts(
        self: Self
    ) -> None:

        if not DETAILED_LOGS:
            return

        current_time = system_time()

        data: dict[str, Any] = {
            'time': current_time,
            'latency': self._get_summary_stats(self._message_latencies)
        }
        buf = json.dumps(data)
        with open(self._log_fname, 'a+') as f:
            f.write(buf + "\n")

    def send_message(
        self: Self,
        msg: bytes
    ) -> bytes:

        if not DETAILED_LOGS:
            return msg

        current_time = system_time()
        buf = struct.pack('d', current_time) + msg

        return buf

    def recv_messsage(
        self: Self,
        buf: bytes
    ) -> bytes:

        if not DETAILED_LOGS:
            return buf

        current_time = system_time()
        send_time, msg = struct.unpack('d', buf[:8])[0], buf[8:]
        self._message_latencies.append(current_time - send_time)

        return msg

    def update_log(
        self: Self,
        force: bool = False
    ) -> None:

        if not DETAILED_LOGS:
            return

        if force or (system_time() - self._last_log_write_time > self._log_interval):
            self._last_log_write_time = system_time()
            self.log_counts()
