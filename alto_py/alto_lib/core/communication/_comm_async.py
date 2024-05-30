# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:30:08 2023

@author: Shahir
"""

import abc
import socket
import warnings
import asyncio
import selectors
from typing import Self, Any, ClassVar, Literal

from alto_lib.core.communication._comm_sync import IPv4Address, _tcp_get_size_data

__all__ = ['TCPClientSocketAsync', 'UDPClientSocketAsync', 'TCPServerSocketAsync', 'UDPServerSocketAsync']


async def _tcp_send_sized_async(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    data: bytes
) -> None:

    size_data = _tcp_get_size_data(len(data))
    await loop.sock_sendall(sock, bytes(size_data) + data)


async def _tcp_receive_size_data_async(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    next_data: bytearray
) -> tuple[int, bytearray]:

    size_data = bytearray()

    while True:
        if len(next_data):
            chunk_bytes_left = (len(size_data) & 0b11) or 4
            size_data.extend(next_data[:chunk_bytes_left])
            next_data = next_data[chunk_bytes_left:]
        if len(size_data) >= 4:
            if size_data[-1] >> 7:
                size_data[-1] = size_data[-1] & 0x7f
                continue
            break
        new = await loop.sock_recv(sock, 8192)
        next_data.extend(new)

    size = 0
    while len(size_data):
        size = (size << 31) | int.from_bytes(size_data[-4:], 'little')
        size_data = size_data[:-4]

    return size, next_data


async def _tcp_receive_sized_async(
    loop: asyncio.AbstractEventLoop,
    sock: socket.socket,
    next_data: bytearray
) -> tuple[bytes, bytearray]:

    # when reading the next packet do not discard the data that was read in the past call
    size, next_data = await _tcp_receive_size_data_async(loop, sock, next_data)

    while len(next_data) < size:
        next_data.extend(await loop.sock_recv(sock, 8192))

    packet_data = bytes(next_data[:size])
    next_data = next_data[size:]

    return packet_data, next_data


class BaseClientSocketAsync(metaclass=abc.ABCMeta):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = NotImplemented
    ADDRESS_FAMILY: ClassVar[socket.AddressFamily] = socket.AF_INET
    ALLOW_REUSE_ADDRESS: ClassVar[bool] = False

    address: IPv4Address

    _sock: socket.socket
    _loop: asyncio.AbstractEventLoop
    _client_start: bool

    def __init__(
        self: Self,
        address: IPv4Address,
    ) -> None:

        self.address = address

        self._sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)
        self._sock.setblocking(False)
        if self.ALLOW_REUSE_ADDRESS:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._loop = asyncio.get_event_loop()

        self._client_start = False

    @abc.abstractmethod
    async def connect(
        self: Self
    ) -> bool:

        pass

    def shutdown(
        self: Self
    ) -> bool:

        if not self._client_start:
            return False

        self._client_start = False

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN here

        self._sock.close()

        return True

    async def is_packet_waiting(
        self: Self,
        timeout: float = 0
    ) -> bool:

        try:
            fut = self._loop.create_future()
            self._loop.add_reader(self._sock, fut.set_result, None)
            fut.add_done_callback(lambda f: self._loop.remove_reader(self._sock))
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @abc.abstractmethod
    async def send(
        self: Self,
        data: bytes
    ) -> None:

        pass

    @abc.abstractmethod
    async def receive(
        self: Self
    ) -> bytes:

        pass

    async def packet_cycle(
        self: Self,
        data: bytes,
        fail_shutdown: bool = False
    ) -> tuple[bool, bytes | Exception]:
        """
        Complete one receive-send cycle and processes errors

        The return value is received data and the error condition
        """

        if not self._client_start:
            raise Exception("Client not started or was closed.")

        try:
            await self.send(data)
            receive = await self.receive()

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()

            return False, e

        return True, receive


class TCPClientSocketAsync(BaseClientSocketAsync):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_STREAM

    _next_data: bytearray

    def __init__(
        self: Self,
        address: IPv4Address
    ) -> None:

        super().__init__(address)

        self._next_data = bytearray()

    async def connect(
        self: Self
    ) -> bool:

        try:
            await self._loop.sock_connect(self._sock, self.address)  # connect to the server
            self._client_start = True
            return True

        except socket.error:
            return False

    async def connect_wait_for_server(
        self: Self
    ) -> bool:

        while True:
            if await self.connect():
                break

        return True

    async def send(
        self: Self,
        data: bytes
    ) -> None:

        await _tcp_send_sized_async(self._loop, self._sock, data)

    async def receive(
        self: Self
    ) -> bytes:

        data, self._next_data = await _tcp_receive_sized_async(self._loop, self._sock, self._next_data)

        return data


class UDPClientSocketAsync(BaseClientSocketAsync):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_DGRAM
    MAX_PACKET_SIZE: ClassVar[int] = 65507

    async def connect(
        self: Self
    ) -> bool:

        await self._loop.sock_connect(self._sock, self.address)
        self._client_start = True

        return True

    async def send(
        self: Self,
        data: bytes
    ) -> None:

        await self._loop.sock_sendto(self._sock, data, self.address)

    async def receive(
        self: Self
    ) -> bytes:

        return await self._loop.sock_recv(self._sock, self.MAX_PACKET_SIZE)


class BaseServerSocketAsync(metaclass=abc.ABCMeta):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = NotImplemented
    ADDRESS_FAMILY: ClassVar[socket.AddressFamily] = socket.AF_INET
    ALLOW_REUSE_ADDRESS: ClassVar[bool] = True

    address: IPv4Address

    _sock: socket.socket
    _loop: asyncio.AbstractEventLoop
    _server_start: bool

    def __init__(
        self: Self,
        address: IPv4Address
    ) -> None:

        self.address = address

        self._sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)
        self._sock.setblocking(False)
        if self.ALLOW_REUSE_ADDRESS:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._loop = asyncio.get_event_loop()

        self._server_start = False

    @abc.abstractmethod
    async def start(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    def shutdown(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    async def send(
        self: Self,
        addr: IPv4Address,
        data: bytes
    ) -> None:

        pass


class TCPServerSocketAsync(BaseServerSocketAsync):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_STREAM

    _num_clients: int
    _client_socks: dict[IPv4Address, socket.socket]
    _client_next_data: dict[IPv4Address, bytearray]
    _all_clients_selector: selectors.BaseSelector

    def __init__(
        self: Self,
        address: IPv4Address,
        num_clients: int = 1
    ) -> None:

        super().__init__(address)

        self._num_clients = num_clients

        self._client_socks = {}
        self._client_next_data = {}

        self._all_clients_selector = selectors.DefaultSelector()

    async def _listen(
        self: Self,
        num_clients: int
    ) -> bool:

        self._sock.listen(num_clients)  # how many connections it can receive at one time

        for _ in range(num_clients):
            try:
                conn, addr = await self._loop.sock_accept(self._sock)  # accept the connection
            except socket.timeout:
                return False

            self._client_socks[addr] = conn
            self._client_next_data[addr] = bytearray()

            conn.setblocking(False)
            self._all_clients_selector.register(conn, selectors.EVENT_READ)

        return True

    async def start(
        self: Self,
        verbose: bool = True
    ) -> bool:

        if verbose:
            print("Starting TCP Server")
            print("Hostname:", socket.gethostname())
            print("Address:", self.address)

        self._sock.bind(self.address)
        success = await self._listen(self._num_clients)
        if success:
            self._server_start = True

        return success

    def get_client_addresses(
        self: Self
    ) -> list[IPv4Address]:

        return list(self._client_socks.keys())

    def shutdown(
        self: Self
    ) -> bool:

        if not self._server_start:
            return False

        self._server_start = False

        for addr, conn in self._client_socks.items():
            try:
                # explicitly shutdown. socket.close() merely releases
                # the socket and waits for GC to perform the actual close.
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # some platforms may raise ENOTCONN here
            conn.close()

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN her

        self._sock.close()

        return True

    async def is_client_packet_waiting(
        self: Self,
        addr: IPv4Address,
        timeout: float = 0
    ) -> bool:

        try:
            fut = self._loop.create_future()
            self._loop.add_reader(self._client_socks[addr], fut.set_result, None)
            fut.add_done_callback(lambda f: self._loop.remove_reader(self._client_socks[addr]))
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_waiting_clients(
        self: Self,
        timeout: float | None = None
    ) -> list[IPv4Address]:

        selected = self._all_clients_selector.select(timeout=timeout)
        if not selected:
            return []

        # only get connection objects
        conns = [n.fileobj for n in list(zip(*selected))[0]]  # type: ignore
        # return addresses not connections
        addrs = [conn.getpeername() for conn in conns]  # type: ignore

        return addrs  # type: ignore

    async def send(
        self: Self,
        addr: IPv4Address,
        data: bytes
    ) -> None:

        await _tcp_send_sized_async(self._loop, self._client_socks[addr], data)

    async def receive(
        self: Self,
        addr: IPv4Address
    ) -> bytes:

        data, self._client_next_data[addr] = await _tcp_receive_sized_async(
            self._loop,
            self._client_socks[addr],
            self._client_next_data[addr]
        )

        return data

    async def packet_cycle(
        self: Self,
        addr: IPv4Address,
        data: bytes,
        fail_shutdown: bool = False
    ) -> tuple[bool, bytes | Exception]:
        """
        Complete one receive-send cycle and processes errors

        The return value is whether it was successful or not, the received data or error if any
        """

        if not self._server_start:
            raise Exception("Server not started or was closed")

        try:
            receive = await self.receive(addr)
            await self.send(addr, data)

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()
            return False, e

        return True, receive


class UDPServerSocketAsync(BaseServerSocketAsync):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_DGRAM
    MAX_PACKET_SIZE: ClassVar[int] = 65507

    def __init__(
        self: Self,
        address: IPv4Address,
        broadcast: bool = False
    ) -> None:

        super().__init__(address)

        if broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    async def start(
        self: Self,
        verbose: bool = True
    ) -> bool:

        if verbose:
            print("Starting UDP Server")
            print("Hostname:", socket.gethostname())
            print("Address:", self.address)

        self._sock.bind(self.address)
        self._server_start = True

        return True

    def shutdown(
        self: Self
    ) -> bool:

        if not self._server_start:
            return False

        self._server_start = False

        try:
            # explicitly shutdown. socket.close() merely releases
            # the socket and waits for GC to perform the actual close.
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # some platforms may raise ENOTCONN here

        self._sock.close()

        return True

    async def is_packet_waiting(
        self: Self,
        timeout: float = 0
    ) -> bool:

        try:
            fut = self._loop.create_future()
            self._loop.add_reader(self._sock, fut.set_result, None)
            fut.add_done_callback(lambda f: self._loop.remove_reader(self._sock))
            await asyncio.wait_for(fut, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send(
        self: Self,
        addr: IPv4Address,
        data: bytes
    ) -> None:

        await self._loop.sock_sendto(self._sock, data, addr)

    async def receive(
        self: Self
    ) -> tuple[bytes, IPv4Address]:

        data, addr = await self._loop.sock_recvfrom(self._sock, self.MAX_PACKET_SIZE)

        return data, addr

    async def packet_cycle(
        self: Self,
        data: bytes,
        fail_shutdown: bool = False
    ) -> tuple[Literal[True], tuple[bytes, IPv4Address]] | tuple[Literal[False], Exception]:
        """
        Complete one receive-send cycle and processes errors
        Not guaranteed to actually communicate with a specific address

        The return value is whether it was successful or not, the received data, and the address or error if any
        """

        warnings.warn(
            "Using UDP socket: not guaranteed to communicate with any specific address,"
            " should only be used for connections with 1 client"
        )

        if not self._server_start:
            raise Exception("Server not started or was closed.")

        try:
            receive, recv_addr = await self.receive()
            await self.send(recv_addr, data)

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()
            return False, e

        return True, (receive, recv_addr)


if __name__ == '__main__':
    import time
    import random
    import secrets
    import warnings

    # some basic functionality tests

    warnings.filterwarnings('ignore')

    addr = ('localhost', 8888)

    def get_random_small_payload(
        max_exp: int = 15
    ) -> bytes:

        size = random.randint(0, 1 << max_exp)

        return secrets.token_bytes(size)

    def get_random_large_size(
        max_exp: int = 256
    ) -> int:

        return random.randint(0, 1 << max_exp)

    async def test1_server_proc(
        tcp: bool,
        payload: bytes
    ) -> tuple[bool, Any]:

        if tcp:
            server_tcp = TCPServerSocketAsync(addr)
            success = await server_tcp.start(verbose=False)
            assert success
            client_addr = server_tcp.get_client_addresses()[0]
            out = await server_tcp.packet_cycle(client_addr, payload)
            assert out[0]
            server_tcp.shutdown()
        else:
            server_udp = UDPServerSocketAsync(addr)
            success = await server_udp.start(verbose=False)
            assert success
            out_udp = await server_udp.packet_cycle(payload)
            assert out_udp[0]
            out = (out_udp[0], out_udp[1][0])
            server_udp.shutdown()

        return out

    async def test1_client_proc(
        tcp: bool,
        payload: bytes
    ) -> tuple[bool, Any]:

        if tcp:
            client_tcp = TCPClientSocketAsync(addr)
            success = await client_tcp.connect_wait_for_server()
            assert success
            out = await client_tcp.packet_cycle(payload)
            client_tcp.shutdown()
        else:
            client_udp = UDPClientSocketAsync(addr)
            success = await client_udp.connect()
            assert success
            out = await client_udp.packet_cycle(payload)
            client_udp.shutdown()

        return out

    async def test2_tcp_server_size_proc(
        size: int,
        payload: bytes
    ) -> int:

        server = TCPServerSocketAsync(addr)
        await server.start(verbose=False)
        client_addr = server.get_client_addresses()[0]
        size_data = _tcp_get_size_data(size)
        recv_size, _ = await _tcp_receive_size_data_async(
            server._loop,  # type: ignore[reportPrivateUsage]
            server._client_socks[client_addr],  # type: ignore[reportPrivateUsage]
            server._client_next_data[client_addr]  # type: ignore[reportPrivateUsage]
        )
        # large size, fake a small payload due to memory constraints
        await server._loop.sock_sendall(  # type: ignore[reportPrivateUsage]
            server._client_socks[client_addr],  # type: ignore[reportPrivateUsage]
            size_data + payload
        )
        server.shutdown()

        return recv_size

    async def test2_tcp_client_size_proc(
        size: int,
        payload: bytes
    ) -> int:

        client = TCPClientSocketAsync(addr)
        await client.connect_wait_for_server()
        size_data = _tcp_get_size_data(size)
        # large size, fake a small payload due to memory constraints
        await client._loop.sock_sendall(
            client._sock,  # type: ignore[reportPrivateUsage]
            size_data + payload
        )
        recv_size, _ = await _tcp_receive_size_data_async(
            client._loop,  # type: ignore[reportPrivateUsage]
            client._sock,  # type: ignore[reportPrivateUsage]
            client._next_data  # type: ignore[reportPrivateUsage]
        )
        client.shutdown()

        return recv_size

    async def test3_server_proc(
        tcp: bool
    ) -> None:

        if tcp:
            server_tcp = TCPServerSocketAsync(addr)
            success = await server_tcp.start(verbose=False)
            assert success
            client_addr = server_tcp.get_client_addresses()[0]
            for _ in range(5):
                if await server_tcp.is_client_packet_waiting(client_addr, timeout=10):
                    await server_tcp.receive(client_addr)
                else:
                    raise ValueError()
            server_tcp.shutdown()
        else:
            server_udp = UDPServerSocketAsync(addr)
            success = await server_udp.start(verbose=False)
            assert success
            for i in range(5):
                if await server_udp.is_packet_waiting(timeout=10):
                    await server_udp.receive()
                else:
                    raise ValueError()
            server_udp.shutdown()

    async def test3_client_proc(
        tcp: bool
    ) -> None:

        payload = b"asdf"
        if tcp:
            client_tcp = TCPClientSocketAsync(addr)
            success = await client_tcp.connect_wait_for_server()
            assert success
            for _ in range(5):
                await client_tcp.send(payload)
                await asyncio.sleep(1)
            client_tcp.shutdown()
        else:
            client_udp = UDPClientSocketAsync(addr)
            success = await client_udp.connect()
            assert success
            for i in range(5):
                await client_udp.send(payload)
                await asyncio.sleep(1)
            client_udp.shutdown()

    async def run_tests(
    ) -> None:

        start_time = time.time()

        # # test random payloads sent between server and client and check if they match for both TCP and UDP packets
        for tcp in [True, False]:
            for _ in range(20):
                payload1 = get_random_small_payload()
                payload2 = get_random_small_payload()
                test1 = await asyncio.gather(
                    test1_server_proc(tcp=tcp, payload=payload1),
                    test1_client_proc(tcp=tcp, payload=payload2)
                )

                server_out1, client_out1 = test1
                assert server_out1 == (True, payload2)
                assert client_out1 == (True, payload1)

        # test very large payload sizes (more than 31 bits to store size) and see if processing functions for TCP packets
        for _ in range(20):
            payload1 = get_random_small_payload()
            payload2 = get_random_small_payload()
            size1 = get_random_large_size()
            size2 = get_random_large_size()

            test2 = await asyncio.gather(
                test2_tcp_server_size_proc(size=size1, payload=payload1),
                test2_tcp_client_size_proc(size=size2, payload=payload2)
            )

            server_out2, client_out2 = test2
            assert server_out2 == size2
            assert client_out2 == size1

        for tcp in [True, False]:
            await asyncio.gather(
                test3_server_proc(tcp=tcp),
                test3_client_proc(tcp=tcp)
            )

        print("Tests took", time.time() - start_time, "seconds")

    asyncio.run(run_tests())
