# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:30:08 2023

@author: Shahir
"""

import abc
import socket
import selectors
import warnings
from typing import Self, Any, ClassVar, Literal


__all__ = ['IPv4Address', 'TCPClientSocket', 'UDPClientSocket', 'TCPServerSocket', 'UDPServerSocket']

IPv4Address = tuple[str, int]


def _tcp_get_size_data(
    size: int
) -> bytes:

    size_data = bytearray()
    while size >= 0x7fffffff:
        size_data.extend((size & 0x7fffffff).to_bytes(4, 'little'))
        size_data[-1] = size_data[-1] | 0x80
        size = size >> 31
    size_data.extend(size.to_bytes(4, 'little'))

    return size_data


def _tcp_send_sized(
    sock: socket.socket,
    data: bytes
) -> None:

    size_data = _tcp_get_size_data(len(data))
    sock.sendall(bytes(size_data) + data)


def _tcp_receive_size_data(
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
        new = sock.recv(8192)
        next_data.extend(new)

    size = 0
    while len(size_data):
        size = (size << 31) | int.from_bytes(size_data[-4:], 'little')
        size_data = size_data[:-4]

    return size, next_data


def _tcp_receive_sized(
    sock: socket.socket,
    next_data: bytearray
) -> tuple[bytes, bytearray]:

    # when reading the next packet do not discard the data that was read in the past call
    size, next_data = _tcp_receive_size_data(sock, next_data)

    while len(next_data) < size:
        next_data.extend(sock.recv(8192))

    packet_data = bytes(next_data[:size])
    next_data = next_data[size:]

    return packet_data, next_data


class BaseClientSocket(metaclass=abc.ABCMeta):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = NotImplemented
    ADDRESS_FAMILY: ClassVar[socket.AddressFamily] = socket.AF_INET
    ALLOW_REUSE_ADDRESS: ClassVar[bool] = False

    address: IPv4Address

    _sock: socket.socket
    _blocking: bool
    _timeout: float
    _selector: selectors.BaseSelector
    _client_start: bool

    def __init__(
        self: Self,
        address: IPv4Address,
        blocking: bool = True,
        timeout: float = 60
    ) -> None:

        self.address = address

        self._blocking = bool(blocking)
        self._timeout = float(timeout)

        self._sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)
        self._sock.setblocking(self._blocking)
        if self._timeout:
            self._sock.settimeout(self._timeout)
        if self.ALLOW_REUSE_ADDRESS:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._sock, selectors.EVENT_READ)

        self._client_start = False

    @abc.abstractmethod
    def connect(
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

    def set_timeout(
        self: Self,
        timeout: float
    ) -> None:

        self._timeout = timeout
        self._sock.settimeout(timeout)

    def is_packet_waiting(
            self: Self,
            timeout: float = 0) -> bool:

        return bool(self._selector.select(timeout=timeout))

    @abc.abstractmethod
    def send(
        self: Self,
        data: bytes
    ) -> None:

        pass

    @abc.abstractmethod
    def receive(
        self: Self
    ) -> bytes:

        pass

    def packet_cycle(
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
            self.send(data)
            receive = self.receive()

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()

            return False, e

        return True, receive


class TCPClientSocket(BaseClientSocket):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_STREAM

    _next_data: bytearray

    def __init__(
        self: Self,
        address: IPv4Address,
        blocking: bool = True,
        timeout: float = 60
    ) -> None:

        super().__init__(address, blocking=blocking, timeout=timeout)

        self._next_data = bytearray()

    def connect(
        self: Self
    ) -> bool:

        try:
            self._sock.connect(self.address)  # connect to the server
            self._client_start = True
            return True

        except socket.error:
            return False

    def connect_wait_for_server(
        self: Self
    ) -> bool:

        self._sock.setblocking(True)
        if self._timeout:
            self._sock.settimeout(5)
        while True:
            if self.connect():
                break
        if self._timeout:
            self._sock.settimeout(self._timeout)
        self._sock.setblocking(self._blocking)

        return True

    def send(
        self: Self,
        data: bytes
    ) -> None:

        _tcp_send_sized(self._sock, data)

    def receive(
        self: Self
    ) -> bytes:

        data, self._next_data = _tcp_receive_sized(self._sock, self._next_data)

        return data


class UDPClientSocket(BaseClientSocket):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_DGRAM
    MAX_PACKET_SIZE: ClassVar[int] = 65507

    def connect(
        self: Self
    ) -> bool:

        self._sock.connect(self.address)
        self._client_start = True

        return True

    def send(
        self: Self,
        data: bytes
    ) -> None:

        self._sock.sendto(data, self.address)

    def receive(
        self: Self
    ) -> bytes:

        return self._sock.recv(self.MAX_PACKET_SIZE)


class BaseServerSocket(metaclass=abc.ABCMeta):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = NotImplemented
    ADDRESS_FAMILY: ClassVar[socket.AddressFamily] = socket.AF_INET
    ALLOW_REUSE_ADDRESS: ClassVar[bool] = True

    address: IPv4Address

    _sock: socket.socket
    _blocking: bool
    _timeout: float
    _server_start: bool

    def __init__(
        self: Self,
        address: IPv4Address,
        blocking: bool = True,
        timeout: float = 60
    ) -> None:

        self.address = address

        self._blocking = bool(blocking)
        self._timeout = float(timeout)

        self._sock = socket.socket(self.ADDRESS_FAMILY, self.SOCKET_TYPE)
        self._sock.setblocking(self._blocking)
        if self._timeout:
            self._sock.settimeout(self._timeout)
        if self.ALLOW_REUSE_ADDRESS:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._server_start = False

    @abc.abstractmethod
    def start(
        self: Self
    ) -> bool:

        pass

    @abc.abstractmethod
    def shutdown(
        self: Self
    ) -> bool:

        pass

    def set_timeout(
        self: Self,
        timeout: float
    ) -> None:

        self._timeout = timeout
        self._sock.settimeout(timeout)

    @abc.abstractmethod
    def send(
        self: Self,
        addr: IPv4Address,
        data: bytes
    ) -> None:

        pass


class TCPServerSocket(BaseServerSocket):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_STREAM

    _num_clients: int
    _client_socks: dict[IPv4Address, socket.socket]
    _client_next_data: dict[IPv4Address, bytearray]
    _client_selectors: dict[IPv4Address, selectors.BaseSelector]
    _all_clients_selector: selectors.BaseSelector

    def __init__(
        self: Self,
        address: IPv4Address,
        blocking: bool = True,
        timeout: float = 60,
        num_clients: int = 1
    ) -> None:

        super().__init__(address, blocking=blocking, timeout=timeout)

        self._num_clients = num_clients

        self._client_socks = {}
        self._client_next_data = {}

        self._client_selectors = {}
        self._all_clients_selector = selectors.DefaultSelector()

    def _listen(
        self: Self,
        num_clients: int
    ) -> bool:

        self._sock.listen(num_clients)  # how many connections it can receive at one time

        for _ in range(num_clients):
            try:
                conn, addr = self._sock.accept()  # accept the connection
            except socket.timeout:
                return False

            self._client_socks[addr] = conn
            self._client_next_data[addr] = bytearray()

            conn.setblocking(self._blocking)
            conn.settimeout(self._timeout)

            selector = selectors.DefaultSelector()
            selector.register(conn, selectors.EVENT_READ)
            self._client_selectors[addr] = selector
            self._all_clients_selector.register(conn, selectors.EVENT_READ)

        return True

    def start(
        self: Self,
        verbose: bool = True
    ) -> bool:

        if verbose:
            print("Starting TCP Server")
            print("Hostname:", socket.gethostname())
            print("Address:", self.address)

        self._sock.bind(self.address)
        success = self._listen(self._num_clients)
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
                self._client_selectors[addr].close()
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

    def is_packet_waiting(
        self: Self,
        timeout: float = 0
    ) -> bool:

        return bool(self._all_clients_selector.select(timeout=timeout))

    def is_client_packet_waiting(
        self: Self,
        addr: IPv4Address,
        timeout: float = 0
    ) -> bool:

        return bool(self._client_selectors[addr].select(timeout=timeout))

    def get_waiting_clients(
        self: Self,
        timeout: float = 0
    ) -> list[IPv4Address]:

        selected = self._all_clients_selector.select(timeout=timeout)
        if not selected:
            return []

        # only get connection objects
        conns = [n.fileobj for n in list(zip(*selected))[0]]  # type: ignore
        # return addresses not connections
        addrs = [conn.getpeername() for conn in conns]  # type: ignore

        return addrs  # type: ignore

    def send(
        self: Self,
        addr: IPv4Address,
        data: bytes
    ) -> None:

        _tcp_send_sized(self._client_socks[addr], data)

    def receive(
        self: Self,
        addr: IPv4Address
    ) -> bytes:

        data, self._client_next_data[addr] = _tcp_receive_sized(
            self._client_socks[addr],
            self._client_next_data[addr]
        )

        return data

    def packet_cycle(
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
            receive = self.receive(addr)
            self.send(addr, data)

        except socket.timeout as e:
            if fail_shutdown:
                self.shutdown()
            return False, e

        return True, receive


class UDPServerSocket(BaseServerSocket):
    SOCKET_TYPE: ClassVar[socket.SocketKind] = socket.SOCK_DGRAM
    MAX_PACKET_SIZE: ClassVar[int] = 65507

    _selector: selectors.BaseSelector

    def __init__(
        self: Self,
        address: IPv4Address,
        blocking: bool = True,
        broadcast: bool = False,
        timeout: float = 60
    ) -> None:

        super().__init__(address, blocking=blocking, timeout=timeout)

        if broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._selector = selectors.DefaultSelector()
        self._selector.register(self._sock, selectors.EVENT_READ)

    def start(
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

    def is_packet_waiting(
        self: Self,
        timeout: float = 0
    ) -> bool:

        return bool(self._selector.select(timeout=timeout))

    def send(
        self: Self,
        addr: IPv4Address,
        data: bytes
    ) -> None:

        self._sock.sendto(data, addr)

    def receive(
        self: Self
    ) -> tuple[bytes, IPv4Address]:

        data, addr = self._sock.recvfrom(self.MAX_PACKET_SIZE)

        return data, addr

    def packet_cycle(
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
            receive, recv_addr = self.receive()
            self.send(recv_addr, data)

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
    from concurrent.futures import ThreadPoolExecutor

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

    def test1_server_proc(
        tcp: bool,
        payload: bytes
    ) -> tuple[bool, Any]:

        if tcp:
            server_tcp = TCPServerSocket(addr, timeout=5)
            success = server_tcp.start(verbose=False)
            assert success
            client_addr = server_tcp.get_client_addresses()[0]
            out = server_tcp.packet_cycle(client_addr, payload)
            assert out[0]
            server_tcp.shutdown()
        else:
            server_udp = UDPServerSocket(addr, timeout=5)
            success = server_udp.start(verbose=False)
            assert success
            out_udp = server_udp.packet_cycle(payload)
            assert out_udp[0]
            out = (out_udp[0], out_udp[1][0])
            server_udp.shutdown()

        return out

    def test1_client_proc(
        tcp: bool,
        payload: bytes
    ) -> tuple[bool, Any]:

        if tcp:
            client_tcp = TCPClientSocket(addr, timeout=5)
            success = client_tcp.connect_wait_for_server()
            assert success
            out = client_tcp.packet_cycle(payload)
            client_tcp.shutdown()
        else:
            client_udp = UDPClientSocket(addr, timeout=5)
            success = client_udp.connect()
            assert success
            out = client_udp.packet_cycle(payload)
            client_udp.shutdown()

        return out

    def test2_tcp_server_size_proc(
        size: int,
        payload: bytes
    ) -> int:

        server = TCPServerSocket(addr, timeout=5)
        server.start(verbose=False)
        client_addr = server.get_client_addresses()[0]
        size_data = _tcp_get_size_data(size)
        recv_size, _ = _tcp_receive_size_data(
            server._client_socks[client_addr],  # type: ignore[reportPrivateUsage]
            server._client_next_data[client_addr])  # type: ignore[reportPrivateUsage]
        # large size, fake a small payload due to memory constraints
        server._client_socks[client_addr].sendall(size_data + payload)  # type: ignore[reportPrivateUsage]
        server.shutdown()

        return recv_size

    def test2_tcp_client_size_proc(
        size: int,
        payload: bytes
    ) -> int:

        client = TCPClientSocket(addr, timeout=5)
        client.connect_wait_for_server()
        size_data = _tcp_get_size_data(size)
        # large size, fake a small payload due to memory constraints
        client._sock.sendall(size_data + payload)  # type: ignore[reportPrivateUsage]
        recv_size, _ = _tcp_receive_size_data(client._sock, client._next_data)  # type: ignore[reportPrivateUsage]
        client.shutdown()

        return recv_size

    def run_tests(
    ) -> None:

        start_time = time.time()

        # test random payloads sent between server and client and check if they match for both TCP and UDP packets
        for tcp in [True, False]:
            for _ in range(20):
                payload1 = get_random_small_payload()
                payload2 = get_random_small_payload()
                with ThreadPoolExecutor(max_workers=2) as executor:
                    server_fut1 = executor.submit(test1_server_proc, tcp=tcp, payload=payload1)
                    client_fut1 = executor.submit(test1_client_proc, tcp=tcp, payload=payload2)

                    server_out1 = server_fut1.result()
                    client_out1 = client_fut1.result()

                    assert server_out1 == (True, payload2)
                    assert client_out1 == (True, payload1)

                    executor.shutdown()

        # test very large payload sizes (more than 31 bits to store size) and see if processing functions for TCP packets
        for _ in range(20):
            payload1 = get_random_small_payload()
            payload2 = get_random_small_payload()
            size1 = get_random_large_size()
            size2 = get_random_large_size()
            with ThreadPoolExecutor(max_workers=2) as executor:
                server_fut2 = executor.submit(test2_tcp_server_size_proc, size=size1, payload=payload1)
                client_fut2 = executor.submit(test2_tcp_client_size_proc, size=size2, payload=payload2)

                server_out2 = server_fut2.result()
                client_out2 = client_fut2.result()

                assert server_out2 == size2
                assert client_out2 == size1

                executor.shutdown()

        print("Tests took", time.time() - start_time, "seconds")

    run_tests()
