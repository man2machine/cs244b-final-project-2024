# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:12:04 2024

@author: Shahir
"""

import abc
import struct
from dataclasses import dataclass
from typing import Self, TypeVar, Generic, final

from alto_lib.utils import BytesSerializable
from alto_lib.core.manager._base import QueueItemSerializer


class QueueMessage(BytesSerializable):
    @abc.abstractmethod
    def __str__(
        self: Self
    ) -> str:

        pass

    def __repr__(
        self: Self
    ) -> str:

        return self.__str__()


QueueMessageT = TypeVar('QueueMessageT', bound=QueueMessage)


class QueueMessageSerializer(Generic[QueueMessageT], QueueItemSerializer[QueueMessageT]):
    message_type: QueueMessageT

    def __init__(
        self: Self,
        message_type: type[QueueMessageT]
    ) -> None:

        self.message_type

    def to_bytes(
        self: Self,
        item: QueueMessageT
    ) -> bytes:

        return item.to_bytes()

    def from_bytes(
        self: Self,
        data: bytes
    ) -> QueueMessageT:

        item = self.message_type.from_bytes(data)

        return item


@final
@dataclass(kw_only=True, frozen=True, slots=True)
class QueueRequestMessage(Generic[QueueMessageT]):
    user_request_id: int
    message: QueueMessageT


@final
class QueueRequestMessageSerializer(Generic[QueueMessageT], QueueItemSerializer[QueueRequestMessage[QueueMessageT]]):
    message_type: type[QueueMessageT]

    def __init__(
        self: Self,
        message_type: type[QueueMessageT]
    ) -> None:

        self.message_type = message_type

    def to_bytes(
        self: Self,
        item: QueueRequestMessage[QueueMessageT]
    ) -> bytes:

        return struct.pack('L', item.user_request_id) + item.message.to_bytes()

    def from_bytes(
        self: Self,
        data: bytes
    ) -> QueueRequestMessage[QueueMessageT]:

        item = QueueRequestMessage(
            user_request_id=struct.unpack('L', data[:8])[0],
            message=self.message_type.from_bytes(data[8:])
        )

        return item
