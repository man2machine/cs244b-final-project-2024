# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:14:22 2024

@author: Shahir

What is a stage processor function?

Each "stage" is a collection of identical processes have each are instantiations of the Stage class. For a Stage class
the user has the ability to define functions which are decorated with the function processor to create a Processor
class. 

The input or output of a given processor can be a singular type (BytesSerializable) or a streaming type (Stream[T]).

We have the following operations the user can use:
- pmap applies a function A -> B to a streaming type Stream[A], resulting in a streaming type Stream[B].
- join operation combines two types A & B (which may or may not be streaming) into a single type Joined[A, B].
- flatten converts a 2d nested streaming type Stream[Stream[T]] to a 1d streaming type Stream[T].
- select
- apply

We apply the following constraints to the "processing functions" of a stage:
- Each processing function only takes one input argument, whether that is a regular or joined value.
- Each processing function only returns one output value, whether that is a single value T or an AsyncIterator[T].
- The output value of a processing function is either a single value or a streaming type, with at most one level of
nesting. This means that the output cannot be Stream[Stream[T]], it can only either be T or Stream[T].

After these conditions we are left with the following possibilities:
- Output is singular vs. output is a stream (T vs. Stream[T])
- Input is singular from one stage vs. input is a joined value from multiple stages (T, vs. Joined[A, B])

A joined type takes in two types A and B, either of which can be stream types, and results in a single non-stream type.

These possibilities result in 4 possible stages:
1. A -> B, where A may/may not be a Stream type and B is not a Stream types
2. A -> Stream[B], where A may/may not be a Stream type and B is not a Stream types
3. Joined[A, B] -> C, where either A or B can be Stream types, and C is not a Stream type
4. Joined[A, B] -> Stream[C], where either A or B can be Stream types, and C is not a Stream type

If we provision for all 4 of these stage types, we should theoretically be able to write any pipeline.

Examples of stage type 1 are question answering, bm25
Examples of stage type 2 are claim extraction, query generation
Examples of stage type 3 are reranker, claim appraisal
Examples of stage type 4 are ... (not in factool)

There is also a minor consideration of whether a the input is read in a batch (and output returned in batch
of the same size) or not. With this consideration we add the following stage types:
5. list[A] -> list[B], where A may/may not be a Stream type and B is not a Stream types
6. list[Joined[A, B]] -> list[C], where either A or B can be Stream types, and C is not a Stream type

Examples of stage type 5 are colbert encoder

How does joining work?

Consider the type following graph that follows a given pipeline
               
[level 0]    [level 0]   [level 2]  [level 3]   [level 3] 
 requests ━━━ answers ━━━ claims ━━━ queries ━┳━ docs
                                              ┗━ embedding

If we join docs and embeddings we get Joined[Docs, Embeddings], since both are on the same level. This is
the input to the reranker stage.

We increment the level each time when we apply a pmap operation. The type annotation is such that each time pmap
is applied it produces a stream level, hence this is done automatically.

We don't increment a level from requests to answers since pmap is not applied there, there is only
one answer per request. Similarly, we don't increment a level from queries to docs & embeddings
since there is only one docs and one embedding per query.

[level 0]    [level 0]   [level 2]  [level 3]   [level 3]         [level 3]         [level 3]
 requests ━━━ answers ━━━ claims ━┳━ queries ━┳━ docs ━━━━━━━┳━ reranked_docs ━┳━ verified_claims
                                  ┃           ┗━ embeddings ━┛                 ┃ 
                                  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

When we join the claims and the reranked docs we get a Joined[Claims, Stream[RerankedDocs]] type. We have a stream
type for the docs since we are one level deeper than the claims. This is *automatically* inferred by the type
annotations. Hence a user in using an IDE would automatically see the types of each item, and the IDE would throw
errors if the static types don't align.

Through similar means we can get Joined[Stream[A], B], Joined[Stream[A], Stream[B]], Joined[A, Stream[Stream[B]]], etc.
as possible types.

 A ━┳━ B ━━━ C ━━━ D
    ┗━ E ━━━ F

For example with this graph (assumimg pmap/a new ancestry level happens at each link), joining D and F would
result in Joined[Stream[Stream[Stream[D]]], Stream[Stream[F]]]. The user would have to perform this join call
at the scope in which A is processed (where pmap is applied to A), and this is the only scope where these variables
would be accessible. Because the pmap function requires a function as an input, it creates a new variable scope
each time within the function passed in. Hence it is not possible for Stream[Stream[D]] and F to be joined
for example, unless the user does some ugly global variable type stuff. So even statically the right joining
semantics are accounted for via python type annotations, and during runtime, this can be checked completely.

The user can use the flatten function before joining so that we get Joined[Stream[D], Stream[F]] instead. In order to 
flatten nested operations that happen as a result of joining after several levels of ancestry added due to pmap,
we can use use the flatten function.

How do stages get created?

Every stage should inherit from the Stage class. Any method in the stage class can be decorated with the @processor
function, which will note to the system that this will be part of the pipeline. The @generator function is used to
to annotate stages that are the start of the pipeline.

Every stage then specifies a create_stage method that returns an instance of the stage. This hides all the information
of the number of instances, etc. under the hood.

So for example with a stage like this (stage type 2):

class ExampleStage(Stage):
    _internal_per_instance_state: int
    
    def _internal_function(
        self: Self
    ) -> None:
        pass
        
    @processor
    async def do_stuff(
        self: Self,
        val: A
    ) -> AsyncIterator[B]:
    
        pass

The user can instantiate the stage with:
exmaple_stage = ExampleStage.create_stage()

Then the user has access to the ExampleStage.do_stuff method. However, any function not decorated with @processor
is not available. This is to ensure that the user only uses the functions that are registered parts of the pipeline.
The internal state that is stored in a stage is a per instance state, and is not shared between instances.
So all of the message passing and routing between multiple stages is done under the hood.

In order to employ thread-level parallelism within a stage, the user can use the Stage.spawn_iter and Stage.spawn
methods. These are used in the bm25 stage for example, where we want to process each token in parallel.

The LMGeneration class represents the output of a language model. It is a character stream, however it is not a Stream
type. The LMGeneration class offers methods like read_iter that allow the user to split the lm output into a stream
and also the to_str method that allows the user to convert the LMGeneration to a string.

The LMGeneration instance internally is sent over TCP. By default the stream has an <eos> at the end of the LM output.
The splitting just delineates to the orchestration that a <eos> is now put in place of the split type. Thus when the 
user uses read_iter or read_until on the output of the LM that is passed to a future stream, the character stream
is split such that the next stage processor function is called with a stream that ends at the split
(which is now <eos>). By doing this, we always preserve the lowest level of granularity of a text stream,
while still leaving the splitting semantics to the user, and not the orchestration.
"""

from __future__ import annotations

import abc
from typing import Self, Generic, TypeVar, Literal, ClassVar, Any, overload, final
from collections.abc import Callable, Awaitable, AsyncIterator, Iterator

import msgspec

from alto_lib.utils import BytesSerializable

T = TypeVar('T')
A = TypeVar('A')
B = TypeVar('B')


class Message(msgspec.Struct):
    @staticmethod
    def enc_hook(
        obj: Any
    ) -> bytes:

        if isinstance(obj, BytesSerializable):
            return obj.to_bytes()
        else:
            raise ValueError("Unsupported type {}".format(type(obj)))

    @staticmethod
    def dec_hook(
        obj_type: type[T],
        obj: Any
    ) -> T:

        if issubclass(obj_type, BytesSerializable):
            assert isinstance(obj, bytes)
            return obj_type.from_bytes(obj)  # type: ignore
        else:
            raise ValueError("Unsupported type {}".format(obj_type))

    def to_bytes(
        self: Self
    ) -> bytes:

        return msgspec.msgpack.encode(self, enc_hook=self.enc_hook)

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return msgspec.msgpack.decode(buf, type=cls, dec_hook=cls.dec_hook)


SerT = TypeVar('SerT', bound=BytesSerializable | Message)
SerAT = TypeVar('SerAT', bound=BytesSerializable | Message)
SerBT = TypeVar('SerBT', bound=BytesSerializable | Message)
SerCT = TypeVar('SerCT', bound=BytesSerializable | Message)


@final
class Stream(Generic[SerT], AsyncIterator[SerT], BytesSerializable, metaclass=abc.ABCMeta):
    def __aiter__(
        self: Self
    ) -> AsyncIterator[SerT]:

        return self

    async def __anext__(
        self: Self
    ) -> SerT:

        return NotImplemented

    async def to_list(
        self: Self
    ) -> list[SerT]:

        return NotImplemented

    def to_bytes(
        self: Self
    ) -> bytes:

        return NotImplemented

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return NotImplemented


class TextSplitter:
    @abc.abstractmethod
    def write(
        self: Self,
        data: str
    ) -> None:

        pass

    @abc.abstractmethod
    def read_chunk(
        self: Self
    ) -> str | None:

        pass

    def close(
        self: Self
    ) -> None:

        pass


SplitLevelT = TypeVar('SplitLevelT', bound=TextSplitter)
SplitLevelAT = TypeVar('SplitLevelAT', bound=TextSplitter)
SplitLevelBT = TypeVar('SplitLevelBT', bound=TextSplitter)


@final
class FullText(TextSplitter):
    def write(
        self: Self,
        data: str
    ) -> None:

        pass

    def read_chunk(
        self: Self
    ) -> str | None:

        return NotImplemented


class _SeparatorStringSplitter(TextSplitter):
    _SPLIT_STR: ClassVar[str] = NotImplemented

    def write(
        self: Self,
        data: str
    ) -> None:

        pass

    def read_chunk(
        self: Self
    ) -> str | None:

        return NotImplemented


@final
class Sentence(_SeparatorStringSplitter):
    _SPLIT_STR: ClassVar[str] = "."


@final
class Line(_SeparatorStringSplitter):
    _SPLIT_STR: ClassVar[str] = "\n"


@final
class Word(_SeparatorStringSplitter):
    _SPLIT_STR: ClassVar[str] = " "


@final
class RawTextPipe(BytesSerializable):
    @property
    def fininished(
        self: Self
    ) -> bool:

        return NotImplemented

    def read_batches(
        self: Self
    ) -> AsyncIterator[str]:

        return NotImplemented

    def write(
        self: Self,
        data: str
    ) -> None:

        pass

    def to_bytes(
        self: Self
    ) -> bytes:

        return NotImplemented

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return NotImplemented


@final
class LMTextPipe(Generic[SplitLevelAT], BytesSerializable):
    _splitter: SplitLevelAT

    @property
    def fininished(
        self: Self
    ) -> bool:

        return NotImplemented

    async def to_str(
        self: Self
    ) -> str:

        return NotImplemented

    def split(
        self: Self,
        split_level: type[SplitLevelBT]
    ) -> Stream[LMTextPipe[SplitLevelBT]]:

        return NotImplemented

    def abort(
        self: Self
    ) -> None:

        pass

    def to_bytes(
        self: Self
    ) -> bytes:

        return NotImplemented

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return NotImplemented


class Stage(metaclass=abc.ABCMeta):
    _lm_enabled: bool

    def __init__(
        self: Self,
        lm_enabled: bool = False
    ) -> None:

        self._lm_enabled = lm_enabled

    @classmethod
    def create_stage(
        cls: type[Self]
    ) -> Self:

        return NotImplemented

    def spawn_thread(
        self: Self,
        in_value: A,
        func: Callable[[A], Awaitable[B]]
    ) -> Awaitable[B]:

        return NotImplemented

    def spawn_thread_iter(
        self: Self,
        in_iterable: AsyncIterator[A] | Iterator[A],
        func: Callable[[A], Awaitable[B]],
        preserve_output_order: bool = False
    ) -> AsyncIterator[B]:

        return NotImplemented

    def lm_generate(
        self: Self,
        system_prompt: str | LMTextPipe | RawTextPipe,
        user_prompt: str | LMTextPipe | RawTextPipe
    ) -> LMTextPipe[FullText]:

        return NotImplemented


StageT = TypeVar('StageT', bound=Stage)


@final
class AppPipeline(Generic[SerT], metaclass=abc.ABCMeta):
    def start(
        self: Self
    ) -> None:

        pass

    @property
    def output(
        self: Self
    ) -> SerT:

        return NotImplemented


@final
class StageOutput(Generic[SerAT]):
    def build_pipeline(
        self: Self
    ) -> AppPipeline[SerAT]:

        return NotImplemented


@final
class Joined(Generic[SerAT, SerBT], tuple[Awaitable[SerAT], Awaitable[SerBT]], BytesSerializable):
    def to_bytes(
        self: Self
    ) -> bytes:

        return NotImplemented

    @classmethod
    def from_bytes(
        cls: type[Self],
        buf: bytes
    ) -> Self:

        return NotImplemented


@final
class _Processor(Generic[SerAT, SerBT]):
    def __call__(
        self: Self,
        stage_input: StageOutput[SerAT]
    ) -> StageOutput[SerBT]:

        return NotImplemented


@final
class _Generator(Generic[SerT]):
    def __call__(
        self: Self
    ) -> StageOutput[SerT]:

        return NotImplemented


@final
class _BatchedProcessorDecorator:
    @staticmethod
    def __call__(
        func: Callable[[StageT, list[SerAT]], Awaitable[list[SerBT]]]
    ) -> _Processor[SerAT, SerBT]:

        return NotImplemented


@overload
def processor(
    func: Callable[[StageT, SerAT], AsyncIterator[SerBT]],
    *,
    batched_input: Literal[None] = None
) -> _Processor[SerAT, Stream[SerBT]]:

    ...


@overload
def processor(
    func: Callable[[StageT, SerAT], Awaitable[SerBT]],
    *,
    batched_input: Literal[None] = None
) -> _Processor[SerAT, SerBT]:

    ...


@overload
def processor(
    *,
    func: Literal[None] = None,
    batched_input: Literal[True]
) -> _BatchedProcessorDecorator:

    ...


def processor(
    func: (
        Callable[[StageT, SerAT], AsyncIterator[SerBT]] |
        Callable[[StageT, SerAT], Awaitable[SerBT]]
    ) | None = None,
    *,
    batched_input: Literal[True] | None = None
) -> _Processor[SerAT, Stream[SerBT]] | _Processor[SerAT, SerBT] | _BatchedProcessorDecorator:

    return NotImplemented


@overload
def generator(
    func: Callable[[StageT], AsyncIterator[SerT]]
) -> _Generator[Stream[SerT]]:

    ...


@overload
def generator(
    func: Callable[[StageT], Awaitable[SerT]]
) -> _Generator[SerT]:

    ...


def generator(
    func: Callable[[StageT], AsyncIterator[SerT]] | Callable[[StageT], Awaitable[SerT]]
) -> _Generator[Stream[SerT]] | _Generator[SerT]:

    return NotImplemented


def pmap(
    func: Callable[[StageOutput[SerAT]], StageOutput[SerBT]],
    inputs: StageOutput[Stream[SerAT]],
    preserve_output_order: bool = False
) -> StageOutput[Stream[SerBT]]:

    return NotImplemented


def join(
    output_a: StageOutput[SerAT],
    output_b: StageOutput[SerBT],
    preserve_output_order: bool = False
) -> StageOutput[Joined[SerAT, SerBT]]:

    return NotImplemented


def flatten(
    inputs: StageOutput[Stream[Stream[SerT]]],
    preserve_output_order: bool = False
) -> StageOutput[Stream[SerT]]:

    return NotImplemented


@overload
def select(
    joined_input: StageOutput[Joined[SerAT, SerBT]],
    index: Literal[0]
) -> StageOutput[SerAT]:

    return NotImplemented


@overload
def select(
    joined_input: StageOutput[Joined[SerAT, SerBT]],
    index: Literal[1]
) -> StageOutput[SerBT]:

    return NotImplemented


def select(
    joined_input: StageOutput[Joined[SerAT, SerBT]],
    index: Literal[0] | Literal[1]
) -> StageOutput[SerAT] | StageOutput[SerBT]:

    return NotImplemented


def apply(
    func: Callable[[SerAT], SerBT],
    input_data: StageOutput[SerAT]
) -> StageOutput[SerBT]:

    return NotImplemented


def split_text(
    input_text: StageOutput[LMTextPipe],
    split_level: type[SplitLevelT]
) -> StageOutput[Stream[LMTextPipe[SplitLevelT]]]:

    return NotImplemented
