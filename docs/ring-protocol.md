# Command and Completion Ring Protocol

Stage 6 defines the stable shared-memory ABI only. It does not define an
`mmap` owner, an `eventfd` wrapper, or runtime integration. The normative C ABI
is [`inc/wmfs/protocol/ring.h`](../inc/wmfs/protocol/ring.h).

## Version and byte order

Version 1 uses little-endian integers, IEEE 754 binary64 scalar bits, two's
complement signed integers, and UTF-8 text. A participant must reject a host
that cannot provide those representations; records are not byte-swapped.
`WMFS_RING_MAGIC` has the bytes `WMFSRNG1` in increasing address order on a
little-endian host.

The ring header carries an ABI major and minor version. A different major
version is incompatible. A participant may accept a newer minor version only
when all unknown header flags, record flags, kinds, and statuses are zero or
otherwise explicitly supported. Header size, record size, capacity, magic,
and session generation are validated before either counter is accessed.

## Ring header

Each command and completion ring begins with one 4096-byte, 64-byte-aligned
header. `capacity` is the number of following 16384-byte records and must be
nonzero. Producer and consumer are monotonically increasing `uint64_t` sequence
counters; the record slot for sequence `n` is `n % capacity`.

| Offset | Size | Field                                                              |
| -----: | ---: | ------------------------------------------------------------------ |
|      0 |    8 | magic                                                              |
|      8 |    4 | ABI major                                                          |
|     12 |    4 | ABI minor                                                          |
|     16 |    4 | header size                                                        |
|     20 |    4 | record size                                                        |
|     24 |    4 | capacity                                                           |
|     28 |    4 | header flags                                                       |
|     32 |    8 | session generation                                                 |
|     40 |   24 | reserved, zero                                                     |
|     64 |    8 | producer sequence                                                  |
|     72 |   56 | reserved, zero; producer cache line padding                        |
|    128 |    8 | consumer sequence                                                  |
|    136 | 3960 | reserved, zero; consumer cache line padding and future header data |

The producer and consumer values deliberately have plain integer C layout.
They are not C++ `std::atomic` objects and must never be accessed with ordinary
loads or stores while another process can access them.

## Record layout

Command and completion rings use the same 16384-byte, 64-byte-aligned physical
record. Their kind values occupy separate ranges. Counts are bounded by 16
combined input/output tensor descriptors, 16 scalars, and 16 planned outputs.

| Offset | Size | Field                                             |
| -----: | ---: | ------------------------------------------------- |
|      0 |    4 | record kind                                       |
|      4 |    4 | record flags                                      |
|      8 |    8 | session generation                                |
|     16 |    8 | submission ID                                     |
|     24 |    8 | invocation ID                                     |
|     32 |    4 | operation ID                                      |
|     36 |    4 | completion status; zero in commands               |
|     40 |    2 | tensor count                                      |
|     42 |    2 | scalar count                                      |
|     44 |    2 | planned output count                              |
|     46 |   82 | reserved, zero                                    |
|    128 | 4864 | 16 tensor descriptors, 304 bytes each             |
|   4992 | 4480 | 16 scalars, 280 bytes each                        |
|   9472 | 2304 | 16 planned outputs, 144 bytes each                |
|  11776 |   64 | profiling data                                    |
|  11840 | 1104 | error lengths, flags, type[64], and message[1024] |
|  12944 | 3440 | reserved, zero                                    |

A tensor descriptor contains four 64-bit capability/view values (`buffer_id`,
generation, byte offset, byte length), fixed-width dtype/rank/kind/flags and
parameter index fields, then signed 64-bit `shape[16]` and `strides[16]` at
offsets 48 and 176. Tensor kind distinguishes inputs from outputs. Read-only is
the default; writable access requires `WMFS_RING_TENSOR_FLAG_WRITABLE`.

A scalar stores parameter index, kind, flags, an eight-byte value bit pattern,
a text length, and `text[256]`. Boolean, integer, and floating values use
`bits`; text uses exactly `text_length` bytes and need not be NUL terminated.
A planned output stores dtype, flags, rank, output index, and `shape[16]`.

Profiling values use a common monotonic nanosecond clock. Published/dequeued/
started/completion values are timestamps; input-view, output-view, dispatch,
and kernel values are durations. They are valid only when the profile record
flag is set.

Error strings use explicit byte lengths and need not be NUL terminated. A
writer sets a truncation flag when the source exceeds the corresponding fixed
array. Error fields are meaningful only for a non-OK completion status.

## Publication and atomic access

Each ring is single producer, single consumer. To publish, the producer first
acquires the consumer sequence, waits while `producer - consumer == capacity`,
writes the entire selected record, then release-stores `producer + 1`. To
consume, the consumer acquires the producer sequence, reads the selected
record only when the sequences differ, finishes all record reads, then
release-stores `consumer + 1`. Counter wrap uses unsigned 64-bit arithmetic;
the distance must never exceed capacity.

On GCC and Clang, access the plain counters with compiler atomics, for example:

```c
uint64_t observed = __atomic_load_n(&header->producer, __ATOMIC_ACQUIRE);
__atomic_store_n(&header->producer, next, __ATOMIC_RELEASE);
```

Equivalent platform/compiler intrinsics are permitted only when they provide
interprocess acquire/release semantics for naturally aligned 64-bit values.
Do not cast the storage to `std::atomic<uint64_t>` and do not place process-
local pointers, file descriptor numbers, or synchronization objects in it.

## Notification and backpressure

A session owns four process-local descriptors transferred during startup:

1. command-available eventfd, signaled by the command producer;
1. command-space-available eventfd, signaled by the command consumer;
1. completion-available eventfd, signaled by the completion producer;
1. completion-space-available eventfd, signaled by the completion consumer.

After observing an empty/full condition, a waiter must recheck the relevant
counter after arming its wait to avoid a lost wakeup. Eventfd counts are hints;
counters are authoritative. A full ring applies bounded blocking/backpressure
and never overwrites unread records. Unbounded busy-spinning is forbidden.

## Validation and corruption

All shared pages and every record are zero-initialized before use. Writers
zero the complete destination record before populating it. Every reserved
field, unused array entry, inactive payload area, and unknown flag bit must be
zero. Readers validate bounds, kinds, statuses, rank, text/error lengths,
session generation, and reserved bytes before acting on capabilities.

Bad magic/version/size/alignment, impossible sequence distance, generation
mismatch, unknown nonzero flags, nonzero reserved data, malformed descriptors,
or an unexpected kind is fatal session corruption. Stop publishing, mark the
session failed, and tear it down; do not try to resynchronize by skipping a
record. A supported non-OK completion status is a recoverable operation result
and does not by itself terminate the session.
