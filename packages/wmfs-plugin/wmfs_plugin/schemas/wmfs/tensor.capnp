@0x852089c475f5debb;

enum DType {
  float32 @0;
  float64 @1;
  int64 @2;
  uint8 @3;
}

struct TensorDescriptor {
  bufferId @0 :UInt64;
  generation @1 :UInt32;
  offset @2 :UInt64;
  byteLength @3 :UInt64;
  dtype @4 :DType;
  shape @5 :List(UInt64);
  strides @6 :List(Int64);
  allocationId @7 :UInt64;
}

struct BufferTransferEntry {
  invocationId @0 :UInt64;
  bufferId @1 :UInt64;
  generation @2 :UInt32;
  byteLength @3 :UInt64;
  writable @4 :Bool = false;
  union {
    map @5 :Void;
    retire @6 :Void;
  }
  arena @7 :Bool = false;
  allocationId @8 :UInt64;
}

struct BufferTransfer {
  transferId @0 :UInt64;
  entries @1 :List(BufferTransferEntry);
}

struct BufferTransferAck {
  transferId @0 :UInt64;
  union {
    accepted @1 :Void;
    error @2 :Text;
  }
}
