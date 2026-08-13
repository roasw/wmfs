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
}

struct BufferTransfer {
  transferId @0 :UInt64;
  invocationId @1 :UInt64;
  bufferId @2 :UInt64;
  generation @3 :UInt32;
  byteLength @4 :UInt64;
  writable @5 :Bool = false;
}

struct BufferTransferAck {
  transferId @0 :UInt64;
  union {
    accepted @1 :Void;
    error @2 :Text;
  }
}
