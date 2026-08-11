@0xe52414e8e3d32b51;

const protocolVersion :UInt16 = 1;

enum TensorAccess {
  readOnly @0;
  readWrite @1;
}

enum ScalarKind {
  boolean @0;
  float64 @1;
  int64 @2;
  text @3;
}

struct TensorParameter {
  name @0 :Text;
  access @1 :TensorAccess = readOnly;
}

struct ScalarParameter {
  name @0 :Text;
  kind @1 :ScalarKind;
  required @2 :Bool = true;
}

struct OperationMetadata {
  name @0 :Text;
  tensorInputs @1 :List(TensorParameter);
  tensorOutputs @2 :List(TensorParameter);
  scalarParameters @3 :List(ScalarParameter);
}

struct PluginMetadata {
  name @0 :Text;
  version @1 :Text;
  protocolVersion @2 :UInt16;
  operations @3 :List(OperationMetadata);
}

interface Plugin {
  getMetadata @0 () -> (metadata :PluginMetadata);
  ping @1 (nonce :UInt64) -> (nonce :UInt64);
}
