@0xe52414e8e3d32b51;

using Tensor = import "/wmfs/tensor.capnp";

const protocolVersion :UInt16 = 11;

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
  default @3 :ScalarDefault;
}

struct ScalarDefault {
  union {
    none @0 :Void;
    boolean @1 :Bool;
    float64 @2 :Float64;
    int64 @3 :Int64;
    text @4 :Text;
  }
}

struct InputAxis {
  input @0 :UInt16;
  axis @1 :UInt16;
}

struct SelectDimension {
  scalarParameter @0 :UInt16;
  whenTrue @1 :DimensionExpression;
  whenFalse @2 :DimensionExpression;
}

struct DimensionExpression {
  union {
    constant @0 :UInt64;
    inputAxis @1 :InputAxis;
    minimum @2 :List(DimensionExpression);
    select @3 :SelectDimension;
  }
}

struct PromoteTensorScalar {
  tensorInput @0 :UInt16;
  scalarParameter @1 :UInt16;
}

struct DTypeExpression {
  union {
    fixed @0 :Tensor.DType;
    input @1 :UInt16;
    promoteTensorScalar @2 :PromoteTensorScalar;
  }
}

struct KnownOutput {
  union {
    dimensions @0 :List(DimensionExpression);
    sameShapeAsInput @1 :UInt16;
  }
  dtype @2 :DTypeExpression;
}

struct OutputPlan {
  name @0 :Text;
  union {
    dynamic @1 :Void;
    known @2 :KnownOutput;
  }
}

struct VjpMetadata {
  operationId @0 :UInt32;
  savedInputs @1 :List(UInt16);
  savedOutputs @2 :List(UInt16);
  outputCotangents @3 :List(UInt16);
  inputGradients @4 :List(UInt16);
  scalarParameters @5 :List(UInt16);
}

struct VjpPlan {
  union {
    none @0 :Void;
    known @1 :VjpMetadata;
  }
}

struct OperationMetadata {
  name @0 :Text;
  tensorInputs @1 :List(TensorParameter);
  tensorOutputs @2 :List(TensorParameter);
  scalarParameters @3 :List(ScalarParameter);
  operationId @4 :UInt32;
  outputPlans @5 :List(OutputPlan);
  vjp @6 :VjpPlan;
  internal @7 :Bool = false;
}

struct PluginMetadata {
  name @0 :Text;
  version @1 :Text;
  protocolVersion @2 :UInt16;
  operations @3 :List(OperationMetadata);
  fingerprint @4 :UInt64;
}

struct EnvironmentMetadata {
  pythonVersion @0 :Text;
  torchVersion @1 :Text;
  glibcVersion @2 :Text;
  executable @3 :Text;
}

interface OutputAllocator {
  allocate @0 (shape :List(UInt64), dtype :Tensor.DType) ->
      (tensor :Tensor.TensorDescriptor);
}

struct ScalarArgument {
  parameter @0 :UInt16;
  union {
    boolean @1 :Bool;
    float64 @2 :Float64;
    int64 @3 :Int64;
    text @4 :Text;
  }
}

struct KnownInvocation {
  invocationId @0 :UInt64;
  operationId @1 :UInt32;
  inputs @2 :List(Tensor.TensorDescriptor);
  outputs @3 :List(Tensor.TensorDescriptor);
  scalars @4 :List(ScalarArgument);
}

struct OutputPlanningInvocation {
  invocationId @0 :UInt64;
  operationId @1 :UInt32;
  inputs @2 :List(Tensor.TensorDescriptor);
  scalars @3 :List(ScalarArgument);
}

struct PlannedOutput {
  output @0 :UInt16;
  shape @1 :List(UInt64);
  dtype @2 :Tensor.DType;
}

struct WorkerInvocationMetrics {
  inputViewsNs @0 :UInt64;
  outputViewsNs @1 :UInt64;
  dispatchNs @2 :UInt64;
  kernelNs @3 :UInt64;
}

struct OperationError {
  type @0 :Text;
  message @1 :Text;
}

struct InvocationOutcome {
  union {
    success @0 :Void;
    operationError @1 :OperationError;
  }
}

interface Plugin {
  getMetadata @0 () -> (metadata :PluginMetadata);
  ping @1 (nonce :UInt64) -> (nonce :UInt64);
  getEnvironment @2 () -> (environment :EnvironmentMetadata);
  getProtocolVersion @3 () -> (version :UInt16);
  invokeKnown @4 (invocation :KnownInvocation) -> (outcome :InvocationOutcome);
  invokeKnownProfiled @5 (invocation :KnownInvocation) ->
      (outcome :InvocationOutcome, metrics :WorkerInvocationMetrics);
  planOutputs @6 (invocation :OutputPlanningInvocation) ->
      (outcome :InvocationOutcome, outputs :List(PlannedOutput));
}
