@0x8b425a52f499e38a;

using Runtime = import "/wmfs/runtime.capnp";
using Tensor = import "/wmfs/tensor.capnp";

const pluginMetadata :Runtime.PluginMetadata = (
  name = "reference",
  version = "0.1.0",
  protocolVersion = 3,
  operations = [
    (
      name = "matmul",
      tensorInputs = [
        (name = "a"),
        (name = "b"),
      ],
      tensorOutputs = [(name = "result")],
    ),
    (
      name = "svd",
      tensorInputs = [(name = "a")],
      tensorOutputs = [
        (name = "u"),
        (name = "s"),
        (name = "vh"),
      ],
      scalarParameters = [
        (name = "fullMatrices", kind = boolean, required = false),
      ],
    ),
    (
      name = "add_scalar",
      tensorInputs = [(name = "a")],
      tensorOutputs = [(name = "result")],
      scalarParameters = [(name = "value", kind = float64)],
    ),
  ],
);

interface ReferencePlugin extends(Runtime.Plugin) {
  tensorChecksum @0 (tensor :Tensor.TensorDescriptor) -> (checksum :Float64);
  matmul @1 (
    a :Tensor.TensorDescriptor,
    b :Tensor.TensorDescriptor,
    allocator :Runtime.OutputAllocator,
  ) -> (result :Tensor.TensorDescriptor);
  svd @2 (
    a :Tensor.TensorDescriptor,
    fullMatrices :Bool = true,
    allocator :Runtime.OutputAllocator,
  ) -> (
    u :Tensor.TensorDescriptor,
    s :Tensor.TensorDescriptor,
    vh :Tensor.TensorDescriptor,
  );
  addScalar @3 (
    a :Tensor.TensorDescriptor,
    value :Float64,
    allocator :Runtime.OutputAllocator,
  ) -> (result :Tensor.TensorDescriptor);
}
