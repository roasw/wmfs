@0x8b425a52f499e38a;

using Runtime = import "/wmfs/runtime.capnp";
using Tensor = import "/wmfs/tensor.capnp";

const pluginMetadata :Runtime.PluginMetadata = (
  name = "reference",
  version = "0.1.0",
  protocolVersion = 1,
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
}
