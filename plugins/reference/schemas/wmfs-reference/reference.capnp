@0x8b425a52f499e38a;

using Runtime = import "/wmfs/runtime.capnp";
using Tensor = import "/wmfs/tensor.capnp";

const pluginMetadata :Runtime.PluginMetadata = (
  name = "reference",
  version = "0.1.0",
  protocolVersion = 8,
  fingerprint = 0xe7ba5a83d19c4261,
  operations = [
    (
      name = "matmul",
      tensorInputs = [
        (name = "a"),
        (name = "b"),
      ],
      tensorOutputs = [(name = "result")],
      operationId = 1,
      outputPlans = [
        (
          name = "result",
          known = (
            dimensions = [
              (inputAxis = (input = 0, axis = 0)),
              (inputAxis = (input = 1, axis = 1)),
            ],
            dtype = (input = 0),
          ),
        ),
      ],
      vjp = (
        known = (
          operationId = 4,
          savedInputs = [0, 1],
          outputCotangents = [0],
          inputGradients = [0, 1],
        ),
      ),
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
        (
          name = "fullMatrices",
          kind = boolean,
          required = false,
          default = (boolean = true),
        ),
      ],
      operationId = 2,
      outputPlans = [
        (
          name = "u",
          known = (
            dimensions = [
              (inputAxis = (input = 0, axis = 0)),
              (
                select = (
                  scalarParameter = 0,
                  whenTrue = (inputAxis = (input = 0, axis = 0)),
                  whenFalse = (
                    minimum = [
                      (inputAxis = (input = 0, axis = 0)),
                      (inputAxis = (input = 0, axis = 1)),
                    ],
                  ),
                ),
              ),
            ],
            dtype = (input = 0),
          ),
        ),
        (
          name = "s",
          known = (
            dimensions = [
              (
                minimum = [
                  (inputAxis = (input = 0, axis = 0)),
                  (inputAxis = (input = 0, axis = 1)),
                ],
              ),
            ],
            dtype = (input = 0),
          ),
        ),
        (
          name = "vh",
          known = (
            dimensions = [
              (
                select = (
                  scalarParameter = 0,
                  whenTrue = (inputAxis = (input = 0, axis = 1)),
                  whenFalse = (
                    minimum = [
                      (inputAxis = (input = 0, axis = 0)),
                      (inputAxis = (input = 0, axis = 1)),
                    ],
                  ),
                ),
              ),
              (inputAxis = (input = 0, axis = 1)),
            ],
            dtype = (input = 0),
          ),
        ),
      ],
    ),
    (
      name = "add_scalar",
      tensorInputs = [(name = "a")],
      tensorOutputs = [(name = "result")],
      scalarParameters = [(name = "value", kind = float64)],
      operationId = 3,
      outputPlans = [
        (
          name = "result",
          known = (
            sameShapeAsInput = 0,
            dtype = (
              promoteTensorScalar = (tensorInput = 0, scalarParameter = 0),
            ),
          ),
        ),
      ],
      vjp = (
        known = (
          operationId = 5,
          outputCotangents = [0],
          inputGradients = [0],
        ),
      ),
    ),
    (
      name = "matmul_vjp",
      tensorInputs = [
        (name = "a"),
        (name = "b"),
        (name = "resultCotangent"),
      ],
      tensorOutputs = [
        (name = "aGradient"),
        (name = "bGradient"),
      ],
      operationId = 4,
      outputPlans = [
        (
          name = "aGradient",
          known = (sameShapeAsInput = 0, dtype = (input = 0)),
        ),
        (
          name = "bGradient",
          known = (sameShapeAsInput = 1, dtype = (input = 1)),
        ),
      ],
      internal = true,
    ),
    (
      name = "add_scalar_vjp",
      tensorInputs = [(name = "resultCotangent")],
      tensorOutputs = [(name = "aGradient")],
      operationId = 5,
      outputPlans = [
        (
          name = "aGradient",
          known = (sameShapeAsInput = 0, dtype = (input = 0)),
        ),
      ],
      internal = true,
    ),
  ],
);

interface ReferencePlugin extends(Runtime.Plugin) {
  tensorChecksum @0 (
    invocationId :UInt64,
    tensor :Tensor.TensorDescriptor,
  ) -> (checksum :Float64);
  matmul @1 (
    invocationId :UInt64,
    a :Tensor.TensorDescriptor,
    b :Tensor.TensorDescriptor,
    allocator :Runtime.OutputAllocator,
  ) -> (result :Tensor.TensorDescriptor);
  svd @2 (
    invocationId :UInt64,
    a :Tensor.TensorDescriptor,
    fullMatrices :Bool = true,
    allocator :Runtime.OutputAllocator,
  ) -> (
    u :Tensor.TensorDescriptor,
    s :Tensor.TensorDescriptor,
    vh :Tensor.TensorDescriptor,
  );
  addScalar @3 (
    invocationId :UInt64,
    a :Tensor.TensorDescriptor,
    value :Float64,
    allocator :Runtime.OutputAllocator,
  ) -> (result :Tensor.TensorDescriptor);
}
