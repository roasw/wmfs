@0x8b425a52f499e38a;

using Runtime = import "/wmfs/runtime.capnp";

const pluginMetadata :Runtime.PluginMetadata = (
  name = "reference",
  version = "0.1.0",
  protocolVersion = 11,
  fingerprint = 0xf6ed5672a8a496cb,
  metadataVersion = 2,
  operations = [
    (
      name = "matmul",
      tensorInputs = [
        (name = "a", dtypeVariable = "T"),
        (name = "b", dtypeVariable = "T"),
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
            dtype = (variable = "T"),
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
      dtypeVariables = [
        (name = "T", dtypes = [float32, float64, int64, uint8]),
      ],
    ),
    (
      name = "svd",
      tensorInputs = [(name = "a", dtypeVariable = "T")],
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
            dtype = (variable = "T"),
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
            dtype = (variable = "T"),
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
            dtype = (variable = "T"),
          ),
        ),
      ],
      dtypeVariables = [(name = "T", dtypes = [float32, float64])],
    ),
    (
      name = "add_scalar",
      tensorInputs = [(name = "a", dtypeVariable = "T")],
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
      dtypeVariables = [
        (name = "T", dtypes = [float32, float64, int64, uint8]),
      ],
    ),
    (
      name = "matmul_vjp",
      tensorInputs = [
        (name = "a", dtypeVariable = "T"),
        (name = "b", dtypeVariable = "T"),
        (name = "resultCotangent", dtypeVariable = "T"),
      ],
      tensorOutputs = [
        (name = "aGradient"),
        (name = "bGradient"),
      ],
      operationId = 4,
      outputPlans = [
        (
          name = "aGradient",
            known = (sameShapeAsInput = 0, dtype = (variable = "T")),
        ),
        (
          name = "bGradient",
            known = (sameShapeAsInput = 1, dtype = (variable = "T")),
        ),
      ],
      internal = true,
      dtypeVariables = [
        (name = "T", dtypes = [float32, float64, int64, uint8]),
      ],
    ),
    (
      name = "add_scalar_vjp",
      tensorInputs = [(name = "resultCotangent", dtypeVariable = "T")],
      tensorOutputs = [(name = "aGradient")],
      operationId = 5,
      outputPlans = [
        (
          name = "aGradient",
            known = (sameShapeAsInput = 0, dtype = (variable = "T")),
        ),
      ],
      internal = true,
      dtypeVariables = [
        (name = "T", dtypes = [float32, float64, int64, uint8]),
      ],
    ),
    (
      name = "nonzero",
      tensorInputs = [
        (name = "a", dtypes = [float32, float64, int64, uint8]),
      ],
      tensorOutputs = [(name = "indices")],
      scalarParameters = [
        (
          name = "order",
          kind = int64,
          required = false,
          default = (text = "rowMajor"),
          enumName = "IndexOrder",
          enumValues = ["rowMajor", "columnMajor"],
        ),
      ],
      operationId = 6,
      outputPlans = [(name = "indices", dynamic = void)],
    ),
  ],
);

interface ReferencePlugin extends(Runtime.Plugin) {}
