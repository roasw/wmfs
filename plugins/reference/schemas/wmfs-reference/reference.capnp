@0x8b425a52f499e38a;

using Runtime = import "/wmfs/runtime.capnp";

const pluginMetadata :Runtime.PluginMetadata = (
  name = "reference",
  version = "0.1.0",
  protocolVersion = 8,
  fingerprint = 0xaa78411d1a8057f0,
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

interface ReferencePlugin extends(Runtime.Plugin) {}
