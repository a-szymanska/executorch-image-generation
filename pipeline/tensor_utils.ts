import {
  ScalarType,
  TensorPtr,
} from "react-native-executorch/src/types/common";

export function formatTensor(tensor?: TensorPtr): string {
  if (!tensor) {
    return `Shape: [0]\n Data: []`;
  }
  const shape = tensor.sizes;
  const data = tensor.dataPtr;
  const view = new Float32Array(
    data instanceof ArrayBuffer ? data : data.buffer ?? data
  );
  return `Shape: ${shape}\n Data: [${Array.from(view.slice(0, 5))
    .map((n) => n.toFixed(4))
    .join(", ")} ... ${Array.from(view.slice(-5))
    .map((n) => n.toFixed(4))
    .join(", ")}]`;
}

export function concatenateTensors(
  tensorA: TensorPtr,
  tensorB: TensorPtr
): TensorPtr {
  if (
    tensorA.scalarType !== tensorB.scalarType ||
    tensorA.sizes.slice(1).toString() !== tensorB.sizes.slice(1).toString()
  ) {
    throw new Error(
      "Tensors must have the same shape and scalar type for concatenation."
    );
  }
  const data1 = new Float32Array(tensorA.dataPtr);
  const data2 = new Float32Array(tensorB.dataPtr);

  const concatenated = new Float32Array(data1.length + data2.length);
  concatenated.set(data1, 0);
  concatenated.set(data2, data1.length);
  return {
    dataPtr: concatenated,
    sizes: [2, ...tensorA.sizes.slice(1)],
    scalarType: tensorA.scalarType,
  };
}

export function chunkTensor(tensor: TensorPtr): TensorPtr[] {
  const { dataPtr, sizes, scalarType } = tensor;

  const data =
    dataPtr instanceof ArrayBuffer ? dataPtr : dataPtr.buffer ?? dataPtr;
  let dataArray = new Float32Array(data);

  const numChunks = sizes[0];
  const chunkShape = [1, ...sizes.slice(1)];
  const chunkSize = dataArray.length / numChunks;

  const result: TensorPtr[] = [];

  for (let i = 0; i < numChunks; i++) {
    const start = i * chunkSize;
    const end = start + chunkSize;
    const chunkData = dataArray.slice(start, end);

    result.push({
      dataPtr: chunkData.buffer,
      sizes: chunkShape,
      scalarType: scalarType,
    });
  }

  return result;
}

function elementwiseOperation(
  tensorA: TensorPtr,
  tensorB: TensorPtr,
  operation: (a: number, b: number) => number
): TensorPtr {
  if (tensorA.sizes.toString() !== tensorB.sizes.toString()) {
    throw new Error(
      "Tensors must have the same shape for element-wise operations."
    );
  }

  const dataA = new Float32Array(tensorA.dataPtr);
  const dataB = new Float32Array(tensorB.dataPtr);
  const n = dataA.length;

  const dataRes = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dataRes[i] = operation(dataA[i], dataB[i]);
  }

  return {
    dataPtr: dataRes.buffer,
    sizes: tensorA.sizes,
    scalarType: tensorA.scalarType,
  };
}

export function addTensors(tensorA: TensorPtr, tensorB: TensorPtr): TensorPtr {
  return elementwiseOperation(tensorA, tensorB, (a, b) => a + b);
}

export function subTensors(tensorA: TensorPtr, tensorB: TensorPtr): TensorPtr {
  return elementwiseOperation(tensorA, tensorB, (a, b) => a - b);
}

function tensorScalarOperation(
  tensor: TensorPtr,
  operation: (a: number) => number
): TensorPtr {
  const data = new Float32Array(tensor.dataPtr);
  const n = data.length;
  const dataRes = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    dataRes[i] = operation(data[i]);
  }

  return {
    dataPtr: dataRes.buffer,
    sizes: [...tensor.sizes],
    scalarType: tensor.scalarType,
  };
}

export function mulScalar(tensor: TensorPtr, scalar: number): TensorPtr {
  return tensorScalarOperation(tensor, (a) => a * scalar);
}

export function divScalar(tensor: TensorPtr, scalar: number): TensorPtr {
  return tensorScalarOperation(tensor, (a) => a / scalar);
}

export function addScalar(tensor: TensorPtr, scalar: number): TensorPtr {
  return tensorScalarOperation(tensor, (a) => a + scalar);
}

export function subScalarTensor(tensor: TensorPtr, scalar: number): TensorPtr {
  return tensorScalarOperation(tensor, (a) => scalar - a);
}

export function subTensorScalar(tensor: TensorPtr, scalar: number): TensorPtr {
  return tensorScalarOperation(tensor, (a) => a - scalar);
}

export function applyGuidance(
  noiseUncondTensor: TensorPtr,
  noiseTextTensor: TensorPtr,
  guidanceScale: number
): TensorPtr {
  const diff = subTensors(noiseTextTensor, noiseUncondTensor);
  const scaledDiff = mulScalar(diff, guidanceScale);
  const noiseTensor = addTensors(noiseUncondTensor, scaledDiff);

  return noiseTensor;
}

export function randomNormalTensor(shape: number[]): TensorPtr {
  const size = shape.reduce((acc, val) => acc * val, 1);
  const data = new Float32Array(size);
  for (let i = 0; i < size; i += 2) {
    const u1 = Math.random() || Number.EPSILON;
    const u2 = Math.random();

    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const z1 = mag * Math.cos(2.0 * Math.PI * u2);
    const z2 = mag * Math.sin(2.0 * Math.PI * u2);

    data[i] = z1;
    if (i + 1 < size) {
      data[i + 1] = z2;
    }
  }
  return {
    dataPtr: data,
    sizes: shape,
    scalarType: ScalarType.FLOAT,
  };
}

export function clamp(tensor: TensorPtr, min: number, max: number): TensorPtr {
  return tensorScalarOperation(tensor, (a) => Math.max(min, Math.min(max, a)));
}

export function squeeze(tensor: TensorPtr): TensorPtr {
  return {
    ...tensor,
    sizes: tensor.sizes.filter((size) => size !== 1),
    scalarType: tensor.scalarType,
  };
}

export function permute120(tensor: TensorPtr): TensorPtr {
  const [X, Y, Z] = tensor.sizes;
  const oldData = new Float32Array(tensor.dataPtr);
  const newData = new Float32Array(X * Y * Z);

  for (let x = 0; x < X; x++) {
    for (let y = 0; y < Y; y++) {
      for (let z = 0; z < Z; z++) {
        const oldIndex = x * Y * Z + y * Z + z;
        const newIndex = y * Z * X + z * X + x;
        newData[newIndex] = oldData[oldIndex];
      }
    }
  }

  return {
    dataPtr: newData.buffer,
    sizes: [Y, Z, X],
    scalarType: tensor.scalarType,
  };
}
