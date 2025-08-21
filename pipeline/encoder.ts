import { ExecutorchModule } from "react-native-executorch";
import {
  ScalarType,
  TensorPtr,
} from "react-native-executorch/src/types/common";
import { TEXT_ENCODER } from "./constants";

export class Encoder {
  module = new ExecutorchModule();
  async load(): Promise<void> {
    await this.module.load(TEXT_ENCODER);
  }

  async forward(inputArray: number[], shape: number[]): Promise<TensorPtr[]> {
    const intputTensor = {
      dataPtr: BigInt64Array.from(inputArray.map(BigInt)),
      sizes: shape,
      scalarType: ScalarType.LONG,
    };
    const outputTensor = await this.module.forward([intputTensor]);
    return outputTensor;
  }
}
