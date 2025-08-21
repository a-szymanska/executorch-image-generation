import { ExecutorchModule } from "react-native-executorch";
import {
  ScalarType,
  TensorPtr,
} from "react-native-executorch/src/types/common";
import { UNET } from "./constants";

export class Unet {
  module = new ExecutorchModule();
  async load(): Promise<void> {
    await this.module.load(UNET);
  }

  async forward(
    latentsTensor: TensorPtr,
    timestep: number,
    embeddingsTensor: TensorPtr
  ): Promise<TensorPtr[]> {
    const timestepTensor = {
      dataPtr: BigInt64Array.from([timestep].map(BigInt)),
      sizes: [1],
      scalarType: ScalarType.LONG,
    };
    const intputTensor = [latentsTensor, timestepTensor, embeddingsTensor];
    const outputTensor = await this.module.forward(intputTensor);
    return outputTensor;
  }
}
