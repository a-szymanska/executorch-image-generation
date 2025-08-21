import { ExecutorchModule } from "react-native-executorch";
import { TensorPtr } from "react-native-executorch/src/types/common";
import { VAE } from "./constants";

export class Decoder {
  module = new ExecutorchModule();
  async load(): Promise<void> {
    await this.module.load(VAE);
  }

  async forward(inputTensor: TensorPtr): Promise<TensorPtr[]> {
    const outputTensor = await this.module.forward([inputTensor]);
    return outputTensor;
  }
}
