import { ExecutorchModule } from "react-native-executorch";
import { TensorPtr } from "react-native-executorch/src/types/common";

export class Decoder {
  module = new ExecutorchModule();
  async load(modelSource: string): Promise<void> {
    await this.module.load(modelSource);
  }

  async forward(inputTensor: TensorPtr): Promise<TensorPtr[]> {
    const outputTensor = await this.module.forward([inputTensor]);
    return outputTensor;
  }
}
