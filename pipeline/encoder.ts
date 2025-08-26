import { ExecutorchModule } from "react-native-executorch";
import { TensorPtr } from "react-native-executorch/src/types/common";

export class Encoder {
  module = new ExecutorchModule();
  async load(modelSource: string): Promise<void> {
    await this.module.load(modelSource);
  }

  async forward(inputTensor: TensorPtr): Promise<TensorPtr[]> {
    try {
      const outputTensor = await this.module.forward([inputTensor]);
      return outputTensor;
    } catch (e: any) {
      console.error("Encoder error:", e);
      return [];
    }
  }
}
