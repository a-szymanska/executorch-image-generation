import { ExecutorchModule } from "react-native-executorch";
import { TensorPtr } from "react-native-executorch/src/types/common";
import { TEXT_ENCODER } from "./constants";

export class Encoder {
  module = new ExecutorchModule();
  async load(): Promise<void> {
    await this.module.load(TEXT_ENCODER);
  }

  async forward(inputTensor: TensorPtr): Promise<TensorPtr[]> {
    const outputTensor = await this.module.forward([inputTensor]);
    return outputTensor;
  }
}
