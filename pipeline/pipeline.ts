import {
  applyGuidance,
  chunkTensor,
  concatenateTensors,
  divScalar,
  formatTensor,
  randomNormalTensor,
} from "@/pipeline/tensor_utils";
import { Decoder } from "./decoder";
import { Encoder } from "./encoder";
import { convertTensorToImage, RawImage } from "./image_utils";
import { Scheduler } from "./scheduler";
import { Unet } from "./unet";
import { ScalarType } from "react-native-executorch/src/types/common";
import { TokenizerModule } from "react-native-executorch";
import { ModelSource } from "../constants/Model";

const GUIDANCE_SCALE = 7.5;
const BATCH_SIZE = 1;

export class Pipeline {
  height: number;
  width: number;

  tokenizer: TokenizerModule;
  scheduler: Scheduler;
  encoder: Encoder;
  unet: Unet;
  decoder: Decoder;
  onInferenceStep: (rawImage: RawImage | null) => void;

  constructor(onInferenceStep: (rawImage: RawImage | null) => void) {
    this.tokenizer = new TokenizerModule();
    this.scheduler = new Scheduler();
    this.encoder = new Encoder();
    this.unet = new Unet();
    this.decoder = new Decoder();

    this.onInferenceStep = onInferenceStep;

    this.height = 512;
    this.width = 512;
  }

  async load(model: ModelSource): Promise<void> {
    try {
      await this.tokenizer.load(model.tokenizer);
      await this.scheduler.load(model.schedulerSource);
      await this.encoder.load(model.encoderSource);
      await this.unet.load(model.unetSource);
      await this.decoder.load(model.decoderSource);
    } catch (e: any) {
      throw Error("Failed to load the pipeline components:", e);
    }
  }

  async run(prompt: string, numInferenceSteps: number = 5): Promise<void> {
    try {
      // ----------------------------- Tokenizing -----------------------------
      const tokensArray = await this.tokenizer.encode(
        "<|startoftext|>" + prompt
      );
      const tokensTensor = {
        dataPtr: new BigInt64Array(tokensArray.map(BigInt)),
        sizes: [1, tokensArray.length],
        scalarType: ScalarType.LONG,
      };

      const uncondTokensArray = await this.tokenizer.encode("<|startoftext|>");
      const uncondTokensTensor = {
        dataPtr: new BigInt64Array(uncondTokensArray.map(BigInt)),
        sizes: [1, tokensArray.length],
        scalarType: ScalarType.LONG,
      };

      // ------------------------------ Encoding ------------------------------
      const embeddingsTensor = (await this.encoder.forward(tokensTensor))[0];

      const uncondEmbeddingsTensor = (
        await this.encoder.forward(uncondTokensTensor)
      )[0];
      const textEmbeddingsTensor = concatenateTensors(
        uncondEmbeddingsTensor,
        embeddingsTensor
      );

      const inChannels = 4;
      const latent_height = Math.floor(this.height / 8);
      const latent_width = Math.floor(this.width / 8);

      const shape = [BATCH_SIZE, inChannels, latent_height, latent_width];
      let latentsTensor = randomNormalTensor(shape);

      this.scheduler.set_timesteps(numInferenceSteps);
      const timesteps = this.scheduler.timesteps;

      // ------------------------------ Denoising -----------------------------
      for (let i = 0; i < timesteps.length; i++) {
        const t = timesteps[i];
        console.log(`Step ${i + 1}/${timesteps.length} (t=${t})`);

        const latentsConcatTensor = concatenateTensors(
          latentsTensor,
          latentsTensor
        );
        const noisePredTensor = (
          await this.unet.forward(latentsConcatTensor, t, textEmbeddingsTensor)
        )[0];
        const [noiseUncondTensor, noiseTextTensor] =
          chunkTensor(noisePredTensor);
        const noiseTensor = applyGuidance(
          noiseUncondTensor,
          noiseTextTensor,
          GUIDANCE_SCALE
        );
        latentsTensor = this.scheduler.step(
          noiseTensor,
          t,
          latentsTensor
        ).prev_sample;

        // --------------------- For display of intermediate ---------------------
        const latentsScaledTensor = divScalar(latentsTensor, 0.18215);
        const resultTensor = (
          await this.decoder.forward(latentsScaledTensor)
        )[0];

        const rawImage = convertTensorToImage(resultTensor);
        this.onInferenceStep(rawImage);
        // ----------------------------------------------------------------------
      }
      // ------------------------------ Decoding ------------------------------
      // const latentsScaledTensor = divScalar(latentsTensor, 0.18215);
      // const resultTensor = (await decoder.forward(latentsScaledTensor))[0];

      // const rawImage = convertTensorToImage(resultTensor);
      // onInferenceStep(rawImage);
    } catch (e: any) {
      console.error("Inference error:", e.message || e);
    }
  }
}
