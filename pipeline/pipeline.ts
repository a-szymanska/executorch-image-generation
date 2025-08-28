import {
  applyGuidance,
  chunkTensor,
  concatenateTensors,
  divScalar,
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

  model: ModelSource | null;
  tokenizer: TokenizerModule;
  scheduler: Scheduler;
  encoder: Encoder;
  unet: Unet;
  decoder: Decoder;
  onInferenceStep: ((rawImage: RawImage | null) => void) | null;

  constructor() {
    this.model = null;
    this.tokenizer = new TokenizerModule();
    this.scheduler = new Scheduler();
    this.encoder = new Encoder();
    this.unet = new Unet();
    this.decoder = new Decoder();

    this.onInferenceStep = null;

    this.height = 512;
    this.width = 512;
  }

  async load(model: ModelSource): Promise<void> {
    this.model = model;
    this.height = this.width = model.size;
  }

  async run(
    prompt: string,
    numInferenceSteps: number = 10,
    onInferenceStep: ((rawImage: RawImage | null) => void) | null = null
  ): Promise<RawImage | null> {
    this.onInferenceStep = onInferenceStep;
    try {
      // ----------------------------- Tokenizing -----------------------------
      await this.tokenizer.load(this.model!.tokenizer);
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
      await this.encoder.load(this.model!.encoderSource);
      await this.scheduler.load(this.model!.schedulerSource);

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

      this.encoder.delete();
      // ------------------------------ Denoising -----------------------------
      await this.unet.load(this.model!.unetSource);
      if (onInferenceStep) {
        await this.decoder.load(this.model!.decoderSource);
      }

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
        if (this.onInferenceStep) {
          const latentsScaledTensor = divScalar(latentsTensor, 0.18215);
          const resultTensor = (
            await this.decoder.forward(latentsScaledTensor)
          )[0];

          const rawImage = convertTensorToImage(resultTensor);
          this.onInferenceStep(rawImage);
        }
      }
      this.unet.delete();

      // ------------------------------ Decoding ------------------------------
      if (!onInferenceStep) {
        await this.decoder.load(this.model!.decoderSource);
      }
      const latentsScaledTensor = divScalar(latentsTensor, 0.18215);
      const resultTensor = (await this.decoder.forward(latentsScaledTensor))[0];

      const rawImage = convertTensorToImage(resultTensor);
      return rawImage;
    } catch (e: any) {
      console.error("Inference error:", e.message || e);
      return null;
    }
  }
}
