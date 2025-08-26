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

const height = 512;
const width = 512;
const numInferenceSteps = 5;
const guidanceScale = 7.5;
const batchSize = 1;

export default async function runPipeline(
  tokenizer: TokenizerModule,
  scheduler: Scheduler,
  encoder: Encoder,
  unet: Unet,
  decoder: Decoder,
  onInferenceStep: (rawImage: RawImage | null) => void
) {
  try {
    // ----------------------------- Tokenizing -----------------------------
    const tokensArray = await tokenizer.encode("<|startoftext|>" + "a castle");
    const tokensTensor = {
      dataPtr: new BigInt64Array(tokensArray.map(BigInt)),
      sizes: [1, tokensArray.length],
      scalarType: ScalarType.LONG,
    };

    const uncondTokensArray = await tokenizer.encode("<|startoftext|>");
    const uncondTokensTensor = {
      dataPtr: new BigInt64Array(uncondTokensArray.map(BigInt)),
      sizes: [1, tokensArray.length],
      scalarType: ScalarType.LONG,
    };

    // ------------------------------ Encoding ------------------------------
    const embeddingsTensor = (await encoder.forward(tokensTensor))[0];

    const uncondEmbeddingsTensor = (
      await encoder.forward(uncondTokensTensor)
    )[0];
    const textEmbeddingsTensor = concatenateTensors(
      uncondEmbeddingsTensor,
      embeddingsTensor
    );

    const inChannels = 4;
    const latent_height = Math.floor(height / 8);
    const latent_width = Math.floor(width / 8);

    const shape = [batchSize, inChannels, latent_height, latent_width];
    let latentsTensor = randomNormalTensor(shape);

    scheduler.set_timesteps(numInferenceSteps);
    const timesteps = scheduler.timesteps;

    // ------------------------------ Denoising -----------------------------
    for (let i = 0; i < timesteps.length; i++) {
      const t = timesteps[i];
      console.log(`Step ${i + 1}/${timesteps.length} (t=${t})`);

      const latentsConcatTensor = concatenateTensors(
        latentsTensor,
        latentsTensor
      );
      const noisePredTensor = (
        await unet.forward(latentsConcatTensor, t, textEmbeddingsTensor)
      )[0];
      const [noiseUncondTensor, noiseTextTensor] = chunkTensor(noisePredTensor);
      const noiseTensor = applyGuidance(
        noiseUncondTensor,
        noiseTextTensor,
        guidanceScale
      );
      latentsTensor = scheduler.step(noiseTensor, t, latentsTensor).prev_sample;

      // --------------------- For display of intermediate ---------------------
      const latentsScaledTensor = divScalar(latentsTensor, 0.18215);
      const resultTensor = (await decoder.forward(latentsScaledTensor))[0];

      const rawImage = convertTensorToImage(resultTensor);
      onInferenceStep(rawImage);
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
