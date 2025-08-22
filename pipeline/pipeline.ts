import latentsData from "@/assets/data/latents.json";
import schedulerConfig from "@/assets/data/scheduler_config.json";
import tokensData from "@/assets/data/text_encoder.json";
import uncondTokensData from "@/assets/data/text_encoder_empty.json";
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

const height = 512;
const width = 512;
const numInferenceSteps = 5;
const guidanceScale = 7.5;
const batchSize = 1;

export default async function runPipeline(
  encoder: Encoder,
  unet: Unet,
  decoder: Decoder,
  onInferenceStep: (rawImage: RawImage | null) => void
) {
  try {
    // ------------------------------ Encoding ------------------------------
    const { input_ids: tokensArray, shape: tokensShape } = tokensData!;
    const tokensTensor = {
      dataPtr: BigInt64Array.from(tokensArray.map(BigInt)),
      sizes: tokensShape,
      scalarType: ScalarType.LONG,
    };
    const encoderOutput = (await encoder.forward(tokensTensor))[0];

    const { input_ids: uncondTokensArray, shape: uncondTokensShape } =
      uncondTokensData!;
    const uncondTokensTensor = {
      dataPtr: BigInt64Array.from(uncondTokensArray.map(BigInt)),
      sizes: uncondTokensShape,
      scalarType: ScalarType.LONG,
    };
    const uncondEmbeddingsTensor = (
      await encoder.forward(uncondTokensTensor)
    )[0];
    const textEmbeddingsTensor = concatenateTensors(
      uncondEmbeddingsTensor,
      encoderOutput
    );

    const in_channels = 4;
    const latent_height = Math.floor(height / 8);
    const latent_width = Math.floor(width / 8);

    const shape = [batchSize, in_channels, latent_height, latent_width];
    let latentsTensor = randomNormalTensor(shape);

    const scheduler = new Scheduler(schedulerConfig);
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
