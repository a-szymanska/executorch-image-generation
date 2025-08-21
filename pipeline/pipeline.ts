import latentsData from "@/assets/data/latents.json";
import schedulerConfig from "@/assets/data/scheduler_config.json";
import tokensData from "@/assets/data/text_encoder.json";
import uncondTokensData from "@/assets/data/text_encoder_empty.json";
import {
  applyGuidance,
  chunkTensor,
  concatenateTensors,
  divScalar,
  formatTensor,
} from "@/pipeline/tensor_utils";

import {
  ScalarType,
  TensorPtr,
} from "react-native-executorch/src/types/common";
import { Decoder } from "./decoder";
import { Encoder } from "./encoder";
import { convertTensorToImage } from "./image_utils";
import { Scheduler } from "./scheduler";
import { Unet } from "./unet";

const height = 512;
const width = 512;
const numInferenceSteps = 5;
const guidanceScale = 7.5;
const batchSize = 1;

export default async function runPipeline(
  encoder: Encoder,
  unet: Unet,
  decoder: Decoder
): Promise<{ data: Uint8Array; width: number; height: number } | null> {
  try {
    // ---- Text encoder ----
    console.log("Running text encoder...");
    const { input_ids: tokensArray, shape: tokensShape } = tokensData!;
    const encoderOutput = (await encoder.forward(tokensArray, tokensShape))[0];
    console.log(formatTensor(encoderOutput));

    /*
      max_length = text_input.input_ids.shape[-1]
      uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
      text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
      */
    console.log("Running text encoder again...");
    const { input_ids: uncondTokensArray, shape: uncondTokensShape } =
      uncondTokensData!;
    const uncondEmbeddingsTensor = (
      await encoder.forward(uncondTokensArray, uncondTokensShape)
    )[0];
    const textEmbeddingsTensor = concatenateTensors(
      uncondEmbeddingsTensor,
      encoderOutput
    );
    console.log(formatTensor(textEmbeddingsTensor));

    /*
      in_channels = 4
      latents = torch.randn(
          (batch_size, in_channels, height // 8, width // 8),
          generator=generator,
      ).to(device)
      */
    const { latents: latentsArray, shape: latentsShape } = latentsData!;
    let latentsTensor: TensorPtr = {
      dataPtr: Float32Array.from(latentsArray),
      sizes: latentsShape,
      scalarType: ScalarType.FLOAT,
    };

    // const in_channels = 4;
    // const latent_height = Math.floor(height / 8);
    // const latent_width = Math.floor(width / 8);

    // const shape = [batchSize, in_channels, latent_height, latent_width];
    // const latentsTensor = randomNormalTensor(shape);

    console.log("Latents:\n", formatTensor(latentsTensor));

    /*
    scheduler.set_timesteps(num_inference_steps=5)

    for t in scheduler.timesteps:
        print(t)
        latent_model_input = torch.cat([latents] * 2)

        with torch.no_grad():
            timestep = t.to(torch.int64).unsqueeze(0)
            noise_pred = unet.forward(latent_model_input, timestep, text_embeddings)[0]
            # noise_pred = unet.forward(latent_model_input, t, text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
  */
    const scheduler = new Scheduler(schedulerConfig);
    scheduler.set_timesteps(10);
    const timesteps = scheduler.timesteps;
    console.log(typeof timesteps);

    console.log("Running transformer...");
    for (const t of timesteps) {
      console.log(`Timestep: ${t}`);

      const latentsConcatTensor = concatenateTensors(
        latentsTensor,
        latentsTensor
      );
      const noisePredTensor = (
        await unet.forward(latentsConcatTensor, t, textEmbeddingsTensor)
      )[0];
      const [noiseUncondTensor, noiseTextTensor] = chunkTensor(noisePredTensor);
      // console.log("noisePredUncond:", formatTensor(noiseUncondTensor));
      // console.log("noisePredText:", formatTensor(noiseTextTensor));
      const noiseTensor = applyGuidance(
        noiseUncondTensor,
        noiseTextTensor,
        guidanceScale
      );
      // console.log("noiseTensor:", formatTensor(noiseTensor));
      latentsTensor = scheduler.step(noiseTensor, t, latentsTensor).prev_sample;
      console.log("latentsTensor:", formatTensor(latentsTensor));
    }

    /*
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.forward(latents)[0]
    */
    const latentsScaledTensor = divScalar(latentsTensor, 0.18215);
    const resultTensor = (await decoder.forward(latentsScaledTensor))[0];
    console.log("resultTensor:", formatTensor(resultTensor));

    /*
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.show()
    */
    return convertTensorToImage(resultTensor);
  } catch (e: any) {
    console.error("Inference error:", e.message || e);
    return null;
  }
}
