import { concatenateTensors } from "@/pipeline/tensor_utils";
import React, { useEffect, useState } from "react";
import { Button, ScrollView, StyleSheet, Text } from "react-native";
import { ExecutorchModule } from "react-native-executorch";
import {
  ScalarType,
  TensorPtr,
} from "react-native-executorch/src/types/common";

let encoderInputData: {
  input_ids: number[];
  shape: number[];
} | null = null;
let encoderUncondInputData: {
  input_ids: number[];
  attention_mask: number[];
  shape: number[];
} | null = null;
let latentsInputData: {
  latents: number[];
  shape: number[];
} | null = null;
let unetInputData: {
  latent_model_input: { shape: number[]; values: number[] };
  timestep: { shape: number[]; values: number[] };
  text_embeddings: { shape: number[]; values: number[] };
} | null = null;
let vaeInputData: {
  latents: { shape: number[]; values: number[] };
} | null = null;
let inputLoaded: boolean = false;

try {
  encoderInputData = require("@/assets/data/text_encoder.json");
  encoderUncondInputData = require("@/assets/data/text_encoder_empty.json");
  latentsInputData = require("@/assets/data/latents.json");
  unetInputData = require("@/assets/data/unet.json");
  vaeInputData = require("@/assets/data/vae.json");
  inputLoaded = true;
} catch (e: any) {
  console.error("Failed to load model assets:", e.message || e);
}

const height = 512;
const width = 512;
const numInferenceSteps = 5;
const guidanceScale = 7.5;
const batchSize = 1;

const formatOutput = (tensor?: TensorPtr) => {
  if (!tensor) {
    return null;
  }
  const shape = tensor.sizes;
  const data = tensor.dataPtr;
  const view = new Float32Array(data);
  return `Shape: ${shape}\n Data: [${Array.from(view.slice(0, 5))
    .map((n) => n.toFixed(4))
    .join(", ")} ... ${Array.from(view.slice(-5))
    .map((n) => n.toFixed(4))
    .join(", ")}]`;
};

export default function HomeScreen() {
  const [modelsRunning, setModelsRunning] = useState<boolean>(false);
  const encoder = React.useMemo(() => new ExecutorchModule(), []);
  const unet = React.useMemo(() => new ExecutorchModule(), []);
  const vae = React.useMemo(() => new ExecutorchModule(), []);
  let loadingError: any = null;

  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log("Loading models...");
        await encoder.load(
          "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/text_encoder_model.pte"
        );
        await unet.load(
          "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/unet_model.pte"
        );
        await vae.load(
          "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/vae_model.pte"
        );

        console.log("Models loaded successfully");
      } catch (e: any) {
        console.error("Failed to load models:", e);
        loadingError = e;
      }
    };

    loadModels();
  }, [encoder, vae]);

  const runPipeline = async () => {
    try {
      setModelsRunning(true);
      // ---- Text encoder ----
      console.log("Running text encoder...");
      const { input_ids: inputIdsArray, shape } = encoderInputData!;
      const encoderInputTensor = {
        dataPtr: BigInt64Array.from(inputIdsArray.map(BigInt)),
        sizes: shape,
        scalarType: ScalarType.LONG,
      };
      const encoderOutput = (await encoder.forward([encoderInputTensor]))[0];
      console.log(formatOutput(encoderOutput));

      /*
      max_length = text_input.input_ids.shape[-1]
      uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
      text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
      */

      const { input_ids: uncondInputIdsArray, shape: uncondShape } =
        encoderUncondInputData!;
      const uncondInputTensor = {
        dataPtr: BigInt64Array.from(uncondInputIdsArray.map(BigInt)),
        sizes: uncondShape,
        scalarType: ScalarType.LONG,
      };
      console.log("Running text encoder again...");
      const uncondEmbeddings = (await encoder.forward([uncondInputTensor]))[0];
      console.log(formatOutput(uncondEmbeddings));
      const textEmbeddings = concatenateTensors(
        uncondEmbeddings,
        encoderOutput
      );
      console.log(formatOutput(textEmbeddings));

      /*
      in_channels = 4
      latents = torch.randn(
          (batch_size, in_channels, height // 8, width // 8),
          generator=generator,
      ).to(device)
      latents = latents * scheduler.init_noise_sigma
      */
      const inChannels = 4;
      const { latents: latentsArray, shape: latentsShape } = latentsInputData!;

      // -------- UNet --------
      // console.log("Running Unet...");
      // const {
      //   latent_model_input: latentModelInput,
      //   timestep,
      //   text_embeddings: textEmbeddings,
      // } = unetInputData!;

      // const latentModelInputTensor = {
      //   dataPtr: Float32Array.from(latentModelInput.values),
      //   sizes: latentModelInput.shape,
      //   scalarType: 6,
      // };

      // const timestepTensor = {
      //   dataPtr: BigInt64Array.from(timestep.values.map(BigInt)),
      //   sizes: timestep.shape,
      //   scalarType: 4,
      // };

      // const textEmbeddingsTensor = {
      //   dataPtr: Float32Array.from(textEmbeddings.values),
      //   sizes: textEmbeddings.shape,
      //   scalarType: 6,
      // };
      // const unetInputArray = [
      //   latentModelInputTensor,
      //   timestepTensor,
      //   textEmbeddingsTensor,
      // ];
      // const unetOutput = (await unet.forward(unetInputArray))[0];
      // console.log("Output:\n", formatOutput(unetOutput));

      // ---- Vae decoder ----
      // console.log("Running VAE decoder...");
      // const { latents } = vaeInputData!;
      // const vaeInputTensor = {
      //   dataPtr: Float32Array.from(latents.values),
      //   sizes: latents.shape,
      //   scalarType: 6,
      // };
      // const vaeOutput = (await vae.forward([vaeInputTensor]))[0];
      // console.log("Output:\n", formatOutput(vaeOutput));

      console.log("Pipeline complete!");
    } catch (e: any) {
      console.error("Inference error:", e.message || e);
    } finally {
      setModelsRunning(false);
    }
  };
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Stable Diffusion Pipeline</Text>

      <Button
        title={"Run Pipeline"}
        onPress={runPipeline}
        disabled={!inputLoaded || modelsRunning || !!loadingError}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f5f5f5",
    paddingVertical: 30,
    paddingHorizontal: 20,
    marginTop: 50,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
    color: "#333",
  },
  statusContainer: {
    flexDirection: "column",
    alignSelf: "flex-start",
    alignItems: "flex-start",
    justifyContent: "center",
    padding: 20,
  },
  text: {
    fontSize: 16,
    color: "#666",
  },
});
