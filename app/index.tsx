/* eslint-disable react-hooks/exhaustive-deps */
import { Decoder } from "@/pipeline/decoder";
import { Encoder } from "@/pipeline/encoder";
import { getBase64FromImage, RawImage } from "@/pipeline/image_utils";
import runPipeline from "@/pipeline/pipeline";
import { Unet } from "@/pipeline/unet";
import React, { useEffect, useState } from "react";
import { Button, Image, ScrollView, StyleSheet, Text } from "react-native";
import { Scheduler } from "@/pipeline/scheduler";
import {
  SCHEDULER,
  TEXT_ENCODER,
  TOKENIZER,
  UNET,
  VAE,
} from "@/pipeline/model_paths";
import { TokenizerModule } from "react-native-executorch";

export default function App() {
  const [pipelineRunning, setPipelineRunning] = useState<boolean>(false);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const tokenizer = new TokenizerModule();
  const [scheduler] = useState(() => new Scheduler());
  const [encoder] = useState(() => new Encoder());
  const [unet] = useState(() => new Unet());
  const [decoder] = useState(() => new Decoder());
  let loadingError: any = null;

  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log("Loading models...");
        await tokenizer.load(TOKENIZER);
        await scheduler.load(SCHEDULER);
        await encoder.load(TEXT_ENCODER);
        await unet.load(UNET);
        await decoder.load(VAE);
        console.log("Models loaded!");
      } catch (e: any) {
        console.error("Failed to load models:", e);
        loadingError = e;
      }
    };

    loadModels();
  }, [encoder]);

  const generate = async () => {
    try {
      setPipelineRunning(true);
      await runPipeline(
        tokenizer,
        scheduler,
        encoder,
        unet,
        decoder,
        (rawImage: RawImage | null) => {
          setImageUri(getBase64FromImage(rawImage));
        }
      );
      console.log("Image generated!");
    } catch (e: any) {
      console.error("Generating error:", e.message || e);
    } finally {
      setPipelineRunning(false);
    }
  };
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Stable Diffusion Pipeline</Text>

      <Button
        title={"Run Pipeline"}
        onPress={generate}
        disabled={pipelineRunning || !!loadingError}
      />
      {imageUri && (
        <Image
          style={styles.image}
          source={{ uri: `data:image/png;base64,${imageUri}` }}
        />
      )}
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
  image: {
    width: 320,
    height: 320,
    marginVertical: 50,
    resizeMode: "contain",
  },
});
