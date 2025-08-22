import { Decoder } from "@/pipeline/decoder";
import { Encoder } from "@/pipeline/encoder";
import { getBase64FromImage, RawImage } from "@/pipeline/image_utils";
import runPipeline from "@/pipeline/pipeline";
import { Unet } from "@/pipeline/unet";
import React, { useEffect, useState } from "react";
import { Button, Image, ScrollView, StyleSheet, Text } from "react-native";
import { Scheduler } from "@/pipeline/scheduler";
import { SCHEDULER, TEXT_ENCODER, UNET, VAE } from "@/pipeline/model_paths";

export default function App() {
  const [pipelineRunning, setPipelineRunning] = useState<boolean>(false);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const scheduler = React.useMemo(() => new Scheduler(), []);
  const encoder = React.useMemo(() => new Encoder(), []);
  const unet = React.useMemo(() => new Unet(), []);
  const decoder = React.useMemo(() => new Decoder(), []);
  let loadingError: any = null;

  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log("Loading models...");
        await scheduler.load(SCHEDULER);
        await encoder.load(TEXT_ENCODER);
        await unet.load(UNET);
        await decoder.load(VAE);

        console.log("Models loaded successfully");
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
  image: {
    width: 512,
    height: 512,
    resizeMode: "contain",
  },
});
