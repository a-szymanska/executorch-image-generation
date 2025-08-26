/* eslint-disable react-hooks/exhaustive-deps */
import { getBase64FromImage, RawImage } from "@/pipeline/image_utils";
import { BK_SDM_TINY_VPRED } from "@/pipeline/model_paths";
import { Pipeline } from "@/pipeline/pipeline";
import React, { useEffect, useState } from "react";
import {
  Button,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
} from "react-native";

export default function App() {
  const [pipelineLoading, setPipelineLoading] = useState<boolean>(false);
  const [pipelineRunning, setPipelineRunning] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>("");
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [pipeline] = useState(
    () =>
      new Pipeline((rawImage: RawImage | null) => {
        setImageUri(getBase64FromImage(rawImage));
      })
  );
  let loadingError: any = null;

  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log("Loading models...");
        await pipeline.load(BK_SDM_TINY_VPRED);
        console.log("Models loaded!");
      } catch (e: any) {
        console.error(e);
        loadingError = e;
      }
    };

    setPipelineLoading(true);
    loadModels();
    setPipelineLoading(false);
  }, [pipeline]);

  const generate = async () => {
    try {
      setPipelineRunning(true);
      await pipeline.run(prompt);
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
      <TextInput
        style={styles.input}
        placeholder="a castle"
        maxLength={60}
        value={prompt}
        onChangeText={setPrompt}
      />

      <Button
        title={"Run Pipeline"}
        onPress={generate}
        disabled={pipelineLoading || pipelineRunning || !!loadingError}
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
  input: {
    borderWidth: 1,
    borderColor: "#ccc",
    padding: 8,
    borderRadius: 6,
    width: "100%",
    marginBottom: 12,
  },
  image: {
    width: 320,
    height: 320,
    marginVertical: 50,
    resizeMode: "contain",
  },
});
