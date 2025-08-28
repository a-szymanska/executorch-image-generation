/* eslint-disable react-hooks/exhaustive-deps */
import { getBase64FromImage, RawImage } from "@/pipeline/image_utils";
import {
  BK_SDM_TINY_VPRED_256,
  BK_SDM_TINY_VPRED_512,
} from "@/constants/Model";
import { Pipeline } from "@/pipeline/pipeline";
import React, { useEffect, useRef, useState } from "react";
import {
  Button,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
} from "react-native";

export default function App() {
  const pipelineLoading = useRef<boolean>(false);
  const [pipelineRunning, setPipelineRunning] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>("a castle");
  const [imageUri, setImageUri] = useState<string | null>(null);
  const displayImage = (rawImage: RawImage | null) => {
    setImageUri(getBase64FromImage(rawImage));
  };
  const [pipeline] = useState(() => new Pipeline());
  let loadingError: any = null;

  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log("Loading models...");
        pipelineLoading.current = true;
        await pipeline.load(BK_SDM_TINY_VPRED_256);
        console.log("Models loaded!");
      } catch (e: any) {
        console.error(e);
        loadingError = e;
      } finally {
        pipelineLoading.current = false;
      }
    };

    loadModels();
  }, [pipeline]);

  const generate = async () => {
    try {
      setPipelineRunning(true);
      const rawImage = await pipeline.run(prompt, 20);
      displayImage(rawImage);
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
        disabled={pipelineLoading.current || pipelineRunning || !!loadingError}
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
    width: 256,
    height: 256,
    marginVertical: 50,
    resizeMode: "contain",
  },
});
