import { Buffer } from "buffer";
import { PNG } from "pngjs/browser";
import { TensorPtr } from "react-native-executorch/src/types/common";
import {
  addScalar,
  clamp,
  divScalar,
  mulScalar,
  permute120,
  squeeze,
} from "./tensor_utils";

export interface RawImage {
  data: Uint8Array;
  width: number;
  height: number;
}

export function convertTensorToImage(tensor: TensorPtr) {
  let imageTensor = addScalar(divScalar(tensor, 2), 0.5);
  imageTensor = clamp(imageTensor, 0, 1);
  imageTensor = squeeze(imageTensor);
  imageTensor = permute120(imageTensor);
  imageTensor = mulScalar(imageTensor, 255);
  const [height, width, channels] = imageTensor.sizes;
  const floatData = new Float32Array(imageTensor.dataPtr);

  const rgbaData = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    rgbaData[i * 4 + 0] = floatData[i * channels + 0];
    rgbaData[i * 4 + 1] = floatData[i * channels + 1];
    rgbaData[i * 4 + 2] = floatData[i * channels + 2];
    rgbaData[i * 4 + 3] = 255;
  }
  const image = { data: rgbaData, width, height };
  return image;
}

export function getBase64FromImage(image: RawImage | null): string {
  if (!image) {
    return "";
  }
  const png = new PNG({ width: image.width, height: image.height });
  png.data = Buffer.from(image.data);
  const pngBuffer = PNG.sync.write(png, { colorType: 6 });
  const pngString = pngBuffer.toString("base64");
  return pngString;
}
