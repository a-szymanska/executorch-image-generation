export interface ModelSource {
  schedulerSource: string;
  tokenizer: {
    tokenizerSource: string;
  };
  encoderSource: string;
  unetSource: string;
  decoderSource: string;
  size: number;
}

export const BK_SDM_TINY_VPRED_256 = {
  schedulerSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/scheduler/scheduler_config.json",
  tokenizer: {
    tokenizerSource:
      "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/tokenizer/tokenizer.json",
  },
  encoderSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/text_encoder/model.pte",
  unetSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/unet/model.256.pte",
  decoderSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/vae/model.256.pte",
  size: 256,
};

export const BK_SDM_TINY_VPRED_512 = {
  schedulerSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/scheduler/scheduler_config.json",
  tokenizer: {
    tokenizerSource:
      "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/tokenizer/tokenizer.json",
  },
  encoderSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/text_encoder/model.pte",
  unetSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/unet/model.pte",
  decoderSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/vae/model.pte",
  size: 512,
};
