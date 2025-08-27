export interface ModelSource {
  schedulerSource: string;
  tokenizer: {
    tokenizerSource: string;
  };
  encoderSource: string;
  unetSource: string;
  decoderSource: string;
}

export const BK_SDM_TINY_VPRED = {
  schedulerSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/scheduler/scheduler_config.json",
  tokenizer: {
    tokenizerSource:
      "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/tokenizer.json",
  },
  encoderSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/text_encoder/model.pte",
  unetSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/unet/model.pte",
  decoderSource:
    "https://huggingface.co/aszymanska/bk-sdm-tiny-vpred/resolve/main/vae/model.pte",
};
