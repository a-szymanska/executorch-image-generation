import * as tf from "@tensorflow/tfjs";
import {
  ScalarType,
  TensorPtr,
} from "react-native-executorch/src/types/common";

type Tensor = tf.Tensor;

function convertTfToTensorPtr(tensor: Tensor): TensorPtr {
  const data = tensor.dataSync();

  return {
    dataPtr: new Float32Array(data.buffer),
    sizes: tensor.shape,
    scalarType: ScalarType.FLOAT,
  };
}

function convertTensorPtrToTf(tensorPtr: TensorPtr): Tensor {
  const dtype = tensorPtr.scalarType === ScalarType.FLOAT ? "float32" : "int32";

  const tensor = tf.tensor(
    new Float32Array(tensorPtr.dataPtr),
    tensorPtr.sizes,
    dtype
  );

  return tensor;
}

class SchedulerOutput {
  public prev_sample: TensorPtr;
  [key: string]: any;

  constructor(initialData: { prev_sample: Tensor; [key: string]: any }) {
    Object.assign(this, initialData);
    this.prev_sample = convertTfToTensorPtr(initialData.prev_sample);
  }
}

interface SchedulerConfig {
  beta_start: number;
  beta_end: number;
  num_train_timesteps: number;
  steps_offset: number;
}

export class Scheduler {
  private readonly config: SchedulerConfig;
  private readonly betas: Tensor;
  private readonly alphas: Tensor;
  private readonly alphas_cumprod: Tensor;
  private readonly final_alpha_cumprod: Tensor;
  public readonly init_noise_sigma: number = 1.0;

  private counter: number = 0;
  private cur_sample: Tensor | null = null;
  private ets: Tensor[] = [];
  private num_inference_steps: number | null = null;
  private _timesteps: number[] = [];
  public timesteps: Tensor | null = null;

  constructor(config: SchedulerConfig) {
    this.config = config;
    console.log(config);

    this.betas = tf
      .linspace(
        Math.sqrt(this.config.beta_start),
        Math.sqrt(this.config.beta_end),
        this.config.num_train_timesteps
      )
      .square();

    const betasArray = this.betas.arraySync() as number[];
    const alphasArray: number[] = [];
    for (const betaValue of betasArray) {
      const alphaValue = 1.0 - betaValue;
      alphasArray.push(alphaValue);
    }
    this.alphas = alphasArray;
    const alphasCumprodArray: number[] = [];
    let runningProduct = 1.0;

    for (const alphaValue of this.alphas) {
      runningProduct = runningProduct * alphaValue;
      alphasCumprodArray.push(runningProduct);
    }
    this.alphas_cumprod = alphasCumprodArray;
    this.final_alpha_cumprod = this.alphas_cumprod.slice(0, 1);

    const initialTimesteps = Array.from(
      { length: this.config.num_train_timesteps },
      (_, i) => i
    );
    this._timesteps = initialTimesteps.reverse();
  }

  public set_timesteps(num_inference_steps: number): void {
    this.num_inference_steps = num_inference_steps;

    const step_ratio =
      this.config.num_train_timesteps / this.num_inference_steps;

    this._timesteps = Array.from(
      { length: num_inference_steps },
      (_, i) => Math.round(i * step_ratio) + this.config.steps_offset
    );

    const plms_timesteps = [
      ...this._timesteps.slice(0, -1),
      this._timesteps[this._timesteps.length - 2],
      this._timesteps[this._timesteps.length - 1],
    ].reverse();

    this.timesteps = plms_timesteps;

    this.ets = [];
    this.counter = 0;
  }

  public step(
    model_output_ptr: Tensor,
    timestep: number,
    sample: Tensor
  ): SchedulerOutput {
    if (this.num_inference_steps === null) {
      throw new Error(
        "Number of inference steps is not set. Call `set_timesteps` first."
      );
    }
    const model_output = convertTensorPtrToTf(model_output_ptr);
    let prev_timestep =
      timestep - this.config.num_train_timesteps / this.num_inference_steps;

    if (this.counter !== 1) {
      this.ets = this.ets.slice(-3);
      this.ets.push(model_output);
    } else {
      prev_timestep = timestep;
      timestep =
        timestep + this.config.num_train_timesteps / this.num_inference_steps;
    }

    let ets_output: Tensor;
    if (this.ets.length === 1 && this.counter === 0) {
      ets_output = model_output;
      this.cur_sample = sample;
    } else if (this.ets.length === 1 && this.counter === 1) {
      ets_output = model_output.add(this.ets[0]).div(2);
      sample = this.cur_sample!;
      this.cur_sample = null;
    } else if (this.ets.length === 2) {
      ets_output = this.ets[1].mul(3).sub(this.ets[0]).div(2);
    } else if (this.ets.length === 3) {
      const term1 = this.ets[2].mul(23);
      const term2 = this.ets[1].mul(16);
      const term3 = this.ets[0].mul(5);
      ets_output = term1.sub(term2).add(term3).div(12);
    } else {
      const term1 = this.ets[3].mul(55);
      const term2 = this.ets[2].mul(59);
      const term3 = this.ets[1].mul(37);
      const term4 = this.ets[0].mul(9);
      ets_output = term1.sub(term2).add(term3).sub(term4).div(24);
    }

    const prev_sample = this._get_prev_sample(
      sample,
      timestep,
      prev_timestep,
      ets_output
    );
    this.counter += 1;

    return new SchedulerOutput({ prev_sample });
  }

  private _get_prev_sample(
    sample: Tensor,
    timestep: number,
    prev_timestep: number,
    model_output: Tensor
  ): Tensor {
    return tf.tidy(() => {
      const alpha_prod_t = this.alphas_cumprod.gather([timestep]);

      const alpha_prod_t_prev =
        prev_timestep >= 0
          ? this.alphas_cumprod.gather([prev_timestep])
          : this.final_alpha_cumprod;

      const beta_prod_t = tf.scalar(1).sub(alpha_prod_t);
      const beta_prod_t_prev = tf.scalar(1).sub(alpha_prod_t_prev);

      const pred_original_sample = model_output
        .mul(alpha_prod_t.sqrt())
        .add(sample.mul(beta_prod_t.sqrt()));

      const sample_coeff = alpha_prod_t_prev.div(alpha_prod_t).sqrt();

      const model_output_denom_coeff = alpha_prod_t
        .mul(beta_prod_t_prev.sqrt())
        .add(alpha_prod_t.mul(beta_prod_t).mul(alpha_prod_t_prev).sqrt());

      const model_output_term = pred_original_sample
        .mul(alpha_prod_t_prev.sub(alpha_prod_t))
        .div(model_output_denom_coeff);

      const prev_sample = sample.mul(sample_coeff).sub(model_output_term);

      return prev_sample;
    });
  }

  public get length(): number {
    return this.config.num_train_timesteps;
  }
}
