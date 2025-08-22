import { TensorPtr } from "react-native-executorch/src/types/common";
import { addTensors, divScalar, mulScalar, subTensors } from "./tensor_utils";

class SchedulerOutput {
  public prev_sample: TensorPtr;
  [key: string]: any;

  constructor(initialData: { prev_sample: TensorPtr; [key: string]: any }) {
    Object.assign(this, initialData);
    this.prev_sample = initialData.prev_sample;
  }
}

interface SchedulerConfig {
  beta_start: number;
  beta_end: number;
  num_train_timesteps: number;
  steps_offset: number;
}

async function fetchConfig(url: string): Promise<SchedulerConfig | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch scheduler config from ${url}`);
    }
    const config = await response.json();
    return config;
  } catch (e) {
    console.error("Error fetching scheduler:", e);
    return null;
  }
}

export class Scheduler {
  public config: SchedulerConfig | null = null;
  private readonly betas: number[];
  private readonly alphas: number[];
  public readonly alphas_cumprod: number[];
  private readonly final_alpha_cumprod: number;
  public readonly init_noise_sigma: number = 1.0;

  private counter: number = 0;
  private cur_sample: TensorPtr | null = null;
  private ets: TensorPtr[] = [];
  private num_inference_steps: number | null = null;
  public timesteps: number[] = [];
  private _timesteps: number[] = [];

  public async load(config: string) {
    this.config = await fetchConfig(config);
    if (!config) {
      console.error("Scheduler config not loaded properly.");
      return;
    }

    const betasArray: number[] = [];
    const start = Math.sqrt(this.config!.beta_start);
    const end = Math.sqrt(this.config!.beta_end);
    const step = (end - start) / (this.config!.num_train_timesteps - 1);
    for (let i = 0; i < this.config!.num_train_timesteps; i++) {
      const value = start + step * i;
      betasArray.push(value * value);
    }
    this.betas = betasArray;

    this.alphas = this.betas.map((beta) => 1.0 - beta);

    const alphasCumprodArray: number[] = [];
    let runningProduct = 1.0;
    for (const alphaValue of this.alphas) {
      runningProduct *= alphaValue;
      alphasCumprodArray.push(runningProduct);
    }
    this.alphas_cumprod = alphasCumprodArray;

    this.final_alpha_cumprod = this.alphas_cumprod[0];

    const initialTimesteps = Array.from(
      { length: this.config!.num_train_timesteps },
      (_, i) => i
    );
    this._timesteps = initialTimesteps.reverse();
  }

  public set_timesteps(num_inference_steps: number): void {
    this.num_inference_steps = num_inference_steps;

    const step_ratio =
      this.config!.num_train_timesteps / this.num_inference_steps;

    this._timesteps = Array.from(
      { length: num_inference_steps },
      (_, i) => Math.round(i * step_ratio) + this.config!.steps_offset
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
    model_output: TensorPtr,
    timestep: number,
    sample: TensorPtr
  ): SchedulerOutput {
    if (this.num_inference_steps === null) {
      throw new Error(
        "Number of inference steps is not set. Call `set_timesteps` first."
      );
    }

    let prev_timestep =
      timestep - this.config!.num_train_timesteps / this.num_inference_steps;

    if (this.counter !== 1) {
      this.ets = this.ets.slice(-3);
      this.ets.push(model_output);
    } else {
      prev_timestep = timestep;
      timestep =
        timestep + this.config!.num_train_timesteps / this.num_inference_steps;
    }

    let ets_output: TensorPtr;
    if (this.ets.length === 1 && this.counter === 0) {
      ets_output = model_output;
      this.cur_sample = sample;
    } else if (this.ets.length === 1 && this.counter === 1) {
      const sum = addTensors(model_output, this.ets[0]);
      ets_output = divScalar(sum, 2);
      sample = this.cur_sample!;
      this.cur_sample = null;
    } else if (this.ets.length === 2) {
      const term1 = mulScalar(this.ets[1], 3);
      const diff = subTensors(term1, this.ets[0]);
      ets_output = divScalar(diff, 2);
    } else if (this.ets.length === 3) {
      const term1 = mulScalar(this.ets[2], 23);
      const term2 = mulScalar(this.ets[1], 16);
      const term3 = mulScalar(this.ets[0], 5);
      const diff1 = subTensors(term1, term2);
      const sum = addTensors(diff1, term3);
      ets_output = divScalar(sum, 12);
    } else {
      const term1 = mulScalar(this.ets[3], 55);
      const term2 = mulScalar(this.ets[2], 59);
      const term3 = mulScalar(this.ets[1], 37);
      const term4 = mulScalar(this.ets[0], 9);
      const diff1 = subTensors(term1, term2);
      const sum = addTensors(diff1, term3);
      const diff2 = subTensors(sum, term4);
      ets_output = divScalar(diff2, 24);
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
    sample: TensorPtr,
    timestep: number,
    prev_timestep: number,
    model_output: TensorPtr
  ): TensorPtr {
    const alpha_prod_t = this.alphas_cumprod[timestep];
    const alpha_prod_t_prev =
      prev_timestep >= 0
        ? this.alphas_cumprod[prev_timestep]
        : this.final_alpha_cumprod;

    const beta_prod_t = 1 - alpha_prod_t;
    const beta_prod_t_prev = 1 - alpha_prod_t_prev;

    const pred_original_sample_term1 = mulScalar(
      model_output,
      Math.sqrt(alpha_prod_t)
    );
    const pred_original_sample_term2 = mulScalar(
      sample,
      Math.sqrt(beta_prod_t)
    );
    const pred_original_sample = addTensors(
      pred_original_sample_term1,
      pred_original_sample_term2
    );

    const sample_coeff = Math.sqrt(alpha_prod_t_prev / alpha_prod_t);

    const model_output_denom_coeff =
      alpha_prod_t * Math.sqrt(beta_prod_t_prev) +
      Math.sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);

    const model_output_term_numerator = mulScalar(
      pred_original_sample,
      alpha_prod_t_prev - alpha_prod_t
    );
    const model_output_term = divScalar(
      model_output_term_numerator,
      model_output_denom_coeff
    );

    const prev_sample_term = mulScalar(sample, sample_coeff);
    const prev_sample = subTensors(prev_sample_term, model_output_term);

    return prev_sample;
  }

  public get length(): number {
    return this.config!.num_train_timesteps;
  }
}
