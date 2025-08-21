import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import json
from tqdm.auto import tqdm
import os

device = "cpu"
model_id = "vivym/bk-sdm-tiny-vpred"

tokenizer = CLIPTokenizer.from_pretrained(
    model_id, subfolder="tokenizer"
)
scheduler = PNDMScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)

text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder", use_safetensors=True
).to(device)
unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", use_safetensors=True
).to(device)
vae = AutoencoderKL.from_pretrained(
    model_id, subfolder="vae", use_safetensors=True
).to(device)

prompt = ["a castle"]
height = 512
width = 512
num_inference_steps = 25
guidance_scale = 7.5
generator = torch.manual_seed(0)
batch_size = len(prompt)

output_dir = "../assets/data"
os.makedirs(output_dir, exist_ok=True)

print("--- Generating input for text encoder ---")
tokens = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)
input_ids_tensor = tokens.input_ids
input_ids_list = input_ids_tensor.squeeze().tolist()
shape = list(input_ids_tensor.shape)

output_data_text_encoder = {
    "prompt": prompt,
    "shape": shape,
    "input_ids": input_ids_list
}
file_path_text_encoder = os.path.join(output_dir, "text_encoder.json")
with open(file_path_text_encoder, 'w') as f:
    json.dump(output_data_text_encoder, f, indent=4)


print("--- Generating text embeddings ---")
with torch.no_grad():
    text_embeddings = text_encoder(input_ids_tensor.to(device))[0]

max_length = tokens.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

print("--- Generating input for UNet ---")
latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
).to(device)

scheduler.set_timesteps(num_inference_steps)
latents = latents * scheduler.init_noise_sigma

t = scheduler.timesteps[0]
latent_model_input = torch.cat([latents] * 2)
latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
timestep_tensor = t.to(torch.int64).unsqueeze(0)

output_data_unet = {
    "latent_model_input": {
        "shape": list(latent_model_input.shape),
        "values": latent_model_input.flatten().tolist()
    },
    "timestep": {
        "shape": list(timestep_tensor.shape),
        "values": timestep_tensor.flatten().tolist()
    },
    "text_embeddings": {
        "shape": list(text_embeddings.shape),
        "values": text_embeddings.flatten().tolist()
    }
}
file_path_unet = os.path.join(output_dir, "unet.json")
with open(file_path_unet, 'w') as f:
    json.dump(output_data_unet, f)


for t in tqdm(scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    with torch.no_grad():
        timestep = t.to(torch.int64).unsqueeze(0).to(device)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    latents = scheduler.step(noise_pred, t, latents).prev_sample


print("--- Generating input for VAE decoder ---")
final_latents = 1 / 0.18215 * latents

output_data_vae = {
    "latents": {
        "shape": list(final_latents.shape),
        "values": final_latents.flatten().tolist()
    }
}
file_path_vae = os.path.join(output_dir, "vae.json")
with open(file_path_vae, 'w') as f:
    json.dump(output_data_vae, f)

print("Script finished successfully!")