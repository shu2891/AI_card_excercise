from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class ModelProfile:
    label: str
    repo_id: str
    preferred_width: int
    preferred_height: int
    txt2img_steps: int
    txt2img_guidance: float
    sharpen_percent: int


MODEL_PROFILES = {
    "Fast (SDXL Turbo)": ModelProfile(
        label="Fast (SDXL Turbo)",
        repo_id="stabilityai/sdxl-turbo",
        preferred_width=640,
        preferred_height=960,
        txt2img_steps=4,
        txt2img_guidance=0.0,
        sharpen_percent=110,
    ),
    "Balanced (SSD-1B)": ModelProfile(
        label="Balanced (SSD-1B)",
        repo_id="segmind/SSD-1B",
        preferred_width=768,
        preferred_height=1152,
        txt2img_steps=18,
        txt2img_guidance=6.5,
        sharpen_percent=130,
    ),
    "Quality (SDXL Base)": ModelProfile(
        label="Quality (SDXL Base)",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        preferred_width=832,
        preferred_height=1216,
        txt2img_steps=28,
        txt2img_guidance=7.0,
        sharpen_percent=150,
    ),
}

DEFAULT_AI_MODEL = "Quality (SDXL Base)"

_PIPELINE_LOCK = Lock()
_PIPELINES: dict[str, object] = {}
_PIPELINE_LORA_STATE: dict[str, str] = {}


def _get_profile(model_name: str) -> ModelProfile:
    return MODEL_PROFILES.get(model_name, MODEL_PROFILES[DEFAULT_AI_MODEL])


def _load_pipeline(profile: ModelProfile):
    with _PIPELINE_LOCK:
        if profile.label in _PIPELINES:
            return _PIPELINES[profile.label]

        import torch
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            profile.repo_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to("cuda")
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        _PIPELINES[profile.label] = pipe
        return pipe


def _configure_lora(pipe: object, model_label: str, lora_path: str | None) -> None:
    normalized = (lora_path or "").strip()
    previous = _PIPELINE_LORA_STATE.get(model_label, "")
    if previous == normalized:
        return

    if previous and hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()

    if normalized:
        pipe.load_lora_weights(normalized)

    _PIPELINE_LORA_STATE[model_label] = normalized


def _negative_prompt() -> str:
    return ", ".join(
        [
            "blurry",
            "low detail",
            "photorealistic",
            "3d render",
            "anime",
            "human",
            "multiple fish",
            "collage",
            "busy background",
            "text",
            "watermark",
            "tiny subject",
        ]
    )


def _apply_finish_pass(image: Image.Image, profile: ModelProfile) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=1.8, percent=profile.sharpen_percent, threshold=3))


def generate_raw_lora_images(
    prompt: str,
    seed: int,
    num_images: int = 1,
    model_name: str = DEFAULT_AI_MODEL,
    width: int | None = None,
    height: int | None = None,
    lora_path: str | None = None,
    lora_scale: float = 0.95,
    negative_prompt: str | None = None,
) -> list[Image.Image]:
    import torch

    profile = _get_profile(model_name)
    width = width or profile.preferred_width
    height = height or profile.preferred_height
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty")

    pipe = _load_pipeline(profile)
    _configure_lora(pipe, model_label=profile.label, lora_path=lora_path)
    negative_prompt = (negative_prompt or _negative_prompt()).strip()

    images: list[Image.Image] = []
    for index in range(max(1, int(num_images))):
        generator = torch.Generator(device="cuda").manual_seed(int(seed) + index)
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=profile.txt2img_guidance,
            num_inference_steps=profile.txt2img_steps,
            width=width,
            height=height,
            generator=generator,
            cross_attention_kwargs={"scale": lora_scale} if (lora_path or "").strip() else None,
        )
        images.append(_apply_finish_pass(output.images[0].convert("RGBA"), profile))

    return images
