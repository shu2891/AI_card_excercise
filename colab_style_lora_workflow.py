from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_NAME = "翻身"
REPO_URL = "https://github.com/shu2891/AI_card_excercise.git"
DRIVE_ROOT = "/content/drive/MyDrive/AI_Card_Project"
WORKSPACE_ROOT = "/content/AI_Card_Project/workspace"
VENDOR_ROOT = "/content/AI_Card_Project/vendor"
RUNTIME_ROOT = "/content/AI_Card_Project/runtime"

STYLE_TOKEN = "fishseriesstyle"
TRIGGER_PHRASE = "fish series style"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_NAME = "Quality (SDXL Base)"

PROMPT = "fishseriesstyle, same series, pastel chalk fish, single fish hero, mythic regal power"
SEED = 42
NUM_IMAGES = 4
WIDTH = 832
HEIGHT = 1216
LORA_SCALE = 0.95

REPEATS = 12
RESOLUTION = 768
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
MAX_TRAIN_STEPS = 700
RANK = 16
CHECKPOINTING_STEPS = 100
MIXED_PRECISION = "fp16"


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    project_name: str
    base_dir: Path
    source_dir: Path
    dataset_dir: Path
    lora_dir: Path
    output_dir: Path
    repo_dir: Path
    vendor_dir: Path
    runtime_root: Path
    runtime_dir: Path
    runtime_dataset_dir: Path
    runtime_lora_dir: Path
    runtime_output_dir: Path


@dataclass(frozen=True, slots=True)
class StyleDatasetRecord:
    source_path: Path
    target_path: Path
    caption: str


PIP_PACKAGES = [
    "accelerate>=0.34.0",
    "diffusers>=0.35.0",
    "transformers>=4.44.0",
    "huggingface-hub>=0.34.0",
    "safetensors>=0.4.4",
    "Pillow>=10.4.0",
    "PyYAML>=6.0.2",
]

STATUS_SUFFIXES = {
    "生": "fresh",
    "正常": "normal",
    "焦": "charred",
}

SPECIES_ALIASES = {
    "天空": "sky",
    "天空魚": "sky",
    "廚房": "kitchen",
    "廚房魚": "kitchen",
    "蛋": "orb",
    "蛋魚": "orb",
    "馬路": "orb",
    "馬路魚": "orb",
    "球": "orb",
    "球魚": "orb",
}

SPECIES_DESCRIPTIONS = {
    "sky": "sky fish with cloud motifs",
    "kitchen": "kitchen fish with chef motifs",
    "orb": "orb fish with glowing core",
}

STATUS_DESCRIPTIONS = {
    "fresh": "soft mood",
    "normal": "balanced mood",
    "charred": "intense mood",
}

STYLE_SIGNATURE = "same series, pastel chalk fish"


def mount_google_drive() -> None:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)


def install_dependencies() -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", *PIP_PACKAGES])


def clone_repo(repo_url: str, repo_dir: Path) -> Path:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.exists():
        subprocess.check_call(["git", "clone", repo_url, str(repo_dir)])
    return repo_dir


def clone_diffusers_examples(vendor_dir: Path) -> Path:
    diffusers_dir = vendor_dir / "diffusers"
    vendor_dir.mkdir(parents=True, exist_ok=True)
    if not diffusers_dir.exists():
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "https://github.com/huggingface/diffusers.git", str(diffusers_dir)]
        )
    return diffusers_dir


def ensure_repo_imports(repo_dir: Path) -> None:
    src_dir = repo_dir / "src"
    repo_str = str(repo_dir)
    src_str = str(src_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def resolve_project_paths(
    project_name: str,
    drive_root: str = DRIVE_ROOT,
    workspace_root: str = WORKSPACE_ROOT,
    vendor_root: str = VENDOR_ROOT,
    runtime_root: str = RUNTIME_ROOT,
    repo_url: str = REPO_URL,
) -> ProjectPaths:
    base_dir = Path(drive_root) / project_name
    repo_name = Path(repo_url).stem or "AI_card_excercise"
    repo_dir = Path(workspace_root) / repo_name
    vendor_dir = Path(vendor_root)
    runtime_root_path = Path(runtime_root)
    runtime_dir = runtime_root_path / project_name
    return ProjectPaths(
        project_name=project_name,
        base_dir=base_dir,
        source_dir=base_dir / "raw_images",
        dataset_dir=base_dir / "dataset",
        lora_dir=base_dir / "lora",
        output_dir=base_dir / "outputs",
        repo_dir=repo_dir,
        vendor_dir=vendor_dir,
        runtime_root=runtime_root_path,
        runtime_dir=runtime_dir,
        runtime_dataset_dir=runtime_dir / "dataset",
        runtime_lora_dir=runtime_dir / "lora",
        runtime_output_dir=runtime_dir / "outputs",
    )


def ensure_project_dirs(paths: ProjectPaths) -> None:
    for path in (
        paths.base_dir,
        paths.source_dir,
        paths.dataset_dir,
        paths.lora_dir,
        paths.output_dir,
        paths.runtime_root,
        paths.runtime_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def reset_runtime_dirs(paths: ProjectPaths) -> None:
    for path in (paths.runtime_dataset_dir, paths.runtime_lora_dir, paths.runtime_output_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _iter_source_images(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            yield path


def _normalize_status(stem: str) -> tuple[str, str]:
    for suffix, status in STATUS_SUFFIXES.items():
        if stem.endswith(suffix):
            return stem[: -len(suffix)], status
    return stem, "normal"


def _normalize_species(name: str) -> str:
    return SPECIES_ALIASES.get(name, name)


def _parse_source_tags(path: Path, source_root: Path) -> tuple[str, str]:
    stem_base, status = _normalize_status(path.stem)
    candidates = [stem_base]
    try:
        relative_parent = path.parent.relative_to(source_root)
        candidates.extend([part for part in relative_parent.parts if part])
    except ValueError:
        pass

    for candidate in candidates:
        normalized = _normalize_species(candidate)
        if normalized in SPECIES_DESCRIPTIONS:
            return normalized, status
    return _normalize_species(stem_base), status


def _build_style_caption(style_token: str, trigger_phrase: str, species: str, status: str) -> str:
    parts = [
        style_token.strip(),
        trigger_phrase.strip(),
        STYLE_SIGNATURE,
        SPECIES_DESCRIPTIONS.get(species, f"{species} fish"),
        STATUS_DESCRIPTIONS.get(status, "balanced mood"),
        "single fish illustration",
        "dark textured background",
    ]
    return ", ".join(part for part in parts if part)


def _safe_target_name(source_root: Path, source_path: Path) -> str:
    relative = source_path.relative_to(source_root)
    stem_parts = list(relative.with_suffix("").parts)
    flat_stem = "__".join(stem_parts)
    return f"{flat_stem}{source_path.suffix.lower()}"


def _sync_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def prepare_style_dataset(
    source_dir: str | Path,
    dataset_dir: str | Path,
    style_token: str,
    trigger_phrase: str,
    repeats: int = REPEATS,
    runtime_dataset_dir: str | Path | None = None,
    sync_to_drive: bool = True,
) -> list[StyleDatasetRecord]:
    source_dir = Path(source_dir).expanduser().resolve()
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    runtime_dataset_dir = Path(runtime_dataset_dir).expanduser().resolve() if runtime_dataset_dir else dataset_dir

    image_dir = runtime_dataset_dir / "images"
    if runtime_dataset_dir.exists():
        shutil.rmtree(runtime_dataset_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    source_files = list(_iter_source_images(source_dir))
    if not source_files:
        raise ValueError(f"No image files found under {source_dir}")

    records: list[StyleDatasetRecord] = []
    metadata_path = runtime_dataset_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for source_path in source_files:
            species, status = _parse_source_tags(source_path, source_dir)
            caption = _build_style_caption(style_token, trigger_phrase, species, status)
            target_name = _safe_target_name(source_dir, source_path)
            target_path = image_dir / target_name
            shutil.copy2(source_path, target_path)
            relative_name = f"images/{target_name}"

            for _ in range(max(1, int(repeats))):
                metadata_file.write(json.dumps({"file_name": relative_name, "text": caption}, ensure_ascii=False) + "\n")
                records.append(StyleDatasetRecord(source_path=source_path, target_path=target_path, caption=caption))

    if sync_to_drive and runtime_dataset_dir != dataset_dir:
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        _sync_tree(runtime_dataset_dir, dataset_dir)

    return records


def summarize_style_dataset(records: list[StyleDatasetRecord]) -> dict[str, object]:
    return {
        "num_records": len(records),
        "num_unique_sources": len({record.source_path for record in records}),
        "sample_captions": sorted({record.caption for record in records})[:5],
    }


def train_style_lora(
    paths: ProjectPaths,
    base_model: str = BASE_MODEL,
    resolution: int = RESOLUTION,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = LEARNING_RATE,
    max_train_steps: int = MAX_TRAIN_STEPS,
    rank: int = RANK,
    checkpointing_steps: int = CHECKPOINTING_STEPS,
    mixed_precision: str = MIXED_PRECISION,
    resume_from_checkpoint: str | None = None,
    initial_lora_weights_path: str | None = None,
    sync_to_drive: bool = True,
) -> Path:
    diffusers_dir = clone_diffusers_examples(paths.vendor_dir)
    train_script = diffusers_dir / "examples" / "text_to_image" / "train_text_to_image_lora_sdxl.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    if not (paths.runtime_dataset_dir / "metadata.jsonl").exists():
        if not (paths.dataset_dir / "metadata.jsonl").exists():
            raise FileNotFoundError("metadata.jsonl not found. Run prepare_style_dataset() first.")
        if paths.runtime_dataset_dir.exists():
            shutil.rmtree(paths.runtime_dataset_dir)
        _sync_tree(paths.dataset_dir, paths.runtime_dataset_dir)

    if paths.runtime_lora_dir.exists() and not resume_from_checkpoint and not initial_lora_weights_path:
        shutil.rmtree(paths.runtime_lora_dir)
    paths.runtime_lora_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        str(train_script),
        "--pretrained_model_name_or_path",
        base_model,
        "--train_data_dir",
        str(paths.runtime_dataset_dir),
        "--resolution",
        str(resolution),
        "--center_crop",
        "--random_flip",
        "--gradient_checkpointing",
        "--train_batch_size",
        str(train_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--learning_rate",
        str(learning_rate),
        "--max_train_steps",
        str(max_train_steps),
        "--checkpointing_steps",
        str(checkpointing_steps),
        "--rank",
        str(rank),
        "--output_dir",
        str(paths.runtime_lora_dir),
        "--mixed_precision",
        mixed_precision,
        "--report_to",
        "tensorboard",
        "--dataloader_num_workers",
        "0",
    ]

    if resume_from_checkpoint:
        command.extend(["--resume_from_checkpoint", str(resume_from_checkpoint)])
    if initial_lora_weights_path:
        command.extend(["--initial_lora_weights_path", str(initial_lora_weights_path)])

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(paths.repo_dir / "src")
    subprocess.check_call(command, env=env)

    if sync_to_drive:
        if paths.lora_dir.exists():
            shutil.rmtree(paths.lora_dir)
        _sync_tree(paths.runtime_lora_dir, paths.lora_dir)

    return paths.runtime_lora_dir


def latest_lora_path(lora_dir: str | Path) -> Path:
    lora_dir = Path(lora_dir).expanduser().resolve()
    checkpoints = sorted(
        [path for path in lora_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: int(path.name.split("-")[1]),
    )
    if checkpoints:
        return checkpoints[-1]
    return lora_dir


def generate_images(
    prompt: str,
    seed: int,
    num_images: int,
    lora_path: str | Path,
    output_dir: str | Path,
    model_name: str = MODEL_NAME,
    width: int = WIDTH,
    height: int = HEIGHT,
    lora_scale: float = LORA_SCALE,
    runtime_output_dir: str | Path | None = None,
    sync_to_drive: bool = True,
) -> list[Path]:
    from local_stagegen.ai_card_renderer import generate_raw_lora_images

    output_dir = Path(output_dir).expanduser().resolve()
    runtime_output_dir = Path(runtime_output_dir).expanduser().resolve() if runtime_output_dir else output_dir
    runtime_output_dir.mkdir(parents=True, exist_ok=True)

    images = generate_raw_lora_images(
        prompt=prompt,
        seed=seed,
        num_images=num_images,
        model_name=model_name,
        width=width,
        height=height,
        lora_path=str(Path(lora_path).expanduser().resolve()),
        lora_scale=lora_scale,
    )

    saved_paths: list[Path] = []
    for index, image in enumerate(images):
        out_path = runtime_output_dir / f"generated_{seed}_{index:02d}.png"
        image.save(out_path)
        saved_paths.append(out_path)

    if sync_to_drive and runtime_output_dir != output_dir:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        _sync_tree(runtime_output_dir, output_dir)
        return [output_dir / path.name for path in saved_paths]

    return saved_paths


def main(
    project_name: str = PROJECT_NAME,
    prompt: str = PROMPT,
    seed: int = SEED,
    num_images: int = NUM_IMAGES,
    skip_install: bool = False,
    skip_train: bool = False,
) -> dict[str, object]:
    mount_google_drive()
    if not skip_install:
        install_dependencies()

    paths = resolve_project_paths(project_name=project_name)
    ensure_project_dirs(paths)
    reset_runtime_dirs(paths)
    clone_repo(REPO_URL, paths.repo_dir)
    ensure_repo_imports(paths.repo_dir)

    records = prepare_style_dataset(
        source_dir=paths.source_dir,
        dataset_dir=paths.dataset_dir,
        style_token=STYLE_TOKEN,
        trigger_phrase=TRIGGER_PHRASE,
        repeats=REPEATS,
        runtime_dataset_dir=paths.runtime_dataset_dir,
        sync_to_drive=True,
    )
    dataset_summary = summarize_style_dataset(records)

    if not skip_train:
        train_style_lora(paths=paths, sync_to_drive=True)

    active_lora = latest_lora_path(paths.runtime_lora_dir if paths.runtime_lora_dir.exists() else paths.lora_dir)
    saved_images = generate_images(
        prompt=prompt,
        seed=seed,
        num_images=num_images,
        lora_path=active_lora,
        output_dir=paths.output_dir,
        model_name=MODEL_NAME,
        width=WIDTH,
        height=HEIGHT,
        lora_scale=LORA_SCALE,
        runtime_output_dir=paths.runtime_output_dir,
        sync_to_drive=True,
    )

    result = {
        "project_name": project_name,
        "base_dir": str(paths.base_dir),
        "source_dir": str(paths.source_dir),
        "dataset_dir": str(paths.dataset_dir),
        "lora_dir": str(paths.lora_dir),
        "output_dir": str(paths.output_dir),
        "runtime_dataset_dir": str(paths.runtime_dataset_dir),
        "runtime_lora_dir": str(paths.runtime_lora_dir),
        "runtime_output_dir": str(paths.runtime_output_dir),
        "active_lora": str(active_lora),
        "dataset_summary": dataset_summary,
        "generated_images": [str(path) for path in saved_images],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
