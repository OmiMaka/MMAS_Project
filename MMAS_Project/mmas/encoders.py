"""Frozen ImageBind / LanguageBind encoders (GPU, used once to build the cached bundles).

Install the encoder you need first (see README):
  ImageBind     pip install git+https://github.com/facebookresearch/ImageBind.git
  LanguageBind  git clone https://github.com/PKU-YuanGroup/LanguageBind && pip install -r LanguageBind/requirements.txt

Both are imported lazily so the CPU experiments never need torch. The calls follow each project's public
README; if your cached bundles were made with other preprocessing (e.g. the v8/v9 notebooks), keep using those.
"""
import numpy as np

MODALITIES = ("audio", "image", "video", "thermal", "depth")


def _batches(items, n):
    for s in range(0, len(items), n):
        yield items[s:s + n]


class ImageBindEncoder:
    def __init__(self, device="cuda"):
        import torch
        from imagebind import data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType
        self.torch, self.data, self.MT, self.device = torch, data, ModalityType, device
        self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)
        self.key = {"audio": ModalityType.AUDIO, "image": ModalityType.VISION, "video": ModalityType.VISION,
                    "thermal": ModalityType.THERMAL, "depth": ModalityType.DEPTH}

    def _run(self, key, x):
        with self.torch.no_grad():
            return self.model({key: x})[key].float().cpu().numpy()

    def encode_text(self, texts, modality=None, batch=256):
        return np.concatenate([self._run(self.MT.TEXT, self.data.load_and_transform_text(b, self.device))
                               for b in _batches(list(texts), batch)])

    def _load(self, paths, modality):
        d = self.data
        if modality == "audio":
            return d.load_and_transform_audio_data(paths, self.device)
        if modality == "image":
            return d.load_and_transform_vision_data(paths, self.device)
        if modality == "video":
            return d.load_and_transform_video_data(paths, self.device)
        if modality in ("thermal", "depth"):
            # ImageBind's data.py has no loader for these; single-channel 224x224 crop as in its paper.
            from PIL import Image
            from torchvision import transforms
            tf = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.CenterCrop(224), transforms.ToTensor()])
            return self.torch.stack([tf(Image.open(p).convert("L")) for p in paths]).to(self.device)
        raise ValueError(modality)

    def encode_inputs(self, paths, modality, batch=32):
        return np.concatenate([self._run(self.key[modality], self._load(b, modality)) for b in _batches(list(paths), batch)])


class LanguageBindEncoder:
    CKPT = {"video": "LanguageBind_Video_FT", "audio": "LanguageBind_Audio_FT", "thermal": "LanguageBind_Thermal",
            "image": "LanguageBind_Image", "depth": "LanguageBind_Depth"}

    def __init__(self, modality, device="cuda", cache_dir="./lb_cache"):
        import torch
        from languagebind import LanguageBind, LanguageBindImageTokenizer, to_device, transform_dict
        self.torch, self.to_device, self.device, self.modality = torch, to_device, device, modality
        self.model = LanguageBind(clip_type={modality: self.CKPT[modality]}, cache_dir=cache_dir).eval().to(device)
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained("LanguageBind/LanguageBind_Image", cache_dir=cache_dir)
        self.transform = transform_dict[modality](self.model.modality_config[modality])

    def encode_text(self, texts, modality=None, batch=256):
        out = []
        for b in _batches(list(texts), batch):
            tok = self.to_device(self.tokenizer(b, max_length=77, padding="max_length", truncation=True,
                                                return_tensors="pt"), self.device)
            with self.torch.no_grad():
                out.append(self.model({"language": tok})["language"].float().cpu().numpy())
        return np.concatenate(out)

    def encode_inputs(self, paths, modality=None, batch=32):
        out = []
        for b in _batches(list(paths), batch):
            x = self.to_device(self.transform(b), self.device)
            with self.torch.no_grad():
                out.append(self.model({self.modality: x})[self.modality].float().cpu().numpy())
        return np.concatenate(out)


def get_encoder(name, modality, device="cuda"):
    if name == "imagebind":
        return ImageBindEncoder(device)
    if name == "languagebind":
        return LanguageBindEncoder(modality, device)
    raise ValueError(f"unknown encoder {name}")
