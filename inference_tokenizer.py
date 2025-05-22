import os
import torch
import torch.nn as nn
import torch.distributed as dist

from PIL import Image

import hydra
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
from omegaconf import OmegaConf
import pyrootutils

import torch.nn.functional as F
from pytorch_lightning.strategies import DDPStrategy
from einops import rearrange, einsum
import transformers

from functools import partial

import numpy as np

from models.slot_qformer.vit import Block
from models.slot_mllm_tokenizer import ImageTokenizer

from utils.config import build_config

from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableUnCLIPImg2ImgPipeline,
)

from vector_quantize_pytorch import ResidualVQ


class DINOBackbone(nn.Module):
    def __init__(self, dinov2):
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x):
        enc_out = self.dinov2.forward_features(x)
        return rearrange(
            enc_out["x_norm_patchtokens"],
            "b (h w ) c -> b c h w",
            h=int(np.sqrt(enc_out["x_norm_patchtokens"].shape[-2]))
        )

class SlotInferenceWrapper(LightningModule):
    """Inference wrapper for Slot

    Args:
        LightningModule (cfg, model): model should be ImageTokenizer
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stage = 2

        # Load tokenizer
        self.image_tokenizer = ImageTokenizer(
            model_path=cfg.checkpoint_path.model_path,
            diffusion_model_path=None,  # Diffusion model is loaded in TrainingWrapper
            device="cpu",  # For PyTorch Lightning
            load_diffusion=False,
            vq_type=None,
            discarding_thre=None,
            from_pretrained=True if cfg.checkpoint_path.model_path is not None else False,
            vit_precision="fp16",
            diffusion_precision="fp16",
            legacy=False,
        )

        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.stage1.dino_model_name)
        self.backbone = DINOBackbone(dinov2)
        self.backbone = self.backbone.half()

        visual_hidden_dim = self.backbone.dinov2.num_features

        self.visual_embedding_layernorm = nn.LayerNorm(visual_hidden_dim)

        self.visual_embedding_encoder = nn.Sequential(
            nn.Linear(visual_hidden_dim, visual_hidden_dim),
            nn.ReLU(),
            nn.Linear(visual_hidden_dim, 1408)
        )

        self.image_tokenizer.model.visual_encoder = None

        self.out_layer_norm = nn.LayerNorm(768)
        self.out_linear_1024 = nn.Linear(768, 1024)

        ### For diffusion DDP
        diffusion_precision = "fp16"
        pretrained_model_name = "stabilityai/stable-diffusion-2-1-unclip"
        self.diffusion_model = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name,
                                                                           torch_dtype=
                                                                           dict(fp16=torch.float16, fp32=torch.float32)[
                                                                               diffusion_precision])

        self.feature_extractor = self.diffusion_model.feature_extractor
        self.image_encoder = self.diffusion_model.image_encoder
        self.image_normalizer = self.diffusion_model.image_normalizer
        self.image_noising_scheduler = self.diffusion_model.image_noising_scheduler
        if self.diffusion_model.text_encoder is not None:
            self.clip_tokenizer = self.diffusion_model.tokenizer
        else:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        if self.diffusion_model.text_encoder is not None:
            self.text_encoder = self.diffusion_model.text_encoder
        self.unet = self.diffusion_model.unet
        self.vae = self.diffusion_model.vae
        self.unet = self.unet.to(dtype=torch.float32)

        ### For diffusion scheduler

        # Change to DDPMScheduler
        self.diffusion_model.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler",
            torch_dtype=dict(fp16=torch.float16, fp32=torch.float32)[diffusion_precision])

        self.scheduler = self.diffusion_model.scheduler

        # Scheduler for validation
        scheduler_args = {}
        if "variance_type" in self.scheduler.config:
            variance_type = self.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        self.val_schduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config, **scheduler_args
        )

        self.use_slot = True

        self.slot_num = 32
        self.slot_config = cfg.stage1.slot_config

        self.use_causal = True

        self.use_blip_itc = False

        # For logging

        self.image_size = cfg.stage1.image_size
        self.transform_256 = transforms.Resize((self.image_size, self.image_size), antialias=True)
        # Resize for CLIP input
        self.transform_224 = transforms.Resize((224, 224), antialias=True)

        self.normalize_diffusion = transforms.Normalize(mean=[0.5], std=[0.5])
        self.normalize_vit = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # Normalize for CLIP input
        self.normalize_clip = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )

        self.save_path = None

        # Use unused model for stage 1, Quantize is not used
        self.image_tokenizer.model.encode_task_layer = None
        self.image_tokenizer.model.decode_task_layer = None
        self.image_tokenizer.model.quantize = None
        self.image_tokenizer.model.blocks = None
        self.image_tokenizer.model.blocks_image = None

        # itc loss
        self.use_itc = True
        self.temp = self.image_tokenizer.model.temp

        self.vision_proj = self.image_tokenizer.model.vision_proj
        self.text_proj = self.image_tokenizer.model.text_proj

        self.text_max_length = 32

        image_feats_size = 768

        self.pos_embed_image = nn.Parameter(torch.zeros(1, self.slot_num, image_feats_size))
        self.blocks_image = nn.ModuleList([
            Block(dim=image_feats_size,
                  num_heads=16,
                  mlp_ratio=4.0,
                  qkv_bias=True,
                  qk_scale=None,
                  drop=0.0,
                  attn_drop=0.0,
                  drop_path=0.0,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(self.cfg.stage2.blocks_layers)
        ])

        self.image_down = nn.Sequential(
            nn.Linear(image_feats_size, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 32, bias=False),
        )
        self.distill_image_proj = nn.Linear(self.slot_num * 32, 1024, bias=False)

        if self.stage == 2:
            self.codebook_embed_dim = self.cfg.stage2.vq.codebook_embed_dim
            self.n_embed = self.cfg.stage2.vq.n_embed  # 8192

            print(f"n_embed: {self.n_embed}, codebook_embed_dim: {self.codebook_embed_dim}")  # 32

            self.quantize = ResidualVQ(
                dim=image_feats_size,
                num_quantizers=self.cfg.stage2.vq.num_quantizers,
                codebook_size=self.n_embed,
                codebook_dim=self.codebook_embed_dim,
                shared_codebook=True,
            )
            self.num_quantizers = self.cfg.stage2.vq.num_quantizers

            self.pos_embed = nn.Parameter(torch.zeros(1, self.slot_num, image_feats_size))
            self.blocks = nn.ModuleList([
                Block(dim=image_feats_size,
                      num_heads=16,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      qk_scale=None,
                      drop=0.0,
                      attn_drop=0.0,
                      drop_path=0.0,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(self.cfg.stage2.blocks_layers)
            ])

            if self.cfg.stage2.use_blocks_image and \
                    self.cfg.stage2.blocks_image_layers is not None:

                self.use_blocks_image = True
                self.pos_embed_image = nn.Parameter(torch.zeros(1, self.slot_num, image_feats_size))
                self.blocks_image = nn.ModuleList([
                    Block(dim=768,
                          num_heads=16,
                          mlp_ratio=4.0,
                          qkv_bias=True,
                          qk_scale=None,
                          drop=0.0,
                          attn_drop=0.0,
                          drop_path=0.0,
                          norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in
                    range(self.cfg.stage2.blocks_image_layers)
                ])
            else:
                self.use_blocks_image = False

    def get_image_feats(self, batch, batch_idx: int):
        """Extract image features using the backbone and Q-former"""
        if len(batch) == 3:
            image, text, image_id = batch
        elif len(batch) == 2:
            image, text = batch
        else:
            raise ValueError(f"Unknown batch size {len(batch)}")

        # Normalize image
        image = self.normalize_vit(image)

        with torch.no_grad():
            image_embeds = self.backbone(image)  # [b, 1024, 16, 16]

        image_embeds = rearrange(image_embeds, "b d h w -> b (h w) d")  # [b, 256, 1024]
        image_embeds = self.visual_embedding_layernorm(image_embeds)
        image_embeds = self.visual_embedding_encoder(image_embeds)  # [b, 256, 1408]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        query_tokens = self.image_tokenizer.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.image_tokenizer.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
            use_slot=True,
            slot_config=self.slot_config,
            use_causal=True,
        )

        return {"image_feats": query_output.last_hidden_state}

    def forward_stage_1(self, x):
        """First stage of the forward pass"""
        image_feats = self.get_image_feats((x, None), batch_idx=0)['image_feats']
        quant, embed_ind, _ = self.quantize(image_feats)
        return embed_ind

    def forward_stage_2(self, embed_ind):
        """Second stage of the forward pass"""
        ret = {}
        quant = self.quantize.get_output_from_indices(embed_ind)
        reconstructed_image_feats = self.apply_transformer(quant, self.blocks, self.pos_embed)

        reconstructed_image_feats_blocks_image_applied = self.apply_transformer(
            reconstructed_image_feats, 
            self.blocks_image, 
            self.pos_embed_image
        )
        slots_1024 = self.convert_image_feats_to_slots(reconstructed_image_feats_blocks_image_applied)
        ret["slots_1024"] = slots_1024

        # For class label input
        reverse_output_proj = self.get_mlp_decoded_embedding(reconstructed_image_feats_blocks_image_applied)
        ret["reverse_output_proj"] = reverse_output_proj

        return ret

    def generate_image(self, ret):
        """Generate image using the diffusion model"""
        self.diffusion_model.scheduler = self.val_schduler

        return self.diffusion_model(
            prompt_embeds=ret['slots_1024'],
            height=self.image_size,
            width=self.image_size,
            guidance_scale=2,
            num_inference_steps=100,
            image_embeds=ret['reverse_output_proj'],
        ).images

    def apply_transformer(self, slots, transformer_blocks, pos_embed):
        """Apply transformer blocks to the slots"""
        pos_embed_applied_slot = slots + pos_embed.repeat(slots.size(0), 1, 1)
        for blk in transformer_blocks:
            pos_embed_applied_slot = blk(pos_embed_applied_slot, use_causal_mask=False)
        return pos_embed_applied_slot

    def convert_image_feats_to_slots(self, image_feats):
        """Convert image features to slots"""
        slots = self.out_layer_norm(image_feats)
        slots_1024 = self.out_linear_1024(slots)
        return slots_1024

    def get_mlp_decoded_embedding(self, image_feats):
        """Get MLP decoded embedding from image features"""
        reverse_output = self.image_down(image_feats)
        reverse_output = reverse_output.reshape(reverse_output.size(0), -1)
        reverse_output_proj = self.distill_image_proj(reverse_output)
        return reverse_output_proj

if __name__ == "__main__":
    cfg, cfg_yaml = build_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_cfg = OmegaConf.load(cfg.transform_cfg_path)
    transform = hydra.utils.instantiate(transform_cfg)

    os.makedirs(cfg.result_file_path, exist_ok=True)

    model = SlotInferenceWrapper(cfg).to(device)
    model.load_state_dict(torch.load(cfg.weight_path)['state_dict'], strict=False)

    image = Image.open("sample_data/sample_img.jpg")

    image = transform(image).unsqueeze(0).to(device)

    with torch.autocast("cuda", dtype=torch.float16):
        slot_tokens = model.forward_stage_1(image)
        print(slot_tokens)

        slots_1024 = model.forward_stage_2(slot_tokens)
        print(slots_1024)

        image = model.generate_image(slots_1024)
        image[0].save("sample_data/sample_img_reconstructed.jpg")

