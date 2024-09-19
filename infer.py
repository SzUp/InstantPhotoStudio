import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import torch
import torch
from diffusers import ControlNetModel, DDIMScheduler, AutoencoderKL,StableDiffusionControlNetPipeline
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
import numpy as np
from PIL import Image
from utils.landmark import get_68landmarks_img

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(256, 256))

image = cv2.imread("PATH TO ID IMAGE")

pose_image_path = "PATH TO SHAPE REF IMAGE"
std_image_path = "PATH TO MODEL IMAGE"

pose_image = cv2.imread(pose_image_path)
std_image = cv2.imread(std_image_path)

faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face()

v2 = True
base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_ckpt = "ip-adapter-faceid-plusv2_sd15.bin" if not v2 else "ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
control_net_openpose = ControlNetModel.from_pretrained(pretrained_model_name_or_path="/opt/data/private/IP-Adapter-main/Face-Landmark-ControlNet/models_for_diffusers", torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
    controlnet = control_net_openpose
)

pipe.load_lora_weights("PATH TO LoRA Weight",adapter_name='lora')
pipe.set_adapters(['lora'],[0.2])
pipe.fuse_lora()

ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)
ip_model.set_scale(1.0)

prompt = "identification photo, a man, little smiling, very short hair, in suit, body facing to viewer, white background"
negative_prompt = "bad teeth, monochrome, lowres,  bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    image = pose_image,
     prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2, 
     s_scale=0.5,
     num_samples=4, width=512, height=768, num_inference_steps=30, seed=2021
)

images[0]