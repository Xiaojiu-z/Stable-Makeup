import gradio as gr
import torch
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, ControlNetModel
from detail_encoder.encoder_plus import detail_encoder
from spiga_draw import *
from PIL import Image
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
from facelib import FaceDetector

torch.cuda.set_device(0)

processor = SPIGAFramework(ModelConfig("300wpublic"))
detector = FaceDetector(weight_path="./models/mobilenet0.25_Final.pth")

def get_draw(pil_img, size):
    spigas = spiga_process(pil_img, detector)
    if spigas == False:
        width, height = pil_img.size  
        black_image_pil = Image.new('RGB', (width, height), color=(0, 0, 0))  
        return black_image_pil
    else:
        # width, height = pil_img.size  
        # black_image_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
        # return black_image_pil
        spigas_faces = spiga_segmentation(spigas, size=size)
        return spigas_faces

# Initialize the model
model_id = "sd_model_v1-5"  # your sd1.5 model path
base_path = "./models/stablemakeup"
makeup_encoder_path = base_path + "/pytorch_model.bin"
id_encoder_path = base_path + "/pytorch_model_1.bin"
pose_encoder_path = base_path + "/pytorch_model_2.bin"

Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")
id_encoder = ControlNetModel.from_unet(Unet)
pose_encoder = ControlNetModel.from_unet(Unet)
makeup_encoder = detail_encoder(Unet, "./models/image_encoder_l", "cuda", dtype=torch.float32)
makeup_state_dict = torch.load(makeup_encoder_path)
id_state_dict = torch.load(id_encoder_path)
id_encoder.load_state_dict(id_state_dict, strict=False)
pose_state_dict = torch.load(pose_encoder_path)
pose_encoder.load_state_dict(pose_state_dict, strict=False)
makeup_encoder.load_state_dict(makeup_state_dict, strict=False)

pose_encoder.to("cuda")
id_encoder.to("cuda")
makeup_encoder.to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=[id_encoder, pose_encoder],
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Define your ML model or function here
def model_call(id_image, makeup_image, num):
    # # Your ML logic goes here
    id_image = Image.fromarray(id_image.astype('uint8'), 'RGB')
    makeup_image = Image.fromarray(makeup_image.astype('uint8'), 'RGB')
    id_image = id_image.resize((512, 512))
    makeup_image = makeup_image.resize((512, 512))
    pose_image = get_draw(id_image, size=512)
    result_img = makeup_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image, guidance_scale=num, pipe=pipe)
    return result_img

# Create a Gradio interface
image1 = gr.inputs.Image(label="id_image")
image2 = gr.inputs.Image(label="makeup_image")
number = gr.inputs.Slider(minimum=1.01, maximum=5, default=1.5, label="makeup_guidance_scale")
output = gr.outputs.Image(type="pil", label="Output Image")

iface = gr.Interface(
    fn=lambda id_image, makeup_image, num: model_call(id_image, makeup_image, num),
    inputs=[image1, image2, number],
    outputs=output,
    title="Facial Makeup Transfer Demo",
    description="Upload 2 images to see the model output. 1.05-1.15 is suggested for light makeup and 2 for heavy makeup"
)
# Launch the Gradio interface
iface.queue().launch(server_name='0.0.0.0')
