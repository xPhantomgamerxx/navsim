import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101
import cv2

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True).to(device).eval()

# Image transforms
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Load image
img_path = "/home/ubuntu/project_ws/navsim/dataset/sensor_blobs/test/2021.09.29.19.02.14_veh-28_00964_01689/CAM_R0/0eee8028ebea5da1.jpg"
input_image = Image.open(img_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Run segmentation model
with torch.no_grad():
    output = model(input_tensor)['out'][0]
pred = output.argmax(0).byte().cpu().numpy()

# Get human mask (COCO class 15)
human_mask = (pred == 15).astype(np.uint8)  # 0s and 1s
kernel = np.ones((7, 7), np.uint8)  
dilated_mask = cv2.dilate(human_mask, kernel, iterations=1)

# Convert PIL image to OpenCV format (BGR)
img_np = np.array(input_image)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# Resize human_mask to match image size (in case of mismatch)
if dilated_mask.shape != img_bgr.shape[:2]:
    dilated_mask = cv2.resize(dilated_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

# Inpaint the image using Navier-Stokes algorithm
inpainted = cv2.inpaint(img_bgr, dilated_mask, inpaintRadius=6, flags=cv2.INPAINT_NS)

# Convert back to RGB and save
output_img = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
original = Image.fromarray(img_np)
output_img.save("output_inpainted_no_humans.jpg")
original.save("original.jpg")
