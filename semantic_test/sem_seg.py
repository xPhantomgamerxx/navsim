import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained DeepLabV3 model and move to GPU
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

# Run the model
with torch.no_grad():
    output = model(input_tensor)['out'][0]
pred = output.argmax(0).byte().cpu().numpy()

# Human class in COCO = 15
human_mask = pred == 15

# Convert image to numpy
img_np = np.array(input_image)

# Mask out humans with black pixels
img_np[human_mask] = [0, 0, 0]

# Save result
masked_img = Image.fromarray(img_np)
masked_img.save("output_no_humans.jpg")
