import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image, imsize=224):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    return torch.mm(features, features.t()) / (b * c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        else:
            continue
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    return model, style_losses, content_losses

def run_style_transfer(content_img, style_img, steps=50, style_weight=1e6, content_weight=1):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    input_img = content_img.clone().requires_grad_(True)
    model, style_losses, content_losses = get_model_and_losses(
        cnn, cnn_mean, cnn_std, style_img, content_img
    )
    optimizer = optim.Adam([input_img], lr=0.02)

    for step in range(steps):
        with torch.no_grad():
            input_img.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight * style_score + content_weight * content_score
        loss.backward()
        optimizer.step()
    input_img.data.clamp_(0, 1)
    return input_img

# Streamlit UI
st.set_page_config(page_title="Style Transfer", layout="centered")
st.title("🎨 Neural Style Transfer (GPU-Ready)")
st.write("Upload a content image and a style image. The app uses VGG19 and GPU acceleration for fast transfer.")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if st.button("Generate Stylized Image") and content_file and style_file:
    content_img = image_loader(content_file)
    style_img = image_loader(style_file)
    with st.spinner("Generating stylized image using GPU..."):
        output_tensor = run_style_transfer(content_img, style_img)
    output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())
    st.image(output_image, caption="Stylized Result", use_column_width=True)
    output_image.save("stylized_result.png")
    with open("stylized_result.png", "rb") as file:
        st.download_button("Download Image", file, file_name="stylized_result.png")