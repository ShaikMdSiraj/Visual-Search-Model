# src/extract_features.py
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

def extract_features(image_dir, output_file, model_name='resnet50', batch_size=32, device='cuda'):
    # Device setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained model and remove classification head
    if model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model = torch.nn.Sequential(*list(model.children())[:-1])  # remove fc layer
    else:
        raise ValueError("Only resnet50 is implemented for now")

    model = model.to(device)
    model.eval()

    # Image preprocessing: resize, center crop, normalize to ImageNet stats
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Gather all image paths (supports common extensions)
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(img_extensions):
                image_paths.append(os.path.join(root, f))

    print(f"Found {len(image_paths)} images.")

    # Process in batches to be efficient
    all_embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            batch_tensors.append(img_tensor)
        batch_tensors = torch.cat(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            embeddings = model(batch_tensors)
            embeddings = embeddings.squeeze()  # remove the spatial dim
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)  # handle batch size 1
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)

    # Save both embeddings and paths for later use
    np.save(output_file, all_embeddings)
    np.save(output_file.replace('.npy', '_paths.npy'), np.array(image_paths))

    print(f"Saved embeddings shape {all_embeddings.shape} to {output_file}")
    print(f"Saved image paths to {output_file.replace('.npy', '_paths.npy')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image dataset")
    parser.add_argument("--output", type=str, default="../models/embeddings.npy", help="Output .npy file for embeddings")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    extract_features(args.image_dir, args.output, args.model, args.batch_size, args.device)