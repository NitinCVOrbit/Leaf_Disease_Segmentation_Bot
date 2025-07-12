import torch
import cv2
import numpy as np
import torchvision.transforms as T
import segmentation_models_pytorch as smp


def create_model():
    
    # Load Pretrained U-Net with ResNet34 backbone
    model = smp.Unet(
        encoder_name="resnet34",         # Use ResNet34 as encoder
        encoder_weights="imagenet",      # Pretrained on ImageNet
        in_channels=3,                   # RGB input
        classes=2,                       # 2 output channels (BG & Leaf_disase)
    )
        
    return model


def draw_segmentation(image, output, class_colors):
        
    # Define transparency level (0 to 1)
    alpha = 0.5  # 50% transparency

    predicted_mask = torch.argmax(output, dim=0).cpu().numpy()

    # ✅ Fixed: Convert PIL image to NumPy before resizing
    image_np = np.array(image)  # Convert PIL to NumPy

    # ✅ Fixed: Use `.size` instead of `.shape` for PIL
    original_width, original_height = image.size

    # Resize predicted mask to match original image size
    predicted_mask_resized = cv2.resize(predicted_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # Create a colored overlay of the same shape as the image
    overlay = np.zeros_like(image_np, dtype=np.uint8)

    # Apply colors for Background and Leaf
    for class_id, color in class_colors.items():
        overlay[predicted_mask_resized == class_id] = np.array(color, dtype=np.uint8)
    
    blended_image = image_np.copy()

    # Blend the original image and the overlay using alpha
    blended_image[predicted_mask_resized == 1] = cv2.addWeighted(image_np, 1 - alpha, overlay, alpha, 0)[predicted_mask_resized == 1]
    
    return blended_image

def segment(image, class_colors):
    
    # Load the model
    model = create_model()
    model.load_state_dict(torch.load("leaf.pth", map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    
    # ✅ Fixed: Correct preprocessing order
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # Convert PIL to Tensor FIRST ✅
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize AFTER ToTensor ✅
    ])
    
    # Apply model transformations
    input_tensor = transform(image).unsqueeze(0) # ✅ Now correctly formatted

    # Model Prediction
    with torch.no_grad():
        output = model(input_tensor)[0]  # Get model output        
    
    # Draw Predictions on the image
    image_with_segment = draw_segmentation(image, output, class_colors)

    return image_with_segment

