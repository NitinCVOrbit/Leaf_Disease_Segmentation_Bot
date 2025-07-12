import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO
from PIL import Image
from Segmentation import segment

# # Class names and colors (BGR format for OpenCV)
class_names = ['bg', 'Leaf_Disease']
color_sample = {
   0 : (0, 0, 0),  # Background - Black  
   1 : (0, 0, 255),  
}

# Telegram Bot Token (replace with your own)
TOKEN = "7757146225:AAEMvEnsRItUvqPHtLX_xsV269aY3ZlZZQw"    

# Object Detection Function
def object_detection_image(image):
    image_with_segment = segment(image, color_sample)
    # Convert to NumPy array
    image = np.array(image_with_segment)
    _, img_encoded = cv2.imencode('.jpg', image)
    return BytesIO(img_encoded.tobytes())


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "🌿 **Welcome to the Leaf Disease Detection Bot!** 🧬\n\n"
        "📸 **Send me an image of a leaf**, and I will:\n"
        "✅ Segment diseased regions on the leaf\n"
        "✅ Identify potential plant diseases\n"
        "✅ Return an annotated image with highlighted affected areas 🧪\n\n"
        "**Possible Results:**\n"
        "🟥 **Diseased Region**\n"
        "⚠️ **Disease Severity Indicator**\n\n"
        "🚀 *Send a clear image of a leaf to begin detection!*"
    )
    
    
async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "🌿 **Leaf Disease Detection Bot Help** 🧬\n\n"
        "This bot segments diseased areas on leaf images and helps identify potential plant health issues. 📸\n"
        "Simply send a **clear image of a leaf**, and I'll analyze it to detect:\n"
        "🟥 **Diseased Regions**\n"
        "🟩 **Healthy Leaf Areas**\n"
        "⚠️ **Disease Severity Warnings**\n\n"
        "🔍 The bot will also **generate an image with segmented masks** to highlight affected parts of the leaf.\n\n"
        "**Available Commands:**\n"
        "📌 /start - Welcome message\n"
        "📌 /help - Instructions on using the bot\n"
        "📌 /about - Information about the segmentation model\n"
    )
    await update.message.reply_text(help_text)
    

# About Command
async def about_command(update: Update, context: CallbackContext):
    about_text = (
        "🌿 **Leaf Disease Segmentation Bot** 🧬\n\n"
        "This bot detects and segments **diseased regions on leaf images** using deep learning-based image segmentation. 🍃\n\n"
        
        "**🔍 Features:**\n"
        "✅ **Pixel-level segmentation** of leaf vs. background 🌱\n"
        "✅ **Visual overlays** highlight affected areas using colored masks 🎨\n"
        "✅ **U-Net Model** with a **ResNet-34 encoder** trained on real plant leaf datasets 🧠\n"
        "✅ **Advanced Data Augmentation** for better generalization 📈\n"
        "✅ **GPU-accelerated** inference for fast and accurate prediction ⚡\n"
        "✅ Works with **RGB images** and highlights **disease-affected regions** 🌾\n\n"
        
        "**💡 Interesting Facts About Leaf Diseases:**\n"
        "🔹 Leaf diseases can drastically reduce crop yield if not detected early. 🚜\n"
        "🔹 Early detection helps reduce pesticide use, saving costs and the environment. 🌍\n"
        "🔹 Image segmentation helps **pinpoint diseased spots**, allowing for precision agriculture. 📍\n"
        "🔹 Common diseases include **blight, mildew, and rust**, each with unique visual symptoms. 🧫\n\n"

        "📸 *Send a clear image of a leaf to start segmentation and identify diseased regions!*"
    )
    await update.message.reply_text(about_text)


# Handle Photo Messages
async def handle_photo(update: Update, context: CallbackContext):
        photo = update.message.photo[-1:][0]  # Process the highest resolution photo
        file = await photo.get_file()
        image_bytes = BytesIO(await file.download_as_bytearray())
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert NumPy array to PIL image
        image = Image.fromarray(image)
        
        image_with_segment = object_detection_image(image)

        # Send Detected image with boxes result
        image_with_segment.seek(0)  # Reset file pointer before sending    
        await update.message.reply_photo(photo=image_with_segment, caption=f"Leaf Disease Segmentation Result!")


# Main Function
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
