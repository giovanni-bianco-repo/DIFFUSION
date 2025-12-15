# Download images for style training (Textual Inversion)
import os
import requests
from PIL import Image
from io import BytesIO


# Cubism style: working public domain paintings
urls = [
    "https://upload.wikimedia.org/wikipedia/commons/4/4c/Pablo_Picasso%2C_1907%2C_Les_Demoiselles_d%27Avignon%2C_oil_on_canvas%2C_243.9_x_233.7_cm%2C_MoMA%2C_New_York.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/3a/Georges_Braque%2C_1909%2C_Houses_at_L%27Estaque%2C_oil_on_canvas%2C_73_x_60_cm%2C_Musee_National_d%27Art_Moderne%2C_Paris.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/7/7d/Juan_Gris%2C_1915%2C_Still_Life_with_Fruit_Dish_and_Mandolin%2C_oil_on_canvas%2C_92.1_x_73_cm%2C_Metropolitan_Museum_of_Art.jpg"
]

save_path = "./inputs_textual_inversion"
os.makedirs(save_path, exist_ok=True)


def download_image(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

images = list(filter(None, [download_image(url) for url in urls]))
for i, image in enumerate(images):
    image.save(f"{save_path}/{i}.jpeg")

print(f"Downloaded {len(images)} images to {save_path}")
