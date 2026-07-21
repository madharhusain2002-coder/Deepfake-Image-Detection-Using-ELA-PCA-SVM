from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2

def convert_to_ela_image(path, quality=90):
    original = Image.open(path).convert("RGB")

    temp_filename = "temp.jpg"
    original.save(temp_filename, "JPEG", quality=quality)

    compressed = Image.open(temp_filename)

    ela_image = ImageChops.difference(original, compressed)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    ela_array = np.array(ela_image)
    ela_array = cv2.cvtColor(ela_array, cv2.COLOR_RGB2BGR)

    return ela_array
