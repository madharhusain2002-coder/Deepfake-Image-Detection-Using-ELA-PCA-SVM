import cv2
import numpy as np
from ela import convert_to_ela_image

def main():
    image_path = "sample.jpg"

    ela_image = convert_to_ela_image(image_path)

    cv2.imshow("ELA Output", ela_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
