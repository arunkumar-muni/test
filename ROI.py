import cv2
import numpy as np

def main():
    image_path = r"/home/arun/Downloads/images (1).jpeg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load image.")
        return

    print("Select the roi.")
    r= cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)

    cropped_region = image[int(r[1]):int(r[1] + r[3]),
                    int(r[0]):int(r[0] + r[2])]

    cv2.imshow("Cropped Region", cropped_region)

    output_path = 'cropped_region.jpg'
    cv2.imwrite(output_path, cropped_region)
    print(f"Cropped region saved as {output_path}.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
