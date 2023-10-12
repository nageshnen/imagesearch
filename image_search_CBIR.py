import cv2
import numpy as np
import os
import glob
class CBIR:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = []
        self.index = {}

        # Load the dataset
        for image_path in glob.glob(os.path.join(self.dataset_path, "*.jpg")):
            image = cv2.imread(image_path)
            self.dataset.append(image)

            # Extract the color histogram of the image
            color_histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

            # Add the image to the index
            self.index[image_path] = color_histogram

    def retrieve_similar_images(self, query_image):
        # Extract the color histogram of the query image
        query_color_histogram = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Calculate the distance between the query image histogram and all the histograms in the index
        distances = {}
        for image_path, image_histogram in self.index.items():
            distance = cv2.compareHist(query_color_histogram, image_histogram, cv2.HISTCMP_CORREL)
            distances[image_path] = distance

        # Sort the distances in descending order
        sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)

        # Return the top-10 most similar images
        return sorted_distances[:10]

if __name__ == "__main__":
    # Create a CBIR object
    cbir = CBIR("dataset")

    # Load the query image
    query_image = cv2.imread("query_image.jpg")

    # Retrieve the similar images
    similar_images = cbir.retrieve_similar_images(query_image)

    # Display the similar images
    for image_path, distance in similar_images:
        cv2.imshow("Similar Image", cv2.imread(image_path))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
