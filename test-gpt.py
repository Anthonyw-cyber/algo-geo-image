import cv2
import numpy as np
import random

def generateRandomPoints(image, percentage):
    num_pixels = image.size
    n = int(percentage * num_pixels / 1000)

    # Generate random coordinates within the valid range of the image
    rows = np.random.randint(0, image.shape[0], n)
    cols = np.random.randint(0, image.shape[1], n)

    return list(zip(rows, cols))

def calculateRegionVariances(image, voronoiZoneNumber, n):
    region_variances = np.zeros(n + 1)

    for zone in range(1, n + 1):
        region_pixels = image[voronoiZoneNumber == zone]
        variance = np.var(region_pixels)
        region_variances[zone] = variance

    return region_variances

def identifyHomogeneousRegions(variances, threshold):
    homogeneous_regions = np.where(variances < threshold)[0]
    return homogeneous_regions

def adaptiveVoronoi(imgOriginal, nbGerme, initial_percentage, variance_threshold, max_iterations):
    img_shape = imgOriginal.shape[:2]

    # Generate initial random points
    random_points = generateRandomPoints(imgOriginal, initial_percentage)

    for iteration in range(max_iterations):
        voronoiZoneNumber = np.zeros(img_shape, dtype=int)

        # Assign each pixel to the nearest point
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                distances = np.linalg.norm(np.array(random_points) - np.array([i, j]), axis=1)
                voronoiZoneNumber[i, j] = np.argmin(distances) + 1

        # Calculate variances for each region
        region_variances = calculateRegionVariances(imgOriginal, voronoiZoneNumber, nbGerme)

        # Identify homogeneous regions
        homogeneous_regions = identifyHomogeneousRegions(region_variances, variance_threshold)

        # Add new points to homogeneous regions
        new_points = generateRandomPoints(imgOriginal, initial_percentage)
        random_points.extend(new_points)

        print(f"Iteration {iteration + 1}: {len(random_points)} points")

    return voronoiZoneNumber

# Example usage
imgOriginal = cv2.imread("Lenna.png", 1)
if imgOriginal is None:
    print("error: image not read from file\n\n")
    exit()

nbGerme = 50
initial_percentage = 0.1
variance_threshold = 1000
max_iterations = 5

adaptive_voronoi_result = adaptiveVoronoi(imgOriginal, nbGerme, initial_percentage, variance_threshold, max_iterations)

# Visualize the result or perform further processing
cv2.imshow("Adaptive Voronoi Result", adaptive_voronoi_result)
cv2.waitKey(0)
cv2.destroyAllWindows()