import cv2
import numpy as np
import random
import math

# INITIALISATION
imgName = "Lenna.png"
nbGerme = 525


def generateDiscreteVoronoi(imagedist, voronoiZoneNumber, d_height, d_width):
    distanceFromClosestSite = np.zeros((d_height, d_width))
    for i in range(d_height):
        for j in range(d_width):
            distanceFromClosestSite[i][j] = float(imagedist[i][j])

        sqrt2 = math.sqrt(2)

    for i in range(d_height):
        for j in range(d_width):
            if i > 0:
                if j > 0:
                    if distanceFromClosestSite[i][j] > distanceFromClosestSite[i - 1][j - 1] + sqrt2:
                        distanceFromClosestSite[i][j] = distanceFromClosestSite[i - 1][j - 1] + sqrt2
                        voronoiZoneNumber[i][j] = voronoiZoneNumber[i - 1][j - 1]
                if distanceFromClosestSite[i][j] > distanceFromClosestSite[i - 1][j] + 1:
                    distanceFromClosestSite[i][j] = distanceFromClosestSite[i - 1][j] + 1
                    voronoiZoneNumber[i][j] = voronoiZoneNumber[i - 1][j]
                if j < d_width - 1:
                    if distanceFromClosestSite[i][j] > distanceFromClosestSite[i - 1][j + 1] + sqrt2:
                        distanceFromClosestSite[i][j] = distanceFromClosestSite[i - 1][j + 1] + sqrt2
                        voronoiZoneNumber[i][j] = voronoiZoneNumber[i - 1][j + 1]
            if j > 0:
                if distanceFromClosestSite[i][j] > distanceFromClosestSite[i][j - 1] + 1:
                    distanceFromClosestSite[i][j] = distanceFromClosestSite[i][j - 1] + 1
                    voronoiZoneNumber[i][j] = voronoiZoneNumber[i][j - 1]

    for i in range(d_height - 1, -1, -1):
        for j in range(d_width - 1, -1, -1):
            if i < d_height - 1:
                if j < d_width - 1:
                    if distanceFromClosestSite[i][j] > distanceFromClosestSite[i + 1][j + 1] + sqrt2:
                        distanceFromClosestSite[i][j] = distanceFromClosestSite[i + 1][j + 1] + sqrt2
                        voronoiZoneNumber[i][j] = voronoiZoneNumber[i + 1][j + 1]
                if distanceFromClosestSite[i][j] > distanceFromClosestSite[i + 1][j] + 1:
                    distanceFromClosestSite[i][j] = distanceFromClosestSite[i + 1][j] + 1
                    voronoiZoneNumber[i][j] = voronoiZoneNumber[i + 1][j]
                if j > 0:
                    if distanceFromClosestSite[i][j] > distanceFromClosestSite[i + 1][j - 1] + sqrt2:
                        distanceFromClosestSite[i][j] = distanceFromClosestSite[i + 1][j - 1] + sqrt2
                        voronoiZoneNumber[i][j] = voronoiZoneNumber[i + 1][j - 1]
            if j < d_width - 1:
                if distanceFromClosestSite[i][j] > distanceFromClosestSite[i][j + 1] + 1:
                    distanceFromClosestSite[i][j] = distanceFromClosestSite[i][j + 1] + 1
                    voronoiZoneNumber[i][j] = voronoiZoneNumber[i][j + 1]


def CalculateAverageColors(imageOriginal, voronoiZoneNumber, n):
    zoneColorSum = np.zeros((n + 1, 3))
    zonePixelCount = np.zeros(n + 1)

    for i in range(imageOriginal.shape[0]):
        for j in range(imageOriginal.shape[1]):
            # Récupérer couleur du pixel
            pixelColor = imageOriginal[i, j]
            # Récuperer l'emplacement du pixel
            zone = voronoiZoneNumber[i][j]

            zoneColorSum[zone] += pixelColor
            zonePixelCount[zone] += 1
    # Calculer la moyenne des couleurs pour chaque zone
    zoneColors = np.zeros((n + 1, 3), dtype=int)
    for zone in range(1, n + 1):
        if zonePixelCount[zone] > 0:
            zoneColors[zone] = (zoneColorSum[zone] / zonePixelCount[zone]).astype(int)
    return zoneColors
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

def VorDiscret(voronoi, n, imgOriginals):
    random.seed()
    nombreEchantillons = 0
    nbech = 0

    gx = 0
    gy = 0
    r = 0
    r0 = 0

    theta = 0
    xr = 0
    yr = 0
    B = False

    nombreLignes = voronoi.shape[0]
    nombreColonnes = voronoi.shape[1]
    voronoiZoneNumber = np.zeros((nombreLignes, nombreColonnes), dtype=int)
    imagedist = np.zeros((nombreLignes, nombreColonnes), dtype=int)

    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            voronoiZoneNumber[i][j] = 0
            imagedist[i][j] = nombreLignes + nombreColonnes

    cpt = 1
    a = 0
    b = 0
    germeX = []
    germeY = []
    while cpt < n:
        a = random.randint(0, nombreLignes - 1)
        b = random.randint(0, nombreColonnes - 1)
        voronoiZoneNumber[a][b] = cpt
        cpt = cpt + 1

        germeX.append(a)
        germeY.append(b)

    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            if voronoiZoneNumber[i][j] != 0:
                imagedist[i][j] = 0

    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            if imagedist[i][j] != 0:
                voronoiZoneNumber[i][j] = 0
                imagedist[i][j] = nombreLignes + nombreColonnes

    generateDiscreteVoronoi(imagedist, adaptiveVoronoi(imagedist, n, 0.1, 1000, 1), nombreLignes, nombreColonnes)

   # CL = np.zeros((n + 1, 3), dtype=int)
    #for i in range(n + 1):
    #    CL[i][0] = random.randint(50, 249)
    #    CL[i][1] = random.randint(55, 254)
    #    CL[i][2] = random.randint(50, 249)
    zoneColors = CalculateAverageColors(imgOriginals, voronoiZoneNumber, n)
    # Colorer par zones couleurs aleatoire
    # for i in range(nombreLignes):
    #     for j in range(nombreColonnes):
    #         voronoi[i, j][0] = CL[voronoiZoneNumber[i][j]][0]
    #         voronoi[i, j][1] = CL[voronoiZoneNumber[i][j]][1]
    #         voronoi[i, j][2] = CL[voronoiZoneNumber[i][j]][2]
    print(zoneColors[voronoiZoneNumber[1, 1]])
    print(voronoi[1, 1])
    #Couleur moyenne
    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            voronoi[i, j][0] = zoneColors[voronoiZoneNumber[i, j]][0]
            voronoi[i, j][1] = zoneColors[voronoiZoneNumber[i, j]][1]
            voronoi[i, j][2] = zoneColors[voronoiZoneNumber[i, j]][2]



def generateRandomPoints(imgOriginal, initial_percentage):
    img_shape = imgOriginal.shape[:2]
    nb_points = int(initial_percentage * img_shape[0] * img_shape[1])
    random_points = []
    for i in range(nb_points):
        x = random.randint(0, img_shape[0] - 1)
        y = random.randint(0, img_shape[1] - 1)
        random_points.append([x, y])
    return random_points
#implementation de l'algorithme de voronoi adaptatif
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



imgOriginal = cv2.imread(imgName, 1)
if imgOriginal is None:
    print("error: image not read from file\n\n")
    exit()

colonnes = imgOriginal.shape[1]
lignes = imgOriginal.shape[0]

voronoi = np.zeros((imgOriginal.shape[0], imgOriginal.shape[1], 3), dtype=np.uint8)
VorDiscret(voronoi, nbGerme, imgOriginal)


path1 = "res.png"
cv2.imwrite(path1, voronoi)

cv2.namedWindow("resultat", cv2.WINDOW_AUTOSIZE)
cv2.imshow("resultat", voronoi)

cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)
cv2.imshow("imgOriginal", imgOriginal)

cv2.waitKey(0)
cv2.destroyAllWindows()
