from enum import Enum

import cv2
import numpy as np
import random
import math

class Mode(Enum):
    NORMAL = 0
    EUCLIDIENNE = 1
    ABSOLUE = 2
    INFINI = 3

# INITIALISATION
imgName = "zelda.png"
nbGerme = 5000
mode = Mode.NORMAL
nomMode = "NORMAL"

print("OpenCV version: " + cv2.__version__)

def Min(Px, Py, GermeX, GermeY, mode):
    m = 1
    D = dis(Px, Py, GermeX[1], GermeY[1], mode)

    for i in range(len(GermeX)):
        d = dis(Px, Py, GermeX[i], GermeY[i], mode)
        if (d < D):
            m = i
            D = d

    return m

def dis(Px, Py, Qx, Qy, mode):
    if mode == Mode.EUCLIDIENNE:
        return math.sqrt((Px - Qx)**2 + (Py - Qy)**2)
    if mode == Mode.ABSOLUE:
        return abs(Px - Qx) + abs(Py - Qy)
    if mode == Mode.INFINI:
        return max(abs(Px - Qx), abs(Py - Qy))

def generateDiscreteVoronoi(imagedist, voronoiZoneNumber, d_height, d_width, germeX, germeY, mode):
    distanceFromClosestSite = np.zeros((d_height, d_width))
    for i in range(d_height):
        for j in range(d_width):
            if mode == Mode.NORMAL:
                distanceFromClosestSite[i][j] = float(imagedist[i][j])
            if mode != Mode.NORMAL:
                voronoiZoneNumber[i][j] = Min(i, j, germeX, germeY, mode)

    if mode == Mode.NORMAL:
        sqrt2 = math.sqrt(2)

        # Masque appliqué (racine 2) 1er balayage
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

        # Masque appliqué (racine 2) 2eme balayage
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

# Calculer couleurs moyenne d'une zone de Voronoi
def calculateAverageColor(image, voronoiZoneNumber, n):
    # Initialiser un tableau pour stocker la somme des couleurs et le nombre de pixels dans chaque zone
    zoneColorSum = np.zeros((n + 1, 3))
    zonePixelCounts = np.zeros(n + 1)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Récupérer la couleur du pixel de l'image originale
            pixelColor = image[i, j]

            # Récupérer le numéro de zone du pixel dans l'image Voronoi
            zone = voronoiZoneNumber[i][j]

            zoneColorSum[zone] += pixelColor
            zonePixelCounts[zone] += 1

    # Calculer la moyenne des couleurs pour chaque zone
    zoneColors = np.zeros((n + 1, 3), dtype=int)
    for zone in range(1, n + 1):
        if zonePixelCounts[zone] > 0:
            zoneColors[zone] = (zoneColorSum[zone] / zonePixelCounts[zone]).astype(int)

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
def VorDiscret(voronoi, n, imgOriginal, mode):
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

    # voronoiZone = 0 et imageDist = infini (valeur inatteignable)
    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            voronoiZoneNumber[i][j] = 0
            imagedist[i][j] = nombreLignes + nombreColonnes

    cpt = 1
    a = 0
    b = 0
    germeX = []
    germeY = []

    # Germes aléatoires
    while cpt < n:
        a = random.randint(0, nombreLignes - 1)
        b = random.randint(0, nombreColonnes - 1)

        germeX.append(a)
        germeY.append(b)

        voronoiZoneNumber[a][b] = cpt
        cpt = cpt + 1

    # imagedist = 0 la ou il y a un germe
    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            if voronoiZoneNumber[i][j] != 0:
                imagedist[i][j] = 0

    # voromne = 0 si imagedist n'a pas de germe
    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            if imagedist[i][j] != 0:
                voronoiZoneNumber[i][j] = 0
                imagedist[i][j] = nombreLignes + nombreColonnes

    generateDiscreteVoronoi(imagedist, voronoiZoneNumber, nombreLignes, nombreColonnes, germeX, germeY, mode)

    # Couleurs
    # CL = np.zeros((n + 1, 3), dtype=int)
    # for i in range(n + 1):
    #     CL[i][0] = random.randint(50, 249)
    #     CL[i][1] = random.randint(55, 254)
    #     CL[i][2] = random.randint(50, 249)

    zoneColors = calculateAverageColor(imgOriginal, voronoiZoneNumber, n)

    # Colorer par zones couleurs aleatoires
    # for i in range(nombreLignes):
    #     for j in range(nombreColonnes):
    #         voronoi[i, j][0] = CL[voronoiZoneNumber[i][j]][0]
    #         voronoi[i, j][1] = CL[voronoiZoneNumber[i][j]][1]
    #         voronoi[i, j][2] = CL[voronoiZoneNumber[i][j]][2]

    # Colorer par zones couleurs moyennes
    for i in range(nombreLignes):
        for j in range(nombreColonnes):
            voronoi[i, j][0] = zoneColors[voronoiZoneNumber[i][j]][0]
            voronoi[i, j][1] = zoneColors[voronoiZoneNumber[i][j]][1]
            voronoi[i, j][2] = zoneColors[voronoiZoneNumber[i][j]][2]

# MODE CHOISI
if mode == Mode.EUCLIDIENNE:
    nomMode = "EUCLIDIENNE"
if mode == Mode.ABSOLUE:
    nomMode = "ABSOLUE"
if mode == Mode.INFINI:
    nomMode = "INFINI"
print(nomMode)

# QR code reconnu a partir de 8000 germes

imgOriginal = cv2.imread(imgName, 1)
if imgOriginal is None:
    print("error: image not read from file\n\n")
    exit()

print(imgOriginal)

colonnes = imgOriginal.shape[1]
lignes = imgOriginal.shape[0]

# Tableau de zeros
voronoi = np.zeros((imgOriginal.shape[0], imgOriginal.shape[1], 3), dtype=np.uint8)
VorDiscret(voronoi, nbGerme, imgOriginal, mode)

# Sauvegarde
path1 = nomMode + "-" + str(nbGerme) + "-" + "zelda-result.png"
cv2.imwrite("algo_geo/images-result/" + path1, voronoi)
print("Image enregistrée: " + "algo_geo/images-result/" + path1)

# Montrer résultat
cv2.namedWindow("resultat", cv2.WINDOW_AUTOSIZE)
cv2.imshow("resultat", voronoi)

# Montrer original
cv2.namedWindow("imgOriginal", cv2.WINDOW_AUTOSIZE)
cv2.imshow("imgOriginal", imgOriginal)

cv2.waitKey(0)
cv2.destroyAllWindows()