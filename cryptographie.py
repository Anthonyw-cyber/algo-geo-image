import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from PIL import Image

def generate_voronoi_crypto(image_path, n_points):
    # Chargement de l'image
    image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
    image_array = np.array(image)


    # Générer uniformément et aléatoirement n points sur la surface de l'image
    points = np.random.rand(n_points, 2) * np.array(image.size)

    # Calculer le diagramme de Voronoi
    vor = Voronoi(points)

    # Initialiser les masques noir et blanc
    mask1 = np.zeros_like(image_array)
    print('mask1', mask1)
    for tab in mask1:
        for i in range(len(tab)):
            tab[i] = 255
    mask2 = np.zeros_like(image_array)
    for tab in mask2:
        for i in range(len(tab)):
            tab[i] = 255

    def calculate_voronoi_centroid(region_vertices):
        n = len(region_vertices)

        if n < 3:
            return None  # La région doit avoir au moins 3 sommets pour former un polygone

        area = 0.0
        cx = 0.0
        cy = 0.0

        for i in range(n - 1):
            xi, yi = region_vertices[i]
            xi1, yi1 = region_vertices[i + 1]

            ai = xi * yi1 - xi1 * yi
            area += ai
            cx += (xi + xi1) * ai
            cy += (yi + yi1) * ai

        area *= 0.5
        cx /= (6 * area)
        cy /= (6 * area)

        return cx, cy

    # Calculer les centroïdes des régions de Voronoi
    voronoi_centroids = []
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            centroid = calculate_voronoi_centroid(polygon)
            if centroid is not None:
                voronoi_centroids.append(centroid)

    voronoi_centroids = np.array(voronoi_centroids)

    # Calculer les proportions et attribuer les masques
    for region, centroid in zip(vor.regions, voronoi_centroids):
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            mask_points = np.array(polygon, dtype=np.int32)

            # Vérifier que les indices restent dans les limites de l'image
            valid_indices = np.logical_and(mask_points[:, 0] >= 0, mask_points[:, 0] < image_array.shape[1])
            valid_indices = np.logical_and(valid_indices, np.logical_and(mask_points[:, 1] >= 0, mask_points[:, 1] < image_array.shape[0]))

            # Vérifier si le pixel est blanc dans l'image d'origine
            is_white_pixel = image_array[mask_points[valid_indices][:, 1], mask_points[valid_indices][:, 0]] == 255

            # Calculer la position par rapport au centroïde
            is_left_of_centroid = mask_points[valid_indices][:, 0] <= centroid[0]

            # Assigner les masques
            mask1[mask_points[valid_indices][:, 1], mask_points[valid_indices][:, 0]] = ~is_left_of_centroid
            mask2[mask_points[valid_indices][:, 1], mask_points[valid_indices][:, 0]] = is_left_of_centroid


    # Appliquer les masques sur l'image entière
    result_image = (image_array * mask1) + (255 * mask2)

    # Afficher les masques noir et blanc séparément
    plt.imshow(mask1, cmap="gray")
    plt.axis("off")
    plt.title("Masque 1")
    plt.show()

    plt.imshow(mask2, cmap="gray")
    plt.axis("off")
    plt.title("Masque 2")
    plt.show()

    # Afficher l'image résultante
    plt.imshow(result_image, cmap="gray")
    plt.axis("off")
    plt.title("Image Révélée")
    plt.show()

# Utilisation du script avec un nombre spécifique de points
image_path = "testcrupto.png"  # Remplace ça par le chemin de ton image
nombre_points = 4000  # Demande à l'utilisateur

generate_voronoi_crypto(image_path, nombre_points)
