import os
import numpy as np
from scipy.spatial import Voronoi
import cv2

def adaptive_voronoi(image_path, initial_percentage, threshold_metric, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    num_initial_points = round((height * width) * initial_percentage / 100)

    # Génération initiale de points
    initial_points = np.random.rand(num_initial_points, 2) * np.array([width, height])

    # Création du dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    iteration = 0

    while True:
        # Calcul du diagramme de Voronoi
        vor = Voronoi(initial_points)

        # Affichage du diagramme de Voronoi sur l'image
        img_voronoi = img.copy()
        for region in vor.regions:
            if region and -1 not in region and len(region) > 0:
                region_pixels = img_voronoi[
                    np.clip(region[1:], 0, height - 1),
                    np.clip(region[0], 0, width - 1)
                ]
                mean_color = np.mean(region_pixels, axis=0)
                img_voronoi[
                    np.clip(region[1:], 0, height - 1),
                    np.clip(region[0], 0, width - 1)
                ] = mean_color.astype(int)

        # Sauvegarde de l'image
        img_filename = os.path.join(output_folder, f'voronoi_iteration_{iteration}.png')
        cv2.imwrite(img_filename, img_voronoi)

        # Évaluation des régions
        regions_metrics = []
        for region in vor.regions:
            if region and -1 not in region and len(region) > 0:
                region_pixels = img[
                    np.clip(region[1], 0, height - 1),
                    np.clip(region[0], 0, width - 1),
                    0
                ]
                metric = np.var(region_pixels)
                regions_metrics.append(metric)

        # Trouver la région avec la métrique la plus faible
        min_metric_region = np.argmin(regions_metrics)

        # Condition d'arrêt si toutes les régions satisfont le critère ou si aucune nouvelle région n'est ajoutée de manière significative
        if np.min(regions_metrics) >= threshold_metric:
            break

        # Ajout d'un nouveau point dans la région avec la métrique la plus faible
        new_point = np.random.rand(1, 2) * np.array([width, height])
        initial_points = np.vstack((initial_points, new_point))

        # Sauvegarde de l'image avec les nouveaux points
        img_points = img.copy()
        for point in initial_points:
            cv2.circle(img_points, tuple(point.astype(int)), 3, (255, 0, 0), -1)
        img_filename = os.path.join(output_folder, f'points_iteration_{iteration}.png')
        cv2.imwrite(img_filename, img_points)

        iteration += 1

    cv2.destroyAllWindows()

# Exemple d'utilisation en spécifiant le dossier de sortie
output_folder_path = "approx-adaptatif"
adaptive_voronoi("./zelda.png", initial_percentage=0.2, threshold_metric=100, output_folder=output_folder_path)
