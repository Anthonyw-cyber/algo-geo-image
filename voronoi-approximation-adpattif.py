import numpy as np
from scipy.spatial import Voronoi
import cv2


def adaptive_voronoi(image_path, initial_percentage, threshold_metric):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    height, width, _ = img.shape
    num_initial_points = round((height * width) * initial_percentage / 100)

    # Génération initiale de points
    initial_points = np.random.rand(num_initial_points, 2) * np.array([width, height])

    # Affichage de l'image initiale
    cv2.imshow('Image Initiale', img)
    cv2.waitKey(0)

    iteration = 0

    while True:
        # Calcul du diagramme de Voronoi
        vor = Voronoi(initial_points)

        # Affichage du diagramme de Voronoi sur l'image
        img_voronoi = img.copy()
        for region_index, region in enumerate(vor.regions):
            if -1 not in region and len(region) > 0 and region_index in vor.point_region:
                region_pixels = img_voronoi[region[1:], region[0]]
                mean_color = np.mean(region_pixels, axis=0)
                img_voronoi[region[1:], region[0]] = mean_color.astype(int)
        cv2.imshow(f'Diagramme de Voronoi - Itération {iteration}', img_voronoi)
        cv2.waitKey(0)

        # Évaluation des régions
        regions_metrics = []
        for region_index, region in enumerate(vor.regions):
            if -1 not in region and len(region) > 0 and region_index in vor.point_region:
                region_pixels = img[region[1:], region[0], 0]
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

        # Affichage des nouveaux points
        img_points = img.copy()
        for point in initial_points:
            cv2.circle(img_points, tuple(point.astype(int)), 3, (255, 0, 0), -1)
        cv2.imshow(f'Nouveaux Points - Itération {iteration}', img_points)
        cv2.waitKey(0)

        iteration += 1

    # Affichage de l'image finale
    cv2.imshow('Image Finale', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemple d'utilisation
print("Exemple d'utilisation")
adaptive_voronoi("./zelda.png", initial_percentage=0.2, threshold_metric=100)
