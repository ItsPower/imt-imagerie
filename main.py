# Imports nécessaires pour ce TP
import csv

import args as args
from imutils import face_utils
import numpy as np
import argparse
# Bien vérifier que vous avez votre package imutils à jour
import imutils
import dlib
import cv2

# On initialise un module de detection de visage dlib
# puis on lui assigne notre prédicteur
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Charger l'image

cap = cv2.VideoCapture(0)
ret, image = cap.read()

#image = cv2.imread("exemple1.jpg")
image = imutils.resize(image, width=500)
result = image.copy()
result = result.astype(np.float32) / 255.0
# puis la passer en niveau de gris
# Rappel : Preprocessing en niveau de gris pour adherer au model
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# On applique la detection de visage sur notre image en niveau de gris
gray_rec = detector(gray, 1)

# On boucle sur la detection
for (i, rect) in enumerate(gray_rec):
    # Apres avoir determiner les reperes sur le visage
    # On doit convertir les coordonnées des reperes dans
    # un tableau numpy afin de pouvoir les réutilisers pour la partie suivante du TP
    forme = predictor(gray, rect)
    forme = face_utils.shape_to_np(forme)
    # Pour aider à l'affichage on peut ajouter un rectangle
    # sur notre image au niveau de la zone de detection du visage
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    # Il faut alors récupérer les coordonnées de la fonction rectangle
    # de dlib pour la convertir en rectangle au format opencv
    # Dans le cas on nous ajoutons une rêgle de traitement relatif au différents visages
    # nous pouvons les annoter si plusieurs visages

    # Enfin on boucle sur tout nos coordonnées de points d'interets
    # et on les ajoute à notre image de base avec un petit cercle pour être visible
    j = 0
    landmarks = [None] * 100
    for (x, y) in forme:
        print(j)
        landmarks[j] = (x, y)

        j=j+1

    # On enregistre notre résultat de détection
    cv2.imwrite("result.jpg",image)

    dst_pts = np.array(
        [
            forme[48],
            forme[49],
            forme[50],
            forme[51],
            forme[52],
            forme[53],
            forme[54],
            forme[55],
            forme[56],
            forme[57],
            forme[58],
            forme[59],
        ],
        dtype="float32",
    )

    mask_image = "masks/bouche.png"
    mask_points = "masks/bouche.csv"

    with open(mask_points) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        src_pts = []
        for i, row in enumerate(csv_reader):
            # Si présence de ligne vide alors on passe à la suite
            try:
                src_pts.append(np.array([float(row[1]), float(row[2])]))
            except ValueError:
                continue
    src_pts = np.array(src_pts, dtype="float32")


# On applique une superposition si chaque coordonnées à une correspondance
if (forme > 0).all():
    print(">0")
    # Chargement de l'image
    mask_img = cv2.imread(mask_image, cv2.IMREAD_UNCHANGED)
    mask_img = mask_img.astype(np.float32)
    mask_img = mask_img / 255.0
    # On créer une matrice de transformation (perspective)
    M, _ = cv2.findHomography(src_pts, dst_pts)
    # On applique la transformation
    transformed_mask = cv2.warpPerspective(
        mask_img,
        M,
        (result.shape[1], result.shape[0]),
        None,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
    )
    cv2.imshow("Image superposée simple", transformed_mask)
    alpha_mask = transformed_mask[:, :, 3]
    alpha_image = 1.0 - alpha_mask
    for c in range(0, 3):
        result[:, :, c] = (
            alpha_mask * transformed_mask[:, :, c]
            + alpha_image * result[:, :, c]
        )
    cv2.imshow("Image", result)
cv2.waitKey(0)