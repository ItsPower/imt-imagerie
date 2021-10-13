# Imports nécessaires pour ce TP
import csv
import os

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

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    # Charger l'image
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
        cv2.rectangle(result, (x-10,y-10), (x+w+10,y+h+10), (0,255,0), 1)
        cv2.putText(result, "face "+str(i), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

        # Enfin on boucle sur tout nos coordonnées de points d'interets
        # et on les ajoute à notre image de base avec un petit cercle pour être visible
        j = 0
        landmarks = [None] * 100
        for (x, y) in forme:
            landmarks[j] = (x, y)
            #print(j,",",x,",",y)
            #cv2.circle(result, (x, y), 1, (0, 0, 0), 1)
            #cv2.putText(result, str(j+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255))
            j = j + 1

# getting all files of mask folder
        for file in os.listdir("masks"):
# verifying existence of csv and png names files
            if file.endswith(".csv") and os.path.isfile("masks/"+file.split(".")[0]+".png"):
                #print("file: "+file+" has csv and image")

                mask_csv = "masks/"+file;
                mask_image = "masks/"+file.split(".")[0]+".png";

# openning csv file: landmark,src_x,src_y,translation_x,translation_y
                with open(mask_csv) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    tmp_landmarks = []
                    src_pts = []
                    dst_translation = []
                    for i, row in enumerate(csv_reader):
                        # Si présence de ligne vide alors on passe à la suite
                        try:
                            tmp_landmarks.append(forme[int(row[0])-1])
                            #print(mask_csv,"added landmark ",float(row[0])-1)
                            src_pts.append(np.array([float(row[1]), float(row[2])]))
                            #print(mask_csv,"added src_pt ",float(row[1]), float(row[2]))
                            dst_translation.append((float(row[3]), float(row[4])))
                            #print(mask_csv,"added dst_translation ",float(row[3]), float(row[4]))
                        except ValueError:
                            continue
                src_pts = np.array(src_pts, dtype="float32")
                dst_pts = np.array(tmp_landmarks, dtype="float32")

                # On applique une superposition si chaque coordonnées à une correspondance

                if (forme > 0).all():
                    # NEZ
                    for idx, val in enumerate(dst_translation):
                        dst_pts[idx][0] += val[0]
                        dst_pts[idx][1] += val[1]

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

                    alpha_mask = transformed_mask[:, :, 3]
                    alpha_image = 1.0 - alpha_mask
                    for c in range(0, 3):
                        result[:, :, c] = (
                                alpha_mask * transformed_mask[:, :, c]
                                + alpha_image * result[:, :, c]
                        )

                    #cv2.imshow('transformed_mask', transformed_mask)
                    #cv2.imshow('alpha_mask', alpha_mask)
                    # END OF NEZ

    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()