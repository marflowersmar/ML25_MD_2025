import os
import cv2
import torch
import pathlib

from ml25.P02_facial_expressions.network import Network
from ml25.P02_facial_expressions.utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from ml25.P02_facial_expressions.dataset import EMOTIONS_MAP

file_path = pathlib.Path(__file__).parent.absolute()

MODEL_NAME = "modelo_1.pt"


def detect_face(img):
    """
    Detecta un rostro con Haar Cascade.
    Si no encuentra, regresa la imagen completa.
    """
    if img is None:
        raise ValueError("Imagen vacía en detect_face")

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y : y + h, x : x + w]
    else:
        print("No se detectó rostro, usando imagen completa.")
        return img


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"No se pudo leer la imagen {path}")

    face_img = detect_face(img)

    val_transforms, unnormalize = get_transforms("test", img_size=48)
    tensor_img = val_transforms(face_img)            # (1,48,48)
    denormalized = unnormalize(tensor_img.clone())

    if tensor_img.ndim == 3:
        tensor_img = tensor_img.unsqueeze(0)         # (1,1,48,48)

    return face_img, tensor_img, denormalized


def predict(img_paths):
    modelo = Network(input_dim=48, n_classes=7)
    modelo.load_model(MODEL_NAME)

    device = modelo.device

    for path in img_paths:
        full_path = (file_path / path).as_posix()
        print(f"Procesando: {full_path}")

        try:
            original_crop, transformed, denormalized = load_img(full_path)
            transformed = transformed.to(device)

            logits, proba = modelo.predict(transformed)
            pred_idx = torch.argmax(proba, dim=1).item()
            pred_label = EMOTIONS_MAP[pred_idx]
            confidence = proba[0, pred_idx].item() * 100

            # Visualización
            h, w = original_crop.shape[:2]
            resize_h = 300
            resize_w = w * resize_h // h
            disp_img = cv2.resize(original_crop, (resize_w, resize_h))
            disp_img = add_img_text(disp_img, f"{pred_label}: {confidence:.1f}%")

            denorm_np = to_numpy(denormalized)
            denorm_np = cv2.resize(denorm_np, (resize_h, resize_h))

            cv2.imshow("Recorte + predicción", disp_img)
            cv2.imshow("Input a la red (denormalizado)", denorm_np)

            if cv2.waitKey(0) == ord("q"):
                break

        except Exception as e:
            print(f"Error con {full_path}: {e}")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_paths = [
        "./test_imgs/DANIA1.jpg",
        "./test_imgs/DANIA2.jpg",
        "./test_imgs/DANIA3.jpg",
        #"./test_imgs/MARIANA1.jpg",
        #"./test_imgs/MARIANA 3.jpg",
        #"./test_imgs/angry.jpg",
        #"./test_imgs/angry2.webp",
        #"./test_imgs/disgusted.jpg",
        #"./test_imgs/disgusted2.jpg",
        #"./test_imgs/disgusted3.jpg",
        #"./test_imgs/disgusted4.jpg",
        #"./test_imgs/fear.jpg",
        #"./test_imgs/fear2.jpg",
        #"./test_imgs/fear3.jpg",
        #"./test_imgs/happy.jpg",
        #"./test_imgs/happy.png",
        #"./test_imgs/happy2.jpg",
        #"./test_imgs/happy3.webp",
        #"./test_imgs/happy4.jpg",
        #"./test_imgs/Neutral.avif",
        #"./test_imgs/neutral2.webp",
        #"./test_imgs/neutral3.webp",
        #"./test_imgs/sad.jpg",
        #"./test_imgs/sad2.jpg",
        #"./test_imgs/sad3.avif",
        #"./test_imgs/sad4.jpg",
        #"./test_imgs/surprised.jpg",
        #"./test_imgs/surprised2.jpg",
        #"./test_imgs/surprised3.jpg",
    ]
    predict(img_paths)
