import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.io import imread
import cv2
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# --- Important Definitions ---

# URLs dos arquivos CSV no GitHub
url_train = "https://raw.githubusercontent.com/antunesbrunoo/Processamento-Digital-de-Imagens/main/train.csv"
url_test = "https://raw.githubusercontent.com/antunesbrunoo/Processamento-Digital-de-Imagens/main/test.csv"

# URL base das imagens no GitHub
BASE_IMAGE_URL = "https://raw.githubusercontent.com/antunesbrunoo/Processamento-Digital-de-Imagens/f6d0cc566de3d788d323003e83cae8331b0988bb/augmented_dataset/"

# Carregar os dados diretamente do GitHub
train_data = pd.read_csv(url_train)
test_data = pd.read_csv(url_test)

# Lista de categorias das imagens (0,1,2,3,4,5,6,7,8)
categories = [
    "apple",
    "banana",
    "grape",
    "guava",
    "lemon",
    "mango",
    "melon",
    "orange",
    "papaya",
    "pear",
]

# Array de entrada com imagens achatadas
flat_data_arr = []
flat_data_arr_gray = []
flat_data_arr_mean = []

# Função para obter a média de uma imagem
def mean_image(image):
    return np.mean(image, axis=2)  # Calcula a média ao longo do canal de cores

# Função para carregar as imagens de treinamento
def load_image(image_id):
    url = BASE_IMAGE_URL + image_id
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = imread(BytesIO(response.content))
            image = cv2.resize(image, (1280, 720))  # Redimensionar para 1280 x 720
            flat_data_arr.append(image.flatten())
            flat_data_arr_gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten())
            flat_data_arr_mean.append(mean_image(image).flatten())
            return image
        else:
            print(f"Failed to download image: {url}")
            return None
    except Exception as e:
        print(f"Error occurred while downloading image {url}: {e}")
        return None

# *** Isso deve ser chamado para preencher o flat_data_arr ***
# Imagens do dataset de treinamento
print("Getting images, it takes a while now, because of mean")
train_images = train_data["image_id"].apply(load_image)

# Remover imagens que falharam ao ser baixadas
valid_indices = [i for i, img in enumerate(train_images) if img is not None]
flat_data_arr = [flat_data_arr[i] for i in valid_indices]
flat_data_arr_gray = [flat_data_arr_gray[i] for i in valid_indices]
flat_data_arr_mean = [flat_data_arr_mean[i] for i in valid_indices]
y = np.array(train_data["class"])[valid_indices]

# Verificar se todas as imagens foram redimensionadas corretamente
print(f"Number of valid images: {len(flat_data_arr)}")
if len(flat_data_arr) > 0:
    img_shape = np.array(flat_data_arr[0]).shape
    print(f"Shape of the first image in flat_data_arr: {img_shape}")

# Verificar se todas as imagens têm o mesmo número de pixels
image_shapes = {np.array(img).shape for img in flat_data_arr}
print(f"Unique image shapes in flat_data_arr: {image_shapes}")

# Garantir que todos os arrays tenham o mesmo número de linhas
min_len = min(len(flat_data_arr), len(flat_data_arr_gray), len(flat_data_arr_mean))
flat_data_arr = flat_data_arr[:min_len]
flat_data_arr_gray = flat_data_arr_gray[:min_len]
flat_data_arr_mean = flat_data_arr_mean[:min_len]

# Verificar as formas após ajuste
print(f"Shapes after adjustment:")
print(f"flat_data_arr: {[np.array(img).shape for img in flat_data_arr]}")
print(f"flat_data_arr_gray: {[np.array(img).shape for img in flat_data_arr_gray]}")
print(f"flat_data_arr_mean: {[np.array(img).shape for img in flat_data_arr_mean]}")

# Concatenar os dados
flat_data = np.concatenate((np.array(flat_data_arr_gray), np.array(flat_data_arr_mean)), axis=1)

# Preparar os dados para o classificador
df = pd.DataFrame(flat_data)  # dataframe
df["Target"] = y
X = df.iloc[:, :-1]  # dados de entrada
y = df.iloc[:, -1]  # dados de saída

# Separar os dados entre treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("Training data and target sizes: \n{}, {}".format(X_train.shape, y_train.shape))
print("Test data and target sizes: \n{}, {}".format(X_test.shape, y_test.shape))

# Classificador SVC com kernel linear
svclassifier = SVC(kernel="linear", probability=False)

# Treinar o classificador
print("Started train of SVC model...")
svclassifier.fit(X_train, y_train)
print("Finished train.")

# Prever
y_pred = svclassifier.predict(X_test)

# Avaliar o desempenho
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))