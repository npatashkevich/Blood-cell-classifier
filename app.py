import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Определите путь к файлу с сохраненной моделью
MODEL_PATH = 'model.pth'

# Загрузите предобученную модель ResNet18 из файла
def load_model(model_path):
    model = models.resnet18(pretrained=False)  # Загружаем без предобучения
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)  # У вас 4 класса
    
    # Загрузка состояния модели (весов)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Функция для предсказания класса
def predict(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Отображение интерфейса на Streamlit
def main():
    st.title("Классификация изображений клеток крови")
    st.text("Загрузите изображение клетки крови для классификации.")

    uploaded_file = st.file_uploader("Выберите файл изображения", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение', use_column_width=True)
        st.write("")
        st.write("Классификация...")

        class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        
        # Загрузка модели из файла
        model = load_model(MODEL_PATH)
        
        # Предсказание класса
        prediction = predict(model, uploaded_file)

        st.write(f"Предсказанный класс: {class_names[prediction]}")

if __name__ == '__main__':
    main()