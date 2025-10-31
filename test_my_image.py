import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# =====================
# MODELO MULTI-TASK (IGUAL QUE EN EL ENTRENAMIENTO)
# =====================
class MultiTaskDogModel(nn.Module):
    def __init__(self, num_breeds, num_stages):
        super(MultiTaskDogModel, self).__init__()
        
        # Backbone pre-entrenado (igual que en entrenamiento)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remover la última capa (igual que en entrenamiento)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Cabezas para cada tarea
        self.breed_classifier = nn.Linear(512, num_breeds)
        self.stage_classifier = nn.Linear(512, num_stages)
        
    def forward(self, x):
        # Extraer características
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        # Predicciones para cada tarea
        breed_output = self.breed_classifier(features)
        stage_output = self.stage_classifier(features)
        
        return breed_output, stage_output

# =====================
# CONFIGURACIÓN - ¡CAMBIA AQUÍ LA RUTA DE TU IMAGEN!
# =====================

# 🔥 CAMBIA ESTA RUTA POR LA IMAGEN QUE QUIERAS PROBAR 🔥
IMAGE_PATH = "Img/imagen6.webp"

# También puedes usar rutas absolutas como:
# IMAGE_PATH = r"C:\Users\TuUsuario\Downloads\mi_perro.jpg"

MODEL_PATH = "models/best_multitask_dog_model.pth"

# =====================
# FUNCIÓN DE PREDICCIÓN
# =====================
def predict_image(image_path, model_path):
    """
    Predice la raza y etapa de vida de un perro en una imagen
    """
    
    print(f"🔍 Analizando imagen: {image_path}")
    
    # Verificar que la imagen existe
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encontró la imagen en {image_path}")
        return
    
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Usando dispositivo: {device}")
    
    # Definir transformaciones (igual que en entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Definir las clases (deben coincidir con el entrenamiento)
    breed_classes = ['bulldog', 'chihuahua', 'golden retriever']  # alfabético
    stage_classes = ['adulto', 'cachorro', 'joven', 'senior']     # alfabético
    
    # Crear e inicializar el modelo
    model = MultiTaskDogModel(len(breed_classes), len(stage_classes))
    
    # Cargar los pesos del modelo entrenado
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Verificar si el checkpoint tiene metadatos adicionales
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Modelo cargado correctamente (con metadatos)")
            # Mostrar información adicional si está disponible
            if 'epoch' in checkpoint:
                print(f"   Época de entrenamiento: {checkpoint['epoch']}")
            if 'val_breed_acc' in checkpoint:
                print(f"   Precisión en razas: {checkpoint['val_breed_acc']:.3f}")
            if 'val_stage_acc' in checkpoint:
                print(f"   Precisión en etapas: {checkpoint['val_stage_acc']:.3f}")
        else:
            # Si no tiene metadatos, cargar directamente
            model.load_state_dict(checkpoint)
            print(f"✅ Modelo cargado correctamente (formato simple)")
            
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Cargar y procesar la imagen
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"📸 Imagen cargada - Tamaño: {image.size}")
        
        # Aplicar transformaciones
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Hacer predicción
        with torch.no_grad():
            breed_outputs, stage_outputs = model(input_tensor)
            
            # Aplicar softmax para obtener probabilidades
            breed_probs = torch.softmax(breed_outputs, dim=1)
            stage_probs = torch.softmax(stage_outputs, dim=1)
            
            # Obtener las predicciones
            breed_pred_idx = torch.argmax(breed_probs, dim=1).item()
            stage_pred_idx = torch.argmax(stage_probs, dim=1).item()
            
            # Obtener las probabilidades máximas
            breed_confidence = breed_probs[0][breed_pred_idx].item()
            stage_confidence = stage_probs[0][stage_pred_idx].item()
            
            # Mapear índices a nombres
            predicted_breed = breed_classes[breed_pred_idx]
            predicted_stage = stage_classes[stage_pred_idx]
            
            # Mostrar resultados
            print("\n" + "="*50)
            print("🎯 RESULTADOS DE LA PREDICCIÓN")
            print("="*50)
            print(f"🐕 RAZA: {predicted_breed.upper()}")
            print(f"   Confianza: {breed_confidence:.3f} ({breed_confidence*100:.1f}%)")
            
            print(f"📅 ETAPA: {predicted_stage.upper()}")
            print(f"   Confianza: {stage_confidence:.3f} ({stage_confidence*100:.1f}%)")
            
            # Evaluación de confianza
            print("\n📊 EVALUACIÓN:")
            if breed_confidence > 0.8:
                print(f"✅ Raza: MUY CONFIABLE")
            elif breed_confidence > 0.6:
                print(f"⚠️  Raza: MODERADAMENTE CONFIABLE")
            else:
                print(f"❌ Raza: POCO CONFIABLE")
                
            if stage_confidence > 0.8:
                print(f"✅ Etapa: MUY CONFIABLE")
            elif stage_confidence > 0.6:
                print(f"⚠️  Etapa: MODERADAMENTE CONFIABLE")
            else:
                print(f"❌ Etapa: POCO CONFIABLE")
            
            # Mostrar todas las probabilidades
            print(f"\n🔍 TODAS LAS PROBABILIDADES:")
            print("Razas:")
            for i, breed in enumerate(breed_classes):
                prob = breed_probs[0][i].item()
                print(f"  {breed}: {prob:.3f} ({prob*100:.1f}%)")
            
            print("Etapas:")
            for i, stage in enumerate(stage_classes):
                prob = stage_probs[0][i].item()
                print(f"  {stage}: {prob:.3f} ({prob*100:.1f}%)")
            
            print("="*50)
            
    except Exception as e:
        print(f"❌ Error procesando la imagen: {e}")

# =====================
# EJECUTAR PREDICCIÓN
# =====================
if __name__ == "__main__":
    print("🐕 PREDICTOR DE RAZAS Y ETAPAS DE VIDA")
    print("=" * 50)
    
    predict_image(IMAGE_PATH, MODEL_PATH)
    
