# 🐕 Clasificador de Razas y Etapas de Vida de Perros

Un modelo de inteligencia artificial que identifica automáticamente la raza y la etapa de vida (cachorro, joven, adulto o senior) de un perro a partir de una fotografía.

---

## 🎯 ¿Qué hace este proyecto?

Este proyecto utiliza **Deep Learning** para analizar imágenes de perros y predecir dos cosas simultáneamente:

1. **La raza del perro**: Bulldog, Chihuahua o Golden Retriever
2. **La etapa de vida**: Cachorro, Joven, Adulto o Senior

Todo esto a partir de una simple fotografía, sin necesidad de proporcionar información adicional.

---

## 🧠 ¿Cómo funciona?

### Modelo: Multi-Task Learning

Utilizamos una técnica llamada **Multi-Task Learning** (Aprendizaje Multi-Tarea), que significa que nuestro modelo puede hacer dos predicciones al mismo tiempo usando una sola imagen.

Piensa en ello como un experto veterinario que puede ver una foto y decirte tanto la raza como la edad aproximada del perro de un solo vistazo.
---

## 🔬 Transfer Learning: Aprendiendo de millones de imágenes

### ¿Qué es Transfer Learning?

En lugar de enseñarle a nuestro modelo desde cero qué es un perro, usamos un modelo que ya fue entrenado con **1.2 millones de imágenes** de todo tipo de objetos (el dataset ImageNet).

**Analogía**: Es como si contratáramos a alguien que ya sabe reconocer animales en general, y solo le enseñamos los detalles específicos de razas y edades de perros. ¡Mucho más rápido y eficiente!

### Ventajas de Transfer Learning

**Entrena más rápido**: Minutos en lugar de días  
**Necesita menos datos**: Funciona con cientos en lugar de millones de imágenes  
**Mejor precisión**: Aprovecha conocimiento de imágenes similares  
**Menos recursos**: No necesitas supercomputadoras para entrenar  

### ¿Cómo lo implementamos?

```python
# Cargamos ResNet-18 ya entrenada en ImageNet
model = models.resnet18(pretrained=True)

# Solo ajustamos la última capa para nuestras clases específicas
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
```

El 95% del modelo ya está entrenado. Solo personalizamos la parte final para nuestras necesidades específicas.

---

## 📁 Estructura del Dataset

### Organización de las Imágenes

Nuestro dataset está organizado de forma jerárquica, donde cada carpeta representa una categoría:

```
dataset/
├── train/                          # Imágenes para entrenar (80%)
│   ├── bulldog/
│   │   ├── cachorro/               # Bulldogs cachorros
│   │   ├── joven/                  # Bulldogs jóvenes
│   │   ├── adulto/                 # Bulldogs adultos
│   │   └── senior/                 # Bulldogs seniors
│   ├── chihuahua/
│   │   ├── cachorro/
│   │   ├── joven/
│   │   ├── adulto/
│   │   └── senior/
│   └── golden retriever/
│       ├── cachorro/
│       ├── joven/
│       ├── adulto/
│       └── senior/
└── val/                            # Imágenes para validar (20%)
    ├── bulldog/
    ├── chihuahua/
    └── golden retriever/
```

### ¿Por qué esta estructura?

Esta organización permite que el modelo aprenda dos niveles de información:

1. **Nivel de Raza** (carpeta principal): bulldog, chihuahua, golden retriever
2. **Nivel de Etapa** (subcarpetas): cachorro, joven, adulto, senior

Cada imagen tiene **dos etiquetas automáticamente**:
- `train/bulldog/cachorro/foto1.jpg` → Raza: Bulldog, Etapa: Cachorro
- `train/chihuahua/adulto/foto2.jpg` → Raza: Chihuahua, Etapa: Adulto

---

## 🛠️ ¿Cómo usar este proyecto?

### Requisitos Previos

- Python 3.8 o superior
- PyTorch (framework de Deep Learning)
- Una computadora con GPU es ideal, pero también funciona con CPU

Las librerías principales que usamos:
- `torch`: PyTorch, el framework de Deep Learning
- `torchvision`: Modelos pre-entrenados y transformaciones de imágenes
- `pillow`: Procesamiento de imágenes
- `numpy`: Operaciones matemáticas

El entrenamiento tomará unos minutos y guardará el modelo entrenado en la carpeta `models/`.

**Lo que sucede durante el entrenamiento**:
1. Carga las imágenes del dataset
2. Aplica transformaciones (redimensiona, voltea, rota)
3. Pasa las imágenes por la red neuronal
4. Ajusta los pesos para mejorar las predicciones
5. Repite el proceso varias veces (épocas)
6. Guarda el mejor modelo

## 🎨 Procesamiento de Imágenes

Antes de que el modelo pueda analizar una imagen, la procesamos de varias formas:

### Transformaciones Aplicadas

1. **Redimensionar a 224×224 píxeles**
   - El modelo necesita todas las imágenes del mismo tamaño
   - 224×224 es el estándar para ResNet

2. **Volteo Horizontal Aleatorio** (solo en entrenamiento)
   - Aumenta la variedad de datos
   - Un perro viendo a la izquierda o derecha es el mismo perro

3. **Rotación Aleatoria ±10°** (solo en entrenamiento)
   - Simula diferentes ángulos de cámara
   - Hace el modelo más robusto

4. **Normalización**
   - Ajusta los colores a un rango estándar
   - Mejora el aprendizaje del modelo

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),          # Redimensionar
    transforms.RandomHorizontalFlip(),      # Volteo aleatorio
    transforms.RandomRotation(10),          # Rotación aleatoria
    transforms.ToTensor(),                  # Convertir a tensor
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalizar RGB
                        [0.229, 0.224, 0.225])
])
```

---

## 📊 Rendimiento del Modelo

### Métricas que Medimos

- **Accuracy (Precisión)**: Porcentaje de predicciones correctas
- **Loss (Pérdida)**: Qué tan equivocadas son las predicciones
- **Confidence (Confianza)**: Qué tan seguro está el modelo

### Niveles de Confianza

- ✅ **> 80%**: Muy confiable - Puedes confiar en el resultado
- ⚠️ **60-80%**: Moderadamente confiable - Resultado probable pero con dudas
- ❌ **< 60%**: Poco confiable - El modelo no está seguro

---

## 🔍 ¿Qué aprende el modelo?

El modelo **NO** tiene acceso a información como:
- ❌ Edad exacta del perro
- ❌ Peso o altura
- ❌ Nombre de la raza escrito en algún lugar
- ❌ Información del archivo

Todo lo aprende mirando **características visuales**:

### Para identificar la Raza:
- 🐕 Forma del hocico (corto vs. largo)
- 👂 Tamaño y forma de las orejas
- 🎨 Patrones del pelaje
- 📏 Proporciones corporales
- 💪 Constitución física

### Para identificar la Etapa de Vida:
- 📏 Tamaño relativo del cuerpo
- 👶 Proporciones (cachorros tienen cabezas más grandes)
- 🦴 Desarrollo muscular
- 👀 Rasgos faciales juveniles
- 👴 Señales de envejecimiento (pelo gris, etc.)

---

## 💡 Tecnologías Utilizadas

| Tecnología | Propósito |
|------------|-----------|
| **PyTorch** | Framework principal de Deep Learning |
| **ResNet-18** | Arquitectura de red neuronal (18 capas) |
| **ImageNet** | Dataset de pre-entrenamiento (1.2M imágenes) |
| **Adam Optimizer** | Algoritmo para optimizar el aprendizaje |
| **CrossEntropyLoss** | Función para medir errores en clasificación |
| **Data Augmentation** | Técnicas para aumentar variedad de datos |

---

## 📈 Arquitectura ResNet-18

### ¿Por qué ResNet-18?

ResNet (Residual Network) es una arquitectura revolucionaria en Deep Learning:

- **18 capas profundas**: Puede aprender características complejas
- **Conexiones residuales**: Evita problemas de entrenamiento en redes profundas
- **Pre-entrenada**: Ya conoce millones de patrones de imágenes
- **Eficiente**: Balance perfecto entre precisión y velocidad

### Flujo de Información

```
Imagen Original (224×224×3)
    ↓
[Capa Conv 1] → Detecta bordes y colores básicos
    ↓
[Capa Conv 2-5] → Detecta texturas y patrones
    ↓
[Capa Conv 6-10] → Detecta partes (orejas, ojos, hocico)
    ↓
[Capa Conv 11-18] → Detecta conceptos completos (raza, edad)
    ↓
[Fully Connected] → 512 características resumidas
    ↓
┌─────────────┴─────────────┐
↓                           ↓
[3 neuronas]              [4 neuronas]
Razas                     Etapas
```

---

## 🎓 Conceptos Clave

### 1. Deep Learning
Usar redes neuronales con múltiples capas para aprender patrones complejos en datos.

### 2. Convolutional Neural Networks (CNN)
Tipo de red especializada en procesar imágenes, inspirada en cómo funciona el sistema visual humano.

### 3. Transfer Learning
Reutilizar un modelo pre-entrenado en un problema grande para resolver un problema específico más pequeño.

### 4. Multi-Task Learning
Entrenar un solo modelo para resolver múltiples tareas relacionadas simultáneamente.

### 5. Data Augmentation
Crear variaciones artificiales de las imágenes (voltear, rotar, etc.) para tener más datos de entrenamiento.

### 6. Epoch
Una pasada completa del modelo por todo el dataset de entrenamiento.

### 7. Batch Size
Número de imágenes que el modelo procesa a la vez antes de actualizar sus pesos.



