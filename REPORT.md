# Отчет по лабораторной работе 
## по курсу "Искусственый интеллект"

## Нейросетям для распознавания изображений


### Студенты: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Ли Алиса | Все |                                      |


## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |

> *Опоздание с сдаче*

## Тема работы

Вариант 2: 5 первых символов греческого алфавита

С помощью фреймворка Keras требуется реализовать и обучить несколько нейросетей для распознавания картинок с изображениями первых пяти букв греческого алфавита. Также нужно измерить точность на обучающей и тестовой выборке.

Для выполнения задания необходимо составить датасет из 1500 картинок, обработать его и представить в удобном для работы нейросетей формате.

## Подготовка данных

Для создания датасета потребовалось выполнить следующие этапы:
1. Распечатка 15 листов с сеткой из 100 ячеек, каждая ячейка размером примерно 2*2 см
2. Рисование каждым из участников 500 букв на вышеупомянутых листках.
3. Сканирование полученных изображений и обрезка лишних областей.
4. Перевод картинок в одноканальный, черно-белый формат.
5. Нарезка картинки размером 600x600 на 100 картинок размером 60x60
6. Масштабирование изображений отдельных букв до размера 32x32

**_Отсканированные изображения_**

![alpha](/img/a1.png)
![beta](/img/b1.png)
![gamma](/img/g1.png)
![delta](/img/d1.png)
![delta](/img/e1.png)

**_Код для обработки изображений_**
```Python
def load():
    path = "/content/*.*"
    """Загрузка картинок без обрезания"""
    return [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in sorted(glob.glob(path))]

def split(img):
    """Разбиение картинок на маленькие"""
    size = 10
    img_height = img.shape[0] // size
    img_width = img.shape[1] // size
    res = []
    for i in range(size):
        for j in range(size):
            res.append(img[i * img_height:(i+1) * img_height, j * img_width:(j+1) * img_width])
    return res

def squeeze(split_img):
    """Сжатие 32 на 32"""
    return cv2.resize(split_img, (32, 32))

def img_to1d(img):
    """Трансформация к 1 мерному"""
    return img.ravel()

def normalize(img):
    """Нормализация каждого пикселя к отрезку [0, 1]"""
    return img / 255.0

def process_img(img):
    """Обработка исходной картинка"""
    return (squeeze(img))

def img_to1d(img):
    """Трансформация к 1 мерному"""
    return img.ravel()
```

## Загрузка данных

Загрузка картинок выполняется следующим образом:

```Python
labels = np.array([0]*300+[1]*300+[2]*300+[3]*300+[4]*300)
src_imgs = load()
label = 0
features = []
for img in src_imgs:
    split_imgs = split(img)
    for split_img in split_imgs:
        features.append(process_img(split_img))
data_set = features

fig = plt.figure()
for i in range(0,6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(data_set[i], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(labels[i]))
    plt.xticks([])
    plt.yticks([])

for i in range(0, len(data_set)):
    data_set[i] = normalize(data_set[i])
```

Картинки такого вида находятся в датасете:

![sliced](/img/sliced.png)

## Обучение нейросетей

Создание нейросетей такова:

```Python
#Сверточная сеть
model1 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model1.summary()

# Полносвязная сеть многослойная
model2 =  keras.Sequential([
  keras.Input(shape=1024),
  layers.Dense(64, activation='relu'),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes, activation='softmax'),
])

model2.summary()

# Полносвязная сеть однослойная
model3 =  keras.Sequential([
  keras.Input(shape=1024),
  layers.Dense(64, activation='relu'),
  layers.Dense(num_classes, activation='softmax'),
])

model3.summary()
```

Архитектура сетей:

![arch1](/img/conv_arch.png)
![arch2](/img/one_layer_arch.png)
![arch3](/img/two_layer_arch.png)

Результаты обучения(accuracy на тестовой выборке):
Сверточная сеть - 0.90707
Двуслойная сеть - 0.70101
Однослойная сеть - 0.65656



## Выводы

В данной работе были получены навыки собственноручно создавать и обрабатывать наборы изображений для формирования датасетов и дальнейшего их использования для обучения нейросетей, созданных с помощью фреймворка Keras. Анализ результатов показал, что сверточные нейросети - лучший выбор для работы с изображениями. 
