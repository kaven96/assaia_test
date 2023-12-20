# Выбор метрик

Обычно для задач детекции объектов на фото и видео исполльзуют метрику Average Precision и её вариации ([A Review of Video Object Detection: Datasets, Metrics and Methods](https://www.mdpi.com/2076-3417/10/21/7834), [A Survey on Performance Metrics for Object-Detection Algorithms](https://www.researchgate.net/publication/343194514_A_Survey_on_Performance_Metrics_for_Object-Detection_Algorithms)). 

Предполагается, что для вычисления метрики уже есть размеченная выборка, на которой и происходит тренировка и оценка алгоритма.

В данном случае имеются данные не с разметкой кадров, а со временем появления автомобилей в интересующей области и их выходом из неё.

Поэтому в качестве метрики я буду использовать совпадение предсказанных временных интервалов (pred) и ground truth (gt) интервалов. И Precision, Recall на их основе. Будем считать:

* True Positive: длину пересечения участков pred и gt
* False Positive: длину pred участка, которая не пересекается с gt
* False Negative: длину gt участка, которая не пересекается с pred

Также будем использовать метрику F1, которая является комбинацией Precision и Recall, а также Accuracy.

# Подходы к решению задачи

## Первый подход

и кажущийся очевидным --- использовать модели для детекции объектов на фото / видео, определять пересечение bounding box'ов автомобилей с интересующей областью. Плюсы:

1. При достаточно хорошей модели детекции, можно исключить ложные срабатывания при пересечении площадки людьми / пролетом птиц или насекомых перед камерой.

2. Если модель натренирована не только на автомобили, то можно вести в будущем логирование всех перемещений на площадке.

3. Подход простой и понятный, его легко объяснить заказчикам, можно разбить на составляющие (основа --- модель для детекции, остальное --- вспомогательные этапы: предобработка видео, проверка пересечений объектами интересующей зоны)

Минусы:

1. Для работы в real time современных точных моделей для детекции (YOLOv8, detectron2) необходимо иметь достаточно мощный GPU с поддержкой CUDA
2. Более простые модели (Haar Cascade) работают быстро на CPU, но недостаточно точны
3. Не все аэродромные автомобили могут распознаваться предтренированными моделями, поэтому необходимо собирать свой датасет с конкретными аэродромными автомобиями с разных ракурсов.

### Гиперпараметры для Haar Cascade

Лучшие гиперпараметры были подобраны поиском по сетке (tune_haar.py). Neighbors: 7, Scale: 1.22.

Лучшие метрики полученные данной моделью:

* Precision: 0.5588235294117647
* Recall: 0.8444444444444444
* Accuracy: 0.5066666666666667
* F1 Score: 0.672566371681416

Данный результат является абсолютно неудовлетворительным. Необходимо использовать другую модель для детекции автомобилей. Либо использовать второй подход к решению задачи.

## Второй подход

Использовать OwlViT --- модель, позоляющую находить объекты на изображении по открытому словарю. Основана на трансформерах и CLIP.

В качестве самостоятельной модели работает с низким качеством, потому не была применена.

Реализацию можно посмотреть в файле **detect2.py**

## Третий подход 

к решению --- более подходящий для данной задачи. 

Определять движущиеся объекты с помощью OpenCV Background Substraction. Таким образом определяются все движущиеся на видео объекты.

Проблема возникает, если объект перестает двигаться в интересующей области. Для обхода проблемы bounding box последнего объекта сохраняется до момента, пока не будет сравнен с вновь обнаруженным объектом, и если их пересечение достаточно велико, считается, что все это время объект находился в интересующей области.

Другая проблема --- в интересующей области могут перемещаться не только автомобили, но и люди. Для решения этой проблемы можно использовать OwlViT, чтобы оценивать вероятность того, что внутри bounding box'а находится автомобиль. В итоговом решении это не реализовано, но это можно объединить с предыдущим вариантом решения, и получить более высокое качество, т.к. оценка будет проводиться внутри небольшого bounding box'а.

## Выбранный подход

Все три подхода были реализованы, оценены и выбран третий как наиболее рациональный.

* Precision: 0.5555555555555556
* Recall: 0.9935897435897436
* Accuracy: 0.5535714285714286
* F1 Score: 0.7126436781609196

# Запуск

Для установки зависимостей, обработки одного видео, либо всех видео в директории, вычисления метрик необходимо запустить:

```
bash RUN.sh <path_to_video_file> <path_to_polygons.json> <path_to_save_output.json>
```
