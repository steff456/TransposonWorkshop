# Visualizando resultados para evaluación de detección de transposones con Spyder

El objetivo principal de este workshop es explorar algunas de las funciones del editor Spyder para realizar proyectos de computación científica. Para ello, trabajaremos en visualizar los resultados de detección de transposones usando librerias de Python como `numpy`, `matplotlib` y `sklearn`. Los datos que vamos a utilizar incluyen las anotaciones y predicciones de transposones realizadas por el algoritmo X para el ensamblaje de levadura. Para mayor información sobre el problema de detección de transposones, hay mas información en las diapositivas disponibles [acá](https://drive.google.com/file/d/1Mi9FVUcnBEZwJJelOeHxZrj-mD45oXIB/view?usp=sharing).

Después de realizar este workshop podrás lograr siguientes 5 actividades,

1. Abrir y editar un proyecto en Spyder.
2. Cargar datos de archivos separados por coma.
3. Crear una clase en python e importarla en un archivo.
4. Definir funciones para calcular métricas de evaluación para problemas de detección.
5. Graficar curvas de precisión-cobertura y ROC.

## Tabla de contenidos

1. [Prerequisitos](#pre)
2. [Primeros pasos](#pripasos)
3. [Reconociendo el proyecto](#reconociendo)
4. [Importando librerias](#import)
4. [Explorando los datos](#exp)
5. [Cargando y representando los datos](#carg)
6. [Calculando las métricas de evaluación](#met)
7. [Graficando las curvas de precisión/cobertura y ROC](#graf)

## Prerequisitos <a name="pre"></a>

### Creando ambiente de Anaconda

Es necesario realizar la instalación de Anaconda para crear un ambiente de desarrollo. Para instalar este programa por favor vaya a este [enlace](https://www.anaconda.com/products/individual) y descargue el instalador para su sistema operativo. Una vez este instalado, abra una terminal del sistema y cree un ambiente con Python 3.8 llamado `t-workshop` como lo indican las siguientes instruciones,

```
conda create -n t-workshop -c conda-forge python=3.8
conda activate t-workshop
```

### Instalando Spyder

Adicionalmente, necesitaremos instalar el editor Spyder, recomendamos los instaladores de Spyder para los usuarios de Windows o Mac disponibles [aquí](https://github.com/spyder-ide/spyder/releases/tag/v5.0.5). También se puede descargar Spyder desde anaconda desde una terminal del sistema siguiendo la instrucción,

```
conda install spyder -c conda-forge
```

### Instalando paquetes

Las librerias necesarias para este workshop son `numpy`, `pandas`, `matplotlib` y `sklearn`, las cuales se pueden instalar desde una terminal de la siguiente forma,

```
conda install numpy matplotlib scikit-learn pandas -c conda-forge
```

**Nota:** Si utilizó uno de los instaladores para Mac o Windows de Spyder terminal por favor siga las instrucciones para conectar nuestro nuevo ambiente al editor de Spyder disponibles [aquí](https://docs.spyder-ide.org/current/faq.html#using-existing-environment).

## Primeros pasos <a name="pripasos"></a>
Si está familiarizado con `git`, por favor clone el siguiente repositorio,

```
git clone https://github.com/steff456/WorkshopTransposones
```

De otra forma, se pueden descargar los contenidos de este workshop en este [enlace](FALTA EL LINK).

Una vez descargue el material, inicie Spyder utilizando el acceso directo disponible en su computador. Abra el workshop en Spyder como un proyecto utilizando el menu `Proyecto > Abrir proyecto` y seleccionando la carpeta previamente descargada. Finalmente, abra el archivo `workshop.py` haciendo doble click en él en el explorador de archivos dentro de Spyder.

## Reconociendo el proyecto <a name="reconociendo"></a>
Una vez abramos el proyecto en Spyder, vamos a encontrar tres carpetas bajo el mismo nivel. La carpeta `data` contiene los datos sobre los que vamos a trabajar. Dentro de ella, tenemos un archivo que contiene las anotaciones de transposones y adicionalmente tenemos dos archivos con diferentes predicciones realizadas por el algoritmo X. Por otra parte, la carpeta `workshop` contiene los archivos que vamos a modificar hoy. Por último, la carpeta `solution` contiene la solución del workshop que realizaremos hoy.

## Importando librerias <a name="import"></a>
Lo primero que necesitamos hacer es importar las librerias que ya tenemos instaladas en nuestro sistema en nuestro archivo `main.py`.

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from heapq import heappush, heappop
```

## Explorando los datos <a name="exp"></a>
Para explorar los datos de anotaciones y preddiciones de transposones para levadura vamos a cargar los datos en Spyder y utilizando las siguientes instrucciones en la IPython Console,

```
# Cargar las anotaciones
anotaciones = pd.read_csv('./data/groundtruth.csv')

# Cargar los datos de predicción
pred_a = pd.read_csv('./data/predA.csv')
pred_b = pd.read_csv('./data/predA.csv')
```

## Cargando y representando los datos <a name="carg"></a>

Dados los campos de cada uno de los archivos, nos podemos dar cuenta que podemos crear una clase de Python para cargar los archivos y después poder manipularlos de forma sencilla. La idea principal, es que esta clase contenga la información del nombre de la secuencia, en este caso particular el cromosoma, el inicio y fin del transposon y la fiabilidad de la predicción. Adicionalmente vamos a definir funciones que utilizaremos para calcular las métricas y generar las gráficas de nuestra evaluación.

En el archivo `transposon.py` vamos a definir la clase de la siguiente manera:

```
class Transposon():
    # Definir inicializacion del objeto
    # Definir si hay sobrelape con otro transposon
    # Definir el tamaño del sobrelape
    # Retornar la longitud del transposon
    # Retornar la comparacion de la fiabilidad
    # Definir la representacion del String
    pass
```

Pimero, vamos a definir la inicialización del objeto, para ello vamos a hacer un constructor que espere los datos del nombre de la secuencia, inicio, fin y fiabilidad como parámetros,

```
# Definir inicializacion del objeto
def __init__(self, seq_name, first, last, score):
    self.sequence_name = seq_name
    self.first = first
    self.last = last
    self.score = score
```

Dado que definimos nuestra métrica de evaluación como un sobrelape, vamos a desarrollar una función que nos permita definir si dos transposones se sobrelapan,

```
# Definir si hay sobrelape con otro transposon
def is_overlap(self, transposon):
    if self.first <= transposon.last <= self.last:
        return True
    elif self.first <= transposon.first <= self.last:
        return True
    else:
        return False
```

Otra parte fundamental de nuestra métrica es definir la longitud del sobrelape, en caso de que lo haya. Para ello, vamos a construir otro método que se encargue de calcularlo,

```
# Definir el tamaño del sobrelape
def get_overlap(self, transposon):
    return max(0, min(self.last-transposon.first,
                      transposon.last-self.first,
                      len(self), len(transposon)))
```

Adicionalmente, vamos a definir algunos métodos especiales del protocolo de Python, conocidos en ingles como *dunder methods*. Estas funciones ya se encuentran incorporadas al lenguaje, y nos permiten utilizar otros métodos reservados como `len` o inclusive las comparaciones numéricas e igualdades. En este caso, vamos a crear un método que retorne la longitud de nuestro transposon, otro para comparación e igualdad y la representación en cadena de texto,

```
# Retornar la longitud del transposon
def __len__(self):
    return self.last - self.first + 1
```

```
# Retornar la comparacion de la fiabilidad con otro transposon
def __gt__(self, transposon):
    return self.score > transposon.score
```

```
# Retornar la definicion de igualdad con otro transposon
def __eq__(self, transposon):
    return self.score == transposon.score
```

```
# Definir la representacion de cadena de texto
def __str__(self):
    return '{}\t{}\t{}'.format(self.sequence_name, self.first, self.last)
```

Como ya tenemos la representación de nuestro objeto, lo que sigue es poder pasar nuestros datos cargados en el dataframe de Pandas a nuestra clase en el archivo `main.py`. Lo primero que tenemos que hacer, es importar la clase que esta en el archivo `transposon.py`,

```
from workshop.transposon import Transposon
```

Después, vamos a crear una función para procesar la información,

```
#%% Pasar los datos a la clase transposon
def process_info(df):
    act_info = {}
    for _, row in df.iterrows():
        act_seq = row['seq_name']
        act_start = row['start']
        act_end = row['end']
        try:
            act_score = row['score']
        except:
            act_score = 0
        if act_seq in act_info:
            act_info[act_seq].append(
                Transposon(act_seq, act_start, act_end, act_score))
        else:
            act_info[act_seq] = [
                Transposon(act_seq, act_start, act_end, act_score)]
    return act_info
```

Y llamamos a cada uno de nuestros dataframes para convertirlos a una lista de objetos de tipo Transposon,

```
anotaciones = process_info(anotaciones)
pred_a = process_info(pred_a)
pred_b = process_info(pred_b)
```

## Calculando las métricas de evaluación <a name="met"></a>
En muchos problemas de detección encontramos que no hay un umbral específico definido para definir si una predicción es correcta. Por ello, vamos a definir una función que reciba este valor como un parámetro. Esta función va a calcular el valor de verdaderos positivos, falsos negativos, falsos positivos,

```
# Calcular verdaderos positivos, falsos positivos y falsos negativos
def get_single_instance_results(gts, preds, thresh):
    tp = 0
    IoUs = []
    gt_index = []
    pred_index = []
    for x, pred in enumerate(preds):
        for y, gt in enumerate(gts):
            act_overlap = pred.get_overlap(gt)
            IoU = act_overlap/(len(pred) + len(gt) - act_overlap)
            if IoU >= thresh:
                IoUs.append(IoU)
                gt_index.append(y)
                pred_index.append(x)
    IoUs = np.argsort(IoUs)[::-1]
    if len(IoUs) == 0:
        # No hay sobrelapamiento
        return 0, len(preds), len(gts)
    gt_match_idx = []
    pred_match_idx = []
    for idx in IoUs:
        gt_idx = gt_index[idx]
        pr_idx = pred_index[idx]
        if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
            gt_match_idx.append(gt_idx)
            pred_match_idx.append(pr_idx)
    tp = len(gt_match_idx)
    fp = len(preds) - len(pred_match_idx)
    fn = len(gts) - len(gt_match_idx)
    return tp, fp, fn
```

Ya calculados los valores de verdaderos positivos, falsos positivos y falsos negativos, podemos definir las funciones que calculan la presición, cobertura y F-medida de la siguiente forma,

```
# Calcular presicion
def calculate_precision(tp, fp):
    return tp/(tp + fp + 1e-9)
```

```
# Calcular cobertura
def calculate_recall(tp, fn):
    return tp/(tp + fn + 1e-9)
```

```
# Calcular F-medida
def calculate_fmeasure(precision, recall):
    return (2*precision*recall)/(precision + recall + 1e-9)
```

Ya que tenemos todas nuestras funciones auxiliares definidas, podemos definir la función que va a calcular todas las métricas por cromosoma. Así, podemos sacar los valores de las métricas globales, pero también por cromosoma si se deseara.

```
def calculate_metrics(gt, pred, thresh=0.5, verbose=True):
    names = list(set(gt.keys()).union(set(pred.keys())))
    cum_tp, cum_fp, cum_fn, total_p, total_gt = 0, 0, 0, 0, 0
    for seq_name in names:
        if seq_name not in pred:
            cum_fn += len(gt[seq_name])
            total_gt += len(gt[seq_name])
            continue
        elif seq_name not in gt:
            cum_fp += len(pred[seq_name])
            total_p += len(pred[seq_name])
            continue
        act_gt = gt[seq_name]
        act_pred = pred[seq_name]
        tp, fp, fn = get_single_instance_results(act_gt, act_pred, thresh)
        if False:
            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, fn)
            print('----------- {} -----------'.format(seq_name))
            print('TP:', tp, 'FP:', fp, 'FN:', fn)
            print('precision', precision)
            print('recall', recall)
            print('F-measure', calculate_fmeasure(precision, recall))
            print('-----------')
        cum_tp += tp
        cum_fp += fp
        cum_fn += fn
        total_p += len(pred[seq_name])
        total_gt += len(gt[seq_name])
    precision = calculate_precision(cum_tp, cum_fp)
    recall = calculate_recall(cum_tp, cum_fn)
    fmeasure = calculate_fmeasure(precision, recall)
    if verbose:
        print('-----------')
        print('TP:', cum_tp, 'FP:', cum_fp, 'FN:', cum_fn)
        print('Total pred:', total_p, 'Total gt:', total_gt)
        print('threshold', thresh)
        print('precision', precision)
        print('recall', recall)
        print('F-measure', fmeasure)
        print('-----------')
    return precision, recall, fmeasure, cum_fp, cum_tp
```

Así, logramos sacar los resultados para nuestras dos predicciones previamente cargadas,

```
# Sacar los resultados de nuestras predicciones
result_a = calculate_metrics(anotaciones, pred_a)
result_b = calculate_metrics(anotaciones, pred_b)
```

## Graficando las curvas de precisión/cobertura y ROC <a name="graf"></a>
Como ya pudimos calcular los resultados para precisión, cobertura, F-medida, total de falsos positivos y total de verdaderos positivos, podemos empezar a graficar las curvas de precisión/cobertura y ROC. Para la primera gráfica, utilizaremos las funciones previamente realizadas variando la confianza aceptada por el algoritmo x, llamado `score`, de tal forma que solo se tengan en cuenta las detecciones de mayor puntaje. Mientras que para la curva ROC solamente necesitaremos el total de falsos positivos y el total de verdaderos positivos.

### Curva de precisión/cobertura
En este caso, vamos a variar los valores del `score`, para ello definimos una lista con los incrementos que vamos a probar para generar nuestra gráfica,

```
# Puntajes de confianza de 0 a 16000 con pasos de 5
scores = np.linspace(0, 16000, 5)
```

Adicionalmente, vamos a crear una función que solamente tenga en cuenta los transposones detectados con un puntaje mayor al dado,

```
# Calcular todas las métricas para multiples puntajes
def calculate_multiple_scores(gt, preds, thresh, scores):
    precisions = []
    recalls = []
    false_positives = []
    true_positives = []
    for score in scores:
        pred = preds
        total_p = []
        total_r = []
        for name in pred:
            for transposon in pred[name]:
                if transposon.score < score:
                    heappop(pred[name])
        precision, recall, fm, fp, tp = calculate_metrics(
            gt, pred, thresh=thresh)
        total_p.append(precision)
        total_r.append(recall)
        print('-------- Score {} ---------'.format(score))
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('Fmeasure: {}'.format(fm))
        precisions.append(precision)
        recalls.append(recall)
        false_positives.append(fp)
        true_positives.append(tp)
    return precisions, recalls, false_positives, true_positives
```

Utilizamos matplotlib para definir la gráfica de la siguiente manera,

```
# Función para generar la gráfica de precision cobertura
def plot_PR(precisions, recalls, threshs=0.5):
    plt.plot(recalls, precisions, label=threshs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve for Transposon Detection')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc='best')
    plt.show()
```

Seguidamente, graficámos los resultados de nuestras predicciones,

```
# Graficar los resultados de nuestras predicciones
grafica_a = calculate_multiple_scores(anotaciones, pred_a, 0.5, scores)
plot_PR(grafica_a[0], grafica_a[1], 0.5)

grafica_b = calculate_multiple_scores(anotaciones, pred_b, 0.5, scores)
plot_PR(grafica_b[0], grafica_b[1], 0.5)
```

### Curva ROC
De forma similar y utilizando el paquete de metricas de `sklearn` podremos gráficar la curva Característica Operativa del Receptor (ROC). Para ello, vamos a definir una nueva función,

```
# Función para generar la gráfica ROC
def plot_ROC(fpr, tpr):
    """Plot ROC curve."""
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name='TransposonFinder')
    display.plot()
    plt.show()
```

Por último, graficámos los resultados de nuetras predicciones,

```
# Graficar los resultados de nuestras predicciones
plot_ROC(grafica_a[2], grafica_a[3])

plot_ROC(grafica_b[2], grafica_b[3])
```
