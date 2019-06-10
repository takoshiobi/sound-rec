#the aim of this file will be to traverse my dataset and output an array containing features for each track with corresponding labels

import glob
import os
import sys
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
# np.set_printoptions(threshold='nan')

genreDict = {
    'pop'        :   0,
    'rock'       :   1,
    'hiphop'    :   2,
    'country'       :   3,
    # 'blues'     :   0,
    # 'classical' :   1,
    # 'country'   :   2,
    # 'disco'     :   3,
    # 'hiphop'    :   4,
    # 'jazz'      :   5,
    # 'metal'     :   6,
    # 'pop'       :   7,
    # 'reggae'    :   8,
    # 'rock'      :   9,
}

# this function will iterate through each file in the dataset
def extract_features(basedir,extension='.au'):
    """
    Создает мел спектрограмму для каждого трека и сохраняет полученные спектрограммы в массив features, создает массив с жанрами для треков.
    Спектрограммы и жанры соответвуют по инексам в этих двух массивах

    :basefir (строка): папка с папками отсортированными по жанрам со всеми треками 
    :extension (строка): расширение аудио треков, .au по умолчанию 

    :return: кортеж из двух элементов: массив со спектрограммами всех треков и массив с жанрами для каждого трека
    :rtype: кортеж 

    Структура basedir:

    <basedir>
    ├── <blues>
    │   ├── blues.00000.au
    │   ├── blues.00001.au
    │   ├── blues.00002.au
    │   ...
    │   └── blues.00099.au
    ├── <classical>
    │   ├── classical.00000.au
    │   ├── classical.00001.au
    │   ├── classical.00002.au
    │   ...
    │   └── classical.00099.au
    ├── <country>
    │   ├── country.00000.au
    │   ├── country.00001.au
    │   ├── country.00002.au
    │   ...
    │   └── country.00099.au


    Extracts mel spectrogram of each music track in basedir and stores each processed spectrogram as
    an array 'features'.

    :basedir: path to basedir (string)
    :extension: extension of music files, .au by default (string) 

    :return: tuple of array of features (first item), and ...
    :rtype: tuple
    """
    # массив со спектрограммами которые представляют собой числовые массивы
    features=[]
    # массив чисел в который мы почему-то добавляем жанры в этом файле, хотя в других файлах lables это метки,
    # а не жанры. дичь.
    labels=[]

    # iterate over all files in all subdirectories of the base directory
    # обходит все папки в basedir 
    for root, dirs, files in os.walk(basedir):
        # выбирает файлы с расширением .au
        files = glob.glob(os.path.join(root,'*'+extension)) # массив 
        # обрабатывает каждый файл в массиве files
        for f in files :

            # название трека без расширения (.au) и номера трека (rock0001.au => rock)
            # названия треков в папках соответсвуют названиям жанров 
            genre = f.split('/')[4].split('.')[0] # строка 

            if (genre == 'hiphop' or genre == 'rock' or genre == 'pop' or genre == 'country'):
                # вывод жанра в консоль 
                print genre
                # Извлечение мел спектрограммы из трека 
                
                # Получение значений y и sr для трека 
                # (y) временной ряд музона, массив чисел 
                # вики спс (временной ряд представляет собой последовательность точек данных, обычно состоящую из последовательных измерений, выполненных в течение интервала времени.)
                # (sr) частота выборки для временного ряда y, число
                y, sr = librosa.load(f)

                # Let's make and display a mel-scaled power (energy-squared) spectrogram
                # Создает и отображает спектрограмму

                # параметры метода melspectrogram:
                # (y) временной ряд музона, массив чисел 
                # (sr) частота выборки, число
                # (n_mels) количество мел полос для построенния спектрограммы 
                # (hop_length) количество выборок между последовательными кадрами (https://librosa.github.io/librosa/0.4.3/generated/librosa.core.stft.html#librosa.core.stft)
                # (n_fft) размер блока или длина окна быстрого преобразования фурье
                # прикольная картинка спектрограммы:
                # https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html
                mel_spec = librosa.feature.melspectrogram(y, sr=sr,n_mels=128,hop_length=1024,n_fft=2048)
                # Convert to log scale (dB). We'll use the peak power as reference.
                # переводим значения пиков амплитуды в децибелы 

                # longamplitude : преобразовывает спектрограмму мощности (квадрат амплитуды) в децибелы
                # параметры метода longapmlitude
                # (mel_spec) сама спектрограмма созданная на строке 113
                # (ref_power) пиковая мощность 
                log_mel_spec = librosa.logamplitude(mel_spec, ref_power=np.max)
                # make dimensions of the array even 128x1292
                # делает размер массива кратным 128x1292
                log_mel_spec = np.resize(log_mel_spec,(128,644))

                # вывод в консоль спктрограммы в децибелах 
                print log_mel_spec.shape
                
                # добавляем в массив features сплющенную спектрограмму трека  
                features.append(log_mel_spec.flatten())
                # print len(np.array(log_mel_spec.T.flatten()))
                # Extract label

                # получаем номер жанра соответсвенно genreDict 
                label = genreDict.get(genre)
                # и добавляем его в массив с номерами жанров labels
                labels.append(label)
            else:
                pass

    # изменяем форму массива features для того, чтобы получить данные пригодные для последующей обработки программой
    # например 
    # > features = [1,2,3,3]
    # > features = np.asarray(features)
    # > features
    # > array([1, 2, 3, 3])   
    # reshape меняет форму массива на (4, 82432)
    # ((( для матрицы из n строк и m столбцов, shape будет (n,m) )))
    # features = [[1,2, ... 82432], [1,2, ... 82432], [1,2, ... 82432], [1,2, ... 82432]]    
    # почему 82432
    # потому что оно делится без остатка на 128 и 644 (форма массива log_mel_spec строка 125) 
    # и потому что наверное так надо чтобы потом его загнать в тензор флоу ояебу       
    features = np.asarray(features).reshape(len(features),82432)

    # выводим в консоль форму массива features
    # пример 
    # >>> result
    # array([[0., 1., 1., 0.],
    #       [0., 1., 1., 0.],
    #       [0., 1., 0., 0.],
    #       [0., 1., 1., 0.]])
    # >>> result.shape
    # (4, 4)
    # что означает что в 2д массиве 4 колонки и 4 столбца
    print features.shape

    # выводим в консоль длину массива labels 
    print len(labels)

    # features: массив спектрограмм (числовой 2д массив)
    # one_hot_encode(labels): массив меток жанров (тоже числовой 2д массив)
    return (features, one_hot_encode(labels))

def one_hot_encode(labels,num_classes=4):
    """
    Меняет форму числового массива жанров, видимо чтобы сделать из них нормальные метки

    :labels (массив из чисел): список жанров
    :num_classes (число): количество каких-то классов

    :return: тот же массив жанров, но другой формы
    :rtype: массив нампи
    """
    # labels это массив из чисел (см. genreDict на строке 13 в этом же файле)
    assert len(labels) > 0

    if num_classes is None:
        # меняем num_labels с дефолтного значения на значение самого большого числа в массиве labels
        num_classes = np.max(labels)+1
    else:
        # num_classes должен быть больше нуля иначе assertion error
        assert num_classes > 0
        # num_classes должен быть больше или равен самого большого числа в массиве labels
        assert num_classes >= np.max(labels)

    # создает 2д массив длины len(labels) заполненный ноликами
    # н-р ...
    # [in] labels = [0,1,2,0], len(labels) = 4, num_classes = 4
    # [out] array([[0., 0., 0., 0.],
    #              [0., 0., 0., 0.],
    #              [0., 0., 0., 0.],
    #              [0., 0., 0., 0.]])
    result = np.zeros(shape=(len(labels), num_classes))

    # заменят единицами нули в 2д массиве соответственно индексам в массиве labels
    # labels = [0,1,2,0]
    # >>> result[np.arange(len(labels)), labels] = 1
    # >>> result
    # array([[1., 0., 0., 0.], # labels[0] = 0
    #        [0., 1., 0., 0.], # labels[1] = 1
    #        [0., 0., 1., 0.], # labels[2] = 2
    #        [1., 0., 0., 0.]]) # labels[3] = 0
    result[np.arange(len(labels)), labels] = 1

    # возвращаем нампи массив
    return np.array(result)

if __name__ == "__main__":
    trainingPath = '../../gtzanDataset'
    train_data, train_labels = extract_features(trainingPath)

    # store preprocessed data in serialised format so we can save computation time and power
    # создает и пишет в два отдельных файла, один со спектрограммами, а другой с метками
    with open('../../4GenreTest.data', 'w') as f:
        pickle.dump(train_data, f)

    with open('../../4GenreTest.labels', 'w') as f:
        pickle.dump(train_labels, f)
