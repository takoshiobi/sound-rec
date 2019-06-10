import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
np.set_printoptions(threshold='nan')

# function to create convolutional layers
def conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    """
    Создает сверточные слои, охуеть

    :input_data: входные данные, массив (4-D тензор)
    :num_input_channels: количество входных каналов, число
    :num_filters: количество фильтров, число 
    :filter_shape: форма фильтра, массив из 2х чисел
    :pool_shape: тоже массив из 2х чисел 
    :name: название слоя

    :return: сверточный слой
    :rtype: 4-D тензор в виде массива чисел
    """
    # setup the filter input shape for tf.nn.conv_2d
    # форма фильтра изпользуемого в tf.nn.conv_2d
    # пример
    # filter_shape = [4,2]
    # num_input_channels = 3
    # num_filters = 10
    # тогда conv_filt_shape = [4, 2, 3, 10] просто числовой массив
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    # tf.Variable : создает тензор любого типа и формы
    # tf.truncated_normal:
    # (conv_filt_shape): 1-D целочисленный тензор или массив Python. Форма выходного тензора.
    # (stddev): 0-D Tensor. Стандартное отклонение нормального распределения, перед усечением.
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),name=name+'_W')

    # num_filters: количество фильтров. В этом случае используется для создания 1D тензора для вычисления смещений (погрешностей).
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    # conv2d: вычисляет двумерную свертку с учетом четырехмерных входных и фильтрующих тензоров. 
    #  возвращает 4-D тензор того же типа что и input_data
    # (padding) тип используемого алгоритма заполнения
    # (input_data) 4-D тензор в виде массива чисел 
    # (weights) тоже тензор 4д типа [высота_фильтра, ширина_фильтра, каналы_входа, каналы_выхода]
    # ([1,1,1,1]) численный массив 1-D тензора длины 4. Шаг перемещения окна (блока) в зависимости от формы входного массива input_data.
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    # плюсуем погрешность 
    out_layer += bias

    # apply a ReLU non-linear activation
    # что такое ReLU и нахуй оно надо: http://datareview.info/article/eto-nuzhno-znat-klyuchevyie-rekomendatsii-po-glubokomu-obucheniyu-chast-2/
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    # пулинг хуюлинг
    # опять пример для тупых вроде меня
    # pool_shape = [4,2]
    # тогда
    # ksize = [1, 4, 2, 1]
    # strides = [1, 4, 2, 1]
    # одинаковые, да
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_shape[0], pool_shape[1], 1]
    
    # max_pool: делает максимальный пулинг входных данных, возвращает тензор (4-D в нашем случае)
    # (out_layer): 4-D тензор, который прошел через выпрямитель ReLU
    # (ksize): 1-D тензор из 4х чисел, размер окна для каждого измерения входного массива
    # (strides): 1-D тензор из 4х чисел, щаг скольжения окна для каждого измерения входного тензора. 
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,padding='SAME')

    # возвращает сверточный слой (4-D тензор) в виде массива чисел
    return out_layer

# функция для получения партии данных за заданную эпоху
# эпоха это один проход вперед и один проход назад по всем обучающим примерам
def getBatch(data, batchSize, iteration):
    """
    Получение данных за заданную эпоху

    :data (массив): входные данные
    :batchSize (число): размер пакета или количество обучающих примеров за один проход вперед / назад. 
    :iteration (число): количество проходов, каждый проход с использованием количества примеров равного размеру пакета.
    Еще, один проход = один проход вперед + один проход назад (мы не считаем проход вперед и обратный проход как два разных прохода).

    :return: 
    :rtype: массив
    """
    # начало серии 
    # пример
    # data = [1,2,3,4,5,6]
    # iteration = 2
    # batchSize = 10
    # startOfBatch = (2 * 10) % len([1,2,3,4,5,6]) = 20 % 6 = 2 (остаток от деления 20 на 6)
    startOfBatch = (iteration * batchSize) % len(data)
    # конец серии/пакета/хз чего
    # endOfBatch = (2 * 10 + 10) % len([1,2,3,4,5,6]) = 30 % 6 = 0 (остаток от деления 30 на 6, делится четко без остатка)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    # если начало пакета меньше чем конец
    # у нас startOfBatch=2, а endOfBatch=0, значит не наш случай
    if startOfBatch < endOfBatch:
        # если data = [1,2,3,4,5,6]
        # data[2:0] вернет пустой массив, потому что не наш случай 
        # если поменять местами, например, data[0:2], то вернет [1,2] 
        # то есть все элементы между индексами 0 и 2, исключая последний элемент
        return data[startOfBatch:endOfBatch]
    # наоборот, наш случай
    else:
        # вернет cтек массив с вертикальной последовательностью элементов
        # массивы data[startOfBatch:] и data[:endOfBatch] должны быть равной длины
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBatch]))


        return dataBatch




#################################### ПРИМЕНЕНИЕ МЕТОДОВ НА ПРАКТИКЕ ################################



# userChosenSongs.data это скорее всего большой числовой массив с треками, который выбрал пользователь
# для сравнения 
# joblib.load должен подгружать модель которая создана уже при помощи joblib.dump, без этого никак
# так что предположим что она была создана при запуске feature_converter.py
targetData = joblib.load('userChosenSongs.data')

# выводит в консоль форму массива в виде кортежа (n,m), где n количество строк, а m число коллонок 
# [[1,1,1],[1,1,1]] => (2,3)
print targetData.shape

# количество эпох
num_of_epochs = 2
# размер пакета
batch_size = 7

# placeholder: местазаполнитель для тензора
# (tf.float32) тип данных тензора, тут тензор заполнен флоатами ординарной точности (32)
# ([None, 164864]) форма тензора, None в аргументе shape говорит о том, что это измерение не определено, и выводит это измерение из тензора, который передается методу во время его выполнения.
# x - для 1288 x 128 пикселей = 164864 - это данные сглаженного изображения, которые извлекаются из ...
x = tf.placeholder(tf.float32, [None, 164864])
# меняем форму входного массива тензора 
# (x) сам тензор
# ([-1, 1288, 128, 1]) новая 4д форма тензора, где -1 обозначает автоматическое расширение тензора по одному из измерений
x_reshaped = tf.reshape(x, [-1, 1288, 128, 1]) # dynamically reshape the input

# то же самое только без формы что и на строке 142
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# create some convolutional layers
# conv_layer(тензор, количество_входных_каналов, количество_фильтров, форма_фильтра, форма_пула, название_слоя)
hidden_layer1 = conv_layer(x_reshaped, 1, 128, [4, 4], [4, 4], name='layer1')
hidden_layer2 = conv_layer(hidden_layer1, 128, 64, [4, 4], [2, 2], name='layer2')
hidden_layer3 = conv_layer(hidden_layer2, 64, 32, [4, 4], [2, 2], name='layer3')

# трансформируем последний полученный слой в 2д тензор 
flattened = tf.reshape(hidden_layer3, [-1,  81 * 8 * 32])
# truncated_normal: выводит случайные значения для усеченного нормального распределения.
# задает вес и погрешности для тензора
# ([81*8*32, 1024]) размер 1-D тензора который возвращает этот метод
# (stddev) 0-D тензор. Стандартное отклонение нормального распределения перед усечением. 
wd1 = tf.Variable(tf.truncated_normal([81 * 8 * 32, 1024], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
# умножаем тензор flattened на тензор(матрицу) веса wd1 и прибавляем погрешность bd1
dense_layer1 = tf.matmul(flattened, wd1) + bd1
# "выпрямляем" данные тензора при помощи ReLU
dense_layer1 = tf.nn.relu(dense_layer1)
# Дропаут чтобы не было перегрузки в нейронке 
# норм статья про дропауты: http://laid.delanover.com/dropout-explained-and-implementation-in-tensorflow/
# на русском: https://habr.com/ru/company/wunderfund/blog/330814/
dense_layer1 = tf.nn.dropout(dense_layer1, keep_prob)

# Softmax Classifier layer
"""
Softmax — это обобщение логистической функции для многомерного случая. Функция преобразует вектор
z размерности  K в вектор sigma той же размерности, где каждая координата
полученного вектора представлена вещественным числом в интервале [0,1] и сумма координат равна 1.
"""
# всё то же самое, что и выше. Делаем слой для софтмакса.
wd2 = tf.Variable(tf.truncated_normal([1024, 3], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([3], stddev=0.01), name='bd2')
logits = tf.matmul(dense_layer1, wd2) + bd2
# softmax: вычисляет софтмакс активации (см докстринг на строке 191)
# (logits): не пустой тензор, массив чисел
y_ = tf.nn.softmax(logits)

"""
Функция активации (активационная функция, функция возбуждения) – функция, вычисляющая выходной
сигнал искусственного нейрона.
http://www.aiportal.ru/articles/neural-networks/activation-function.html
"""


# global_variables_initializer: оператор, который инициализирует глобальные переменные в графе.
init_op = tf.global_variables_initializer()

# позволяет нам сохранять и восстанавливать веса и отклонения модели для дальнейшего использования после тренировки
# класс Saver() cохраняет и восстанавливает переменные.
saver = tf.train.Saver()

# массив с предсказаниями 
predictions = []

# объект Session инкапсулирует среду, в которой выполняются классы операций, и оцениваются объекты Tensor.
with tf.Session() as sess:
    # initialise the variables
    # метод run запускает операции и оценивает тензоры в выборках.
    sess.run(init_op)

    # восстанавливает ранее сохраненные переменные.
    saver.restore(sess, 'tmp/model.ckpt')

    # в цикле запускаем заданное количество эпох num_of_epochs
    for epoch in range(num_of_epochs):
        # для каждой эпохи прогоняем данные через нейросеть
        data = getBatch(targetData,batch_size,epoch)
        # предсказываем и сохраняем выходные данные для этих песен используя софтмакс 
        # аргумент feed_dict позволяет вызывающей стороне переопределять значение тензоров в графе.
        # словарь, ассоциирующий элементы графа со значениями
        # (y_) Один элемент графа, список элементов графа или словарь, значения которого являются элементами графа или списками графа
        # (y_) инициализируется на линии 189
        softmaxOutput = sess.run(y_, feed_dict={x: data,keep_prob: 1.0})

        # выводим софтмакс в консоль в целях дебагинга 
        print softmaxOutput
        # если эпоха равна нулю, список предсказаний равен массиву софтмакса
        if epoch == 0:
            predictions = softmaxOutput
        else:
            """
            np.vstack 
            Складывает массивы в последовательности по вертикали (по рядам).
            Берет 2 массива predictions и softmaxOutput и складывает их вертикально, чтобы сделать один массив.
            """
            predictions = np.vstack((predictions,softmaxOutput))

# консольные логи
print predictions.shape
np.set_printoptions(suppress=True)
print predictions

# сохраням прогнозы в сериализованном формате для последующей обработки
joblib.dump(predictions,'UserChosenSongs.prediction')
