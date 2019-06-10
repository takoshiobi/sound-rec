import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
np.set_printoptions(threshold='nan')

############################## ДУБЛИРОВАННЫЕ ФУНКЦИИ ИЗ predict.py ###############################


#function to create convolutional layers
def conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),name=name+'_W')

    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')
    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, pool_shape[0], pool_shape[1], 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,padding='SAME')

    return out_layer
#function to get batch data for given epoch
def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBatch:
        return data[startOfBatch:endOfBatch], labels[startOfBatch:endOfBatch]
    else:
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBatch]))
        labelsBatch = np.vstack((labels[startOfBatch:],labels[:endOfBatch]))

        return dataBatch, labelsBatch


###################################### ОБУЧЕНИЕ ##############################################

#import training dataset here using pickle
trainData = []
trainLabels = []

# подгрузка числовых массивов с данными (спкектрограммами) и с метками 
trainData = joblib.load('training.data')
trainLabels = joblib.load('training.labels')
extraHipHopData = joblib.load('hiphop.data')
extraHipHopLabels = joblib.load('hiphop.labels')
extraClassicalData = joblib.load('classical.data')
extraClassicalLabels = joblib.load('classical.labels')

# добвляем в массив trainData другой массив extraHipHopData 
# походу trainData это 2д массив, а extraHipHopData это 1д массив
trainData = np.append(trainData,extraHipHopData, axis=0)
# добавляем туда же extraClassicalData
trainData = np.append(trainData,extraClassicalData, axis=0)

# добавляем в основной массив с метками массив extraHipHopLabels...
trainLabels = np.append(trainLabels,extraHipHopLabels,axis=0)
# и extraClassicalLabels
trainLabels = np.append(trainLabels,extraClassicalLabels,axis=0)

# консоль логи с кортежами, описывающими форму массивов trainData и trainLabels
print trainData.shape
print trainLabels.shape


# перемешиваем тренирочные данные, чтобы избежать переобучения
# метод permutation: прооизвольно переставляет местами элементы в последовательности 
# возвращаем массив с перемешанными элеметами 
permutation = np.random.permutation(trainData.shape[0])
trainData = trainData[permutation]
trainLabels = trainLabels[permutation]

# использует GTZAN датасет на этапе оценки

# тестировочный массив для треков
testData = []
# тестировочный массив для меток 
testLabels = []
# подгружаем данные из файлов test.data и test.labels и заполняем ими инициализированные выше массивы
with open('test.data', 'r') as f:
  testData = pickle.load(f)
with open('test.labels', 'r') as f:
  testLabels = pickle.load(f)

# снова перемешиваем тренирочные данные, чтобы избежать переобучения, но уже для testData и testLabels
permutation = np.random.permutation(testData.shape[0])
testData = testData[permutation]
testLabels = testLabels[permutation]

# оптимальные переменные для питонских нейронных дел
# лучше поменять чтобы не состариться пока ждешь пока модель обучится
learning_rate = 0.001
num_of_epochs = 236
batch_size = 10
dropout = 0.75



############################### ДУБЛЬ КОДА ИЗ predict.py строка 140 #############################




# input x - for 1288 x 128 pixels = 164864 - this is the flattened image data that is drawn from
x = tf.placeholder(tf.float32, [None, 164864])
x_reshaped = tf.reshape(x, [-1, 1288, 128, 1]) # dynamically reshape the input
y = tf.placeholder(tf.float32, [None, 3]) # placeholder for corresponding labels- 4 genres
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# create some convolutional layers
hidden_layer1 = conv_layer(x_reshaped, 1, 128, [4, 4], [4, 4], name='layer1')
hidden_layer2 = conv_layer(hidden_layer1, 128, 64, [4, 4], [2, 2], name='layer2')
hidden_layer3 = conv_layer(hidden_layer2, 64, 32, [4, 4], [2, 2], name='layer3')


flattened = tf.reshape(hidden_layer3, [-1,  81 * 8 * 32])

# setup some weights and bias values for the fully connected layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([81 * 8 * 32, 1024], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
#Relu activation
dense_layer1 = tf.nn.relu(dense_layer1)
#Apply dropout here
dense_layer1 = tf.nn.dropout(dense_layer1, keep_prob)
# Softmax Classifier layer
wd2 = tf.Variable(tf.truncated_normal([1024, 3], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([3], stddev=0.01), name='bd2')
logits = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(logits)



################################## КОНЕЦ ДУБЛИРОВАНИЯ #################################



# перекрестная энтропия (кросс-энтропия)
# метод reduce mean: вычисляет среднее значение элементов по размерам тензора.
# метод softmax_cross_entropy_with_logits: вычисляет кросс-энтропию softmax между логитами и метками.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

"""
Перекрестная энтропия — это показатель, который можно использовать для оценки точности 
вероятностных прогнозов. Перекрестная энтропия тесно связана с оценкой максимального правдоподобия.
Перекрестная энтропия крайне важна для современных систем прогнозирования, потому что она делает
возможным создание высокоточных прогнозов, даже для альтернативных показателей.
https://www.lokad.com/ru/%D0%BF%D0%B5%D1%80%D0%B5%D0%BA%D1%80%D0%B5%D1%81%D1%82%D0%BD%D0%B0%D1%8F-%D1%8D%D0%BD%D1%82%D1%80%D0%BE%D0%BF%D0%B8%D1%8F-%D0%BE%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5
"""

# Оптимизатор, который реализует алгоритм Адама.
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

"""
Adam — adaptive moment estimation, ещё один оптимизационный алгоритм. Он сочетает в себе и идею
накопления движения и идею более слабого обновления весов для типичных признаков. 
https://habr.com/ru/post/318970/
"""

# задать точность оценки
# метод equal возращает булевский тензор со значениями True если tf.argmax(y, 1) равен tf.argmax(y_, 1)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# точность оценки 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# инициализируем глобальные переменные 
init_op = tf.global_variables_initializer()

# позволяет сохранить веса и отклонения модели для дальнейшего использования после тренировки
saver = tf.train.Saver()

with tf.Session() as sess:
    # инициализация переменных
    sess.run(init_op)
    # 10 раз обходим данные (типа 10 эпох создаем, хотя на линии 106 эпох 236) и обучаем нейронту потому что batch_size (размер пакета) = 10 омагад что за быдлокод... пиздец...
    # строка 107
    for i in range(10):
        # вот тут он считает эпохи от 1 до 10 и выводит их в консоль
        # видимо создатель кода решил что 236 эпох это дохуя для тестирования этой фигни и заменил количество
        # эпох на более реалистичное, а обратно поменять забыл. кек
        print "epoch ", (i+1)
        for epoch in range(num_of_epochs):
            # средняя кросс-энтропия для каждой эпохи
            avg_cost = 0
            # for i in range(batch_size):
            # получаем спектры треков batch_x и метки треков batch_y
            batch_x, batch_y = getBatch(trainData, trainLabels, batch_size, epoch)
            # train network with batch data
            # обучаем нейронку при помощи полученных данных 
            _, cost, batch_acc = sess.run([optimiser, cross_entropy,accuracy],feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            # выводим в консоль точность вывода и потери для партии
            print "Mini-Epoch:", (epoch + 1) , "accuracy: {:.3f}".format(batch_acc), "loss: {:.3f}".format(cost)

    # консолим
    print "Training complete! Now time to save the model"
    # сохраняем модель
    saver.save(sess, "tmp/model.ckpt")
    
    # цикл делаем 30 раз, хз почему именно 30, а не 31
    # просто финальную точность выводит в консоль и софтмакс, ничего важного
    for z in range(30):
        finalData,finalLabels = getBatch(testData, testLabels, 10, z)
        final_acc, softmaxOutput = sess.run([accuracy,y_], feed_dict={x: finalData, y: finalLabels,keep_prob: 1.0})
        print "final accuracy:  ",final_acc
        print softmaxOutput
