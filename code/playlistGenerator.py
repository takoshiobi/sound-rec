import numpy as np
from sklearn.externals import joblib
from math import *
import operator

# словарь, где будут храниться пары: песня-мел_спектрограмма_этой_песни (типы данных: строка-массив)
# пример: {Был-пацан.mp3: [1,1,0,1,...], ...}
songLibrary = {}
counter = 0
# predictions = joblib.load('UserTestSongs.prediction')
# подгружаем полученные в predictions.py предсказания
predictions = joblib.load('predictions.data')
rockPredictions = joblib.load('extraRock.prediction')
userPredictions = joblib.load('UserChosenSongs.prediction')

# сопоставляем полученные предсказания с названиями треков
with open('songTitles.txt') as f:
   for line in f:
       songLibrary[line.strip('\n')] = predictions[counter]
       counter += 1

# incubus.txt содержит названия хитов лучшей в мире группы incubus, записанные в этот файл построчно
# мы сопоставляем каждую песню с предсказаниями полученными для жанра роцк и добавляем эти пары в словарь
# songLibrary
rockCounter = 0
with open('incubus.txt') as f:
   for line  in f:
       songLibrary[line.strip('\n')] = rockPredictions[rockCounter]
       rockCounter += 1

# берем одну из песен великолепной группы incubus и ищем топ 10 самых похожих на неё в созданном ранее словаре songLibrary
querySong = "Cocoa Butter Kisses (ft Vic Mensa & Twista) (Prod by Cam for JUSTICE League & Peter Cottont (DatPiff Exclusive)"

# получаем массив со спектрограммами соответствующими анализируемым песням
# элементы в массиве predictions в данном случае представляют собой мел спектрограмму трека 
querySongData = songLibrary[querySong]

del songLibrary[querySong]
# del songLibrary['Big Sean - How It Feel (Lyrics)']
# del songLibrary['The Game - Ali Bomaye (Explicit) ft. 2 Chainz, Rick Ross']
# del songLibrary['Kendrick Lamar - Money Trees (HD Lyrics)']
# del songLibrary['Faint (Official Video) - Linkin Park']
# del songLibrary['Wale-Miami Nights (Ambition)']
# del songLibrary['Wale - Bad Girls Club Ft. J Cole Official Video']
# 3. find top 10 closest songs

# 
topSongs = {}

for key, value in songLibrary.iteritems():
    # calculate distance
    dist = np.linalg.norm(querySongData-songLibrary[key])
    # store in distance directory
    topSongs[key] = dist

# order top songs by distance
sortedSongs = sorted(topSongs.items(), key=operator.itemgetter(1))
# take top 10 closest songs
sortedSongs = sortedSongs[:10]

for value in sortedSongs:
    print value


# for visualisation get coordinates of top 10 songs
topSongDistances = {}
for val in sortedSongs:

    topSongDistances[val[0]] = songLibrary[val[0]]

topSongDistances[querySong] = querySongData

# print topSongDistances

# joblib.dump(topSongDistances, 'kanyeHeartless.playlist')
