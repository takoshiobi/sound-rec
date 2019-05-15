import librosa
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
%matplotlib inline
# rm later

# load audio files
y1, sr1 = librosa.load('../samples/1.mp3')
y2, sr2 = librosa.load('../samples/2.mp3')

# show multiple plots using subplot
plt.subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1,sr1)   # compute MFCC values
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

dist, cost, path = dtw(mfcc1.T, mfcc2.T)
print("The normalized distance between the two : ",dist)   # 0 for similar audios

plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.plot(path[0], path[1], 'w')   # create plot for DTW

plt.show()  # display the plots graphically
