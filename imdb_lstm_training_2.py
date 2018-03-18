import pandas,numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout,Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.utils.visualize_util import plot
from keras. callbacks import TensorBoard

############################################################################################################################################

#thedata = pandas.read_csv("Data/CombinedReviewsThreeTrainData_Movie_Amazon_Newspaper.csv", sep=', ', delimiter=',', header='infer', names=None)
thedata = pandas.read_csv("Data/StanfordMovieTrain8020Split.csv", sep=', ', delimiter=',', header='infer', names=None)
np.random.seed(1337)
#x = thedata['Review']
#y = thedata['Polarity_Numeral']

train,test = train_test_split(thedata, train_size = 0.8)
x = train['Review']
tempy = x
y = train['Polarity_Numeral']
testx = test['Review']
testy = test['Polarity_Numeral']
#testx = test['Review']
#testy = test['Polarity_Numeral']
#print len(testx)
#print len(testy)
#x,y,testx,testy = train[0],train[1],test[0],test[1]
#for i in range(0,len(x)):
#	if type(x[i]) != str:
#  		print i

x = x.iloc[:].values
y = y.iloc[:].values
tk = Tokenizer(nb_words=40000, lower=True, split=" ")
tk.fit_on_texts(x)
testy = testy.iloc[:].values
############################################################################################################################################

x = tk.texts_to_sequences(x)	

############################################################################################################################################

max_len = 170
x = pad_sequences(x, maxlen=max_len)
max_features = 40000
testx = testx.iloc[:].values
tk.fit_on_texts(testx)
testx = tk.texts_to_sequences(testx)
testx = pad_sequences(testx, maxlen=max_len)
print('Build model...')
#data = pandas.read_csv("Data/MovieReviewTrainData.csv", sep=', ', delimiter=',', header='infer', names=None)
#testx = data['Review']
#testy = data['Polarity_Numeral']
############################################################################################################################################
print x[0]
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len,dropout=0.3))
#model.add(GaussianNoise(0.2))
model.add(LSTM(128 , dropout_W=0.3, dropout_U=0.3, return_sequences=True))
model.add(LSTM(56, dropout_W = 0.4, dropout_U=0.4))
model
model.add(Dense(1, W_regularizer=l2(0.2)))
model.add(Activation('sigmoid'))
model.summary()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
model.compile(loss='binary_crossentropy', optimizer=adam,metrics = ['accuracy'] )
plot(model, to_file='model.png', show_shapes=True)
tb_cb = TensorBoard(log_dir='./Graph', histogram_freq=0,
                            write_graph=True, write_images=True)
model_history = model.fit(x, y=y, batch_size=128, nb_epoch=1, verbose=1,validation_data = (testx, testy), callbacks = [tb_cb])
############################################################################################################################################
#model.save('Saved_Models_&_Weights/temporary_model_23417.h5')
#model.save_weights('Saved_Models_&_Weights/temporary_model_weights_23417.h5')
#model_history = load_model('Saved_Models_&_Weights/temporary_model_200417.h5')
#model_history.load_weights('Saved_Models_&_Weights/temporary_model_weights_200417.h5')
#accuracy = model_history.history['acc']
#loss = model_history.history['loss']#
#validation_accuracy = model_history.history['val_acc']
#validation_loss = model_history.history['val_loss']
#print accuracy
#print loss
#print validation_accuracy
#print validation_loss
#numpy_accuracy_history = np.array(accuracy)
#numpy_loss_history = np.array(loss)
#numpy_val_acc_history = np.array(validation_accuracy)
#numpy_val_loss_history = np.array(validation_loss)

############################################################################################################################################
#plt.plot(model_history.history['loss'])
#plt.plot(model_history.history['val_loss'])
#plt.title('Model Loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['Training','Validation'],loc='upper left')
#plt.savefig('loss_20417_2.png')

#plt.plot(model_history.history['acc'])
#plt.plot(model_history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['Training','Validation'], loc='upper left')
#plt.savefig('accuracy_20417_2.png')
#plt.show()

# summarize history for loss

#plt.show()

############################################################################################################################################

#model = load_model('Saved_Models_&_Weights/model_twocategories_adam_sigmoid_100_50_31_3_17.h5')
#model.load_weights('Saved_Models_&_Weights/weights_model_twocategories_adam_sigmoid_100_50_31_3_17.h5')
#model = load_model('Saved_Models_&_Weights/temporary_long_model_110417.h5')
#model.load_weights('Saved_Models_&_Weights/temporary_long_model_weights_110417.h5')

#############################################################################################################################################

#testx = pandas.Series(['I would not say I hated the movie per se ','The internet speed is not bad','pathetic customer service. Will never visit this restaurant at all','you should not watch the movie',"""This is the best movie I have ever seen.  I've seen the movie on Dutch television sometime in 1988 (?).  That month they were showing a Yugoslavian movie every Sunday night.  The next week there was another great movie (involving a train  rather than a bus) the name of which I don't remember. If you know it  please let me know! In any case  how can I get to see this movie again???? A DVD of this movie  where?? Please tell me at vannoord@let.rug.nl"""])

#tempx = testx
#print testx.iloc[2]
#testy = list(testy)
#predictions = model.predict(x)
#predictions = model.predict(testx)
#print predictions

#for i in range(0,len(predictions)):
#	if predictions[i][0] > 0.5:
#		predictions[i][0] = 1
#	else:
#		predictions[i][0] = 0
#print len(predictions)
#print predictions
#testy = [1]
#print testx[2]
#print testy[2]
#print predictions[2]# testy
#testy = [1,1,0,1]
#acc = 100*accuracy_score(predictions,testy)
#print acc
#predictions = list(predictions)
#df = pandas.DataFrame({'Review' : tempy, 'True' : testy, 'Predicted': predictions})
#df.to_csv('Predictions_Train.csv')
#y_f = precision_recall_fscore_support(predictions,y)
#print x_acc
#df = pand
#print y_f
#df = pandas.DataFrame({'predicted' : predictions, 'true' : testy})
#df.to_csv('results_200417_model2_0.3_0.2.csv')
