#Load and clean the data
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from pickle import dump

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	tokens = ' '.join(tokens)
	return tokens

# load all docs in a directory
def process_docs(directory, is_train):
	documents = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load the doc
		doc = load_doc(path)
		# clean doc
		tokens = clean_doc(doc)
		# add to list
		documents.append(tokens)
	return documents

# save a dataset to file
def save_dataset(dataset, filename):
	dump(dataset, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load all training reviews
data_loc = "/Users/tfrisby/Downloads/review_polarity/"
negative_docs = process_docs(data_loc+'txt_sentoken/neg', True)
positive_docs = process_docs(data_loc+'txt_sentoken/pos', True)
trainX = negative_docs + positive_docs
trainy = [0 for _ in range(900)] + [1 for _ in range(900)]
save_dataset([trainX,trainy], 'train.pkl')

# load all test reviews
negative_docs = process_docs(data_loc+'txt_sentoken/neg', False)
positive_docs = process_docs(data_loc+'txt_sentoken/pos', False)
testX = negative_docs + positive_docs
testY = [0 for _ in range(100)] + [1 for _ in range(100)]
save_dataset([testX,testY], 'test.pkl')


#build the model
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt 

# load a clean dataset
def load_dataset(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# calculate the maximum document length
def max_length(lines):
	return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
	# integer encode
	encoded = tokenizer.texts_to_sequences(lines)
	# pad encoded sequences
	padded = pad_sequences(encoded, maxlen=length, padding='post')
	return padded

# define the model
def define_model(length, vocab_size):
	# channel 1
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, 100)(inputs1)
	print(embedding1.shape)
	conv1_1 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding1)
	conv1_2 = Conv1D(filters=32, kernel_size=5, activation='relu')(embedding1)
	conv1_3 = Conv1D(filters=32, kernel_size=7, activation='relu')(embedding1)
	drop1_1 = Dropout(0.5)(conv1_1)
	drop1_2 = Dropout(0.5)(conv1_2)
	drop1_3 = Dropout(0.5)(conv1_3)
	pool1_1 = MaxPooling1D(pool_size=1)(drop1_1)
	pool1_2 = MaxPooling1D(pool_size=1)(drop1_2)
	pool1_3 = MaxPooling1D(pool_size=1)(drop1_3)
	flat1_1 = Flatten()(pool1_1)
	flat1_2 = Flatten()(pool1_2)
	flat1_3 = Flatten()(pool1_3)
	# channel 2
	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, 100)(inputs2)
	conv2_1 = Conv1D(filters=16, kernel_size=1, activation='relu')(embedding2)
	conv2_2 = Conv1D(filters=16, kernel_size=3, activation='relu')(embedding2)
	conv2_3 = Conv1D(filters=16, kernel_size=5, activation='relu')(embedding2)
	drop2_1 = Dropout(0.5)(conv2_1)
	drop2_2 = Dropout(0.5)(conv2_2)
	drop2_3 = Dropout(0.5)(conv2_3)
	pool2_1 = MaxPooling1D(pool_size=1)(drop2_1)
	pool2_2 = MaxPooling1D(pool_size=1)(drop2_2)
	pool2_3 = MaxPooling1D(pool_size=1)(drop2_3)
	flat2_1 = Flatten()(pool2_1)
	flat2_2 = Flatten()(pool2_2)
	flat2_3 = Flatten()(pool2_3)
	'''
	# channel 3
	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, 100)(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	'''
	# merge
	merged = concatenate([flat1_1,flat1_2,flat1_3,
	                      flat2_1,flat2_2,flat2_3])
	# interpretation
	dense1 = Dense(10, activation='relu')(merged)
	outputs = Dense(1, activation='sigmoid')(dense1)
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# summarize
	print(model.summary())
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	return model

# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
print(trainX.shape)

# define model
model = define_model(length, vocab_size)
# fit model
history = model.fit([trainX,trainX], array(trainLabels), epochs=5, batch_size=16)
# save the model

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
plt.clf()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

#model.save('model.h5')











