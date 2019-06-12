#IMPORT STATEMENTS

import pandas as pd
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow import keras

#LOADING DATA

english_df = pd.read_csv("english_text.csv")
hinglish_df = pd.read_csv("hinglish_text.csv")

#ADDING AN ADDITIONAL LABELS COLUMN

english_df["label"] = 0
hinglish_df["label"] = 1

#MIXING BOTH DATASETS AND THEN SEPARATING TEXTS AND LABELS 

mixed_df = english_df.append(hinglish_df)
random_mixed_df = mixed_df.sample(len(mixed_df)).values
random_mixed_df= random_mixed_df[:,1:]

labels = []
texts= []
for i in range(len(random_mixed_df)):
    
    texts.append(random_mixed_df[i][0])
    labels.append(random_mixed_df[i][1])
    

#CREATING A COUNTER OBJECT TO KEEP TRACK OF OCCURENCE OF WORDS

english_counts = Counter()
hinglish_counts = Counter()
total_counts = Counter()

#UPDATING THE COUNTERS FOR ALL OUR TEXTS

for i in range(len(texts)):
    r =texts[i].split(' ')
    if(labels[i] == 0):
        english_counts.update(r)
    else:
        hinglish_counts.update(r)
    
    total_counts.update(r)
    r=[]

#CONVERTING EACH WORD IN A SENTENCE TO THEIR COUNTER VALUES IN THE VOCAB(total_counts)

layer_0 = []
for sentence in texts:
    temp=[]
    words = sentence.split(' ')
    
    for x in words:
        temp.append(total_counts[x])
    
    layer_0.append(temp)


#PADDING EACH VECTOR SO AS TO HAVE EQUAL LENGTH VECTORS TO GIVEN AS AN INPUT TO OUR NEURAL NETWORK

x_train = keras.preprocessing.sequence.pad_sequences(layer_0,value=0,padding='post',maxlen=971)


#CREATING OUR DEEP LEARNING MODEL

vocab_size = len(total_counts)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()

#COMPILING THE MODEL

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

#SPLITING TRAINING AND TESTING SETS

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = labels[:10000]
partial_y_train = labels[10000:]


#TRAINING THE MODEL ON TRAINING SET (partial_x_train and partial_y_train)

history = model.fit(partial_x_train,partial_y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)


#VALIDATION TESTING

model.evaluate(x_val,y_val)


#CREATING A FUNCTION TO PREDICT A CLASSIFICATION FOR THE TYPE OF LANGUAGE WHICH IS PASSED

def Testing(sentence):
    
    prediction_vector = []
    temp = []
    word_list = sentence.split(' ')
    
    for x in word_list:
        
        temp.append(total_counts[x])
        
    prediction_vector.append(temp)
 

    vector = keras.preprocessing.sequence.pad_sequences(prediction_vector,value=0,padding='post',maxlen=971)
    
    a = model.predict(vector)
    
    if(a[0][0]<0.45):
        return("English Sentence")
    else:
        return("Hinglish Sentence")
        


#TESTING OUR MODEL WITH NEW SENTENCES

prediction1 = Testing("This is a Test Sentence")
prediction2 = Testing("tum idhar kya kar rahe ho")

print(prediction1)
print(prediction2)
