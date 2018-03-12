
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from nextbatch import next_batch

data = pd.read_csv('Iris.csv')
data.head()

x_data = data.drop(['Species'], axis=1)
y_data = data['Species']

encoder = LabelEncoder()

label_encoded = encoder.fit(y_data).transform(y_data)
hotencoder = OneHotEncoder(3)
y_data = hotencoder.fit_transform(label_encoded).toarray().reshape(150,3)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = pd.DataFrame(scalar.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scalar.transform(X_test), index=X_test.index, columns=X_test.columns)

y_true = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([5,3]))
b = tf.Variable(tf.random_uniform([3]))

y_pred = tf.add(tf.matmul(x,W), b)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(error)

init = tf.global_variables_initializer()
num_epochs = 1000

saver = tf.train.Saver()

with tf.Session() as sess:
    
    sess.run(init)
    accs = []
    epoch_list=[]
    
    for epoch in range(num_epochs):
        
        x_batch, y_batch = next_batch(X_train, y_train)
        
        sess.run(optimizer, feed_dict = {x:x_batch, y_true:y_batch})
        
        if epoch%10==0:
            print('Epoch number {}'.format(epoch))

            
            preds = sess.run(y_pred, feed_dict={x: X_train})
                        
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y_train, 1)), tf.float32))
            acc = sess.run(acc)
            
            print("Current accuracy = {}".format(acc))

            # TODO: make a dynamic graph that updates as the accuracy changes

            # accs.append(acc)
            # epoch_list.append(epoch)
            # plt.plot(epoch_list, accs)
            # plt.show()
            
    
    preds = sess.run(y_pred, feed_dict={x: X_train})
    acc_train = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y_train, 1)), tf.float32))
    acc_train = sess.run(acc_train)
    
    preds = sess.run(y_pred, feed_dict={x: X_test})
    acc_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y_test, 1)), tf.float32))
    acc_test = sess.run(acc_test)
    
    print("Final train Accuracy = {}".format(acc_train))
    print("Test accuracy = {}".format(acc_test))

    #final_w, final_b = sess.run([W, b])

    tmp = saver.save(sess, 'models/model.ckpt')

    #To load this model, uncomment the following lines:
    '''
    saver.restore(sess, 'models/model.ckpt')
    restored_slope , restored_intercept = sess.run([W,b])
    '''