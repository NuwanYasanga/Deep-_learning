import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import data_prep


all_users = pd.read_csv('C:/Users/s3929438/all_features_desktop_100_latest_final_all.csv')

X_train, y_train, X_test, y_test = data_prep(all_users)

max_sequence_length = max([len(x) for x in X_train])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(1, max_sequence_length)))
#model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(128)))
#model.add(BatchNormalization()) 
model.add(Dropout(0.5))

model.add(Dense(y_train_encoded.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_reshaped, y_train_encoded, epochs=25, batch_size=8)


loss, accuracy = model.evaluate(X_test_reshaped, y_test_encoded)
print('Test accuracy:', accuracy)

#predictions = model.predict(X_test_reshaped)
#pred_prob = model.predict_prob(X_test_reshaped)
#predicted_classes = np.argmax(predictions, axis=1)
#true_classes = np.argmax(y_test_encoded, axis=1)



#cm = confusion_matrix(true_classes, predicted_classes)


#plt.figure(figsize=(10, 10))  # Increase figure size for larger confusion matrices
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 'fmt' specifies the string formatting; 'd' means decimal
#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
#plt.title('Confusion Matrix')
#plt.show()



