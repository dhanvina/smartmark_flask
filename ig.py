import cv2
import numpy as np
# from google.colab.patches import cv2_imshow
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
import cvfile

# load_model = pickle.load(open('./digitRec.pkl','rb'))



model_final = load_model('model.h5')

def pre():
  cvfile.divide()
  img_size=28
  sums=[]
  w1=120
  nums=[]
  for i in range(5):
    img = cv2.imread(f"images\cell_{i}.jpg")
    for i in range(0,350,120):
      num=img[0:65,i+0:i+120]
      nums.append(num)
      # cv2.imshow("nums",num)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
      gray = cv2.cvtColor(num , cv2.COLOR_BGR2GRAY)
      resize = cv2.resize(gray,(28,28), interpolation = cv2.INTER_AREA)
      new_img = tf.keras.utils.normalize(resize, axis=1)
      #plt.imshow(new_img)
      new_img = np.array(new_img).reshape(-1,img_size,img_size,1)
      predictions =model_final.predict(new_img)
      sums.append(np.argmax(predictions))
  return(sum(sums))


