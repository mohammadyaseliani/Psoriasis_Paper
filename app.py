# Importing the Necessary Libraries for Loading the Models 

 
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import requests
import tensorflow.compat.v1.keras.backend as K
from streamlit_option_menu import option_menu

st.set_page_config(layout="centered")

@st.cache(allow_output_mutation=True)
# The Function of Loading the Hybrid CNN Model for Feature Extraction
def load_classifiers():
  url1 = 'https://drive.google.com/u/0/uc?id=19yQXM-v_Q0h9sQGdqmGgiDjkqXAj7pQG&export=download'
  r1 = requests.get(url1, allow_redirects=True)
  url2 = 'https://drive.google.com/u/0/uc?id=1-2oglfOHsJ8BsJ98q5b8xLBiytGM0ez3&export=download'
  r2 = requests.get(url2, allow_redirects=True)
  url3 = 'https://drive.google.com/u/0/uc?id=1-5r031ZBkGVitnuV4Xu8z0Hx2ZMwqWYT&export=download&confirm=t'
  r3 = requests.get(url3, allow_redirects=True)
  url4 = 'https://drive.google.com/u/0/uc?id=1-3SVUDHrKmV-dUz1uppdLrIoV_favOZt&export=download&confirm=t'
  r4 = requests.get(url4, allow_redirects=True)
  url5 = 'https://drive.google.com/u/0/uc?id=1-64EcLv277pWm4pfrlAeXLkygT2JIvKj&export=download&confirm=t'
  r5 = requests.get(url5, allow_redirects=True)
  url6 = 'https://drive.google.com/u/0/uc?id=1-4E8kBAFuoToexf_Nh1-i3MxQeQ7wC4D&export=download&confirm=t'
  r6 = requests.get(url6, allow_redirects=True)
  url7 = 'https://drive.google.com/u/0/uc?id=1C2X4worq9sGrBlURKLzqucDtDYQCvBA-&export=download'
  r7 = requests.get(url7, allow_redirects=True)
  url8 = 'https://drive.google.com/u/0/uc?id=1-5zYystBU_xRA4iihAcRUqsU3EC5QUnm&export=download'
  r8 = requests.get(url8, allow_redirects=True)
  with open('ResNet50V2_Binary.hdf5', 'wb') as f:
    f.write(r1.content)
  with open('ResNet50V2_MultiClass.hdf5', 'wb') as f:
    f.write(r2.content)
  with open('ResNet101V2_Binary.hdf5', 'wb') as f:
    f.write(r3.content)
  with open('ResNet101V2_MultiClass.hdf5', 'wb') as f:
    f.write(r4.content)
  with open('ResNet152V2_Binary.hdf5', 'wb') as f:
    f.write(r5.content)
  with open('ResNet152V2_MultiClass.hdf5', 'wb') as f:
    f.write(r6.content)
  with open('weights_binary.npy', 'wb') as f:
    f.write(r7.content)
  with open('weights_MultiClass.npy', 'wb') as f:
    f.write(r8.content)
  network1=tf.keras.models.load_model('ResNet50V2_Binary.hdf5')
  network1.make_predict_function()
  network2=tf.keras.models.load_model('ResNet101V2_Binary.hdf5')
  network2.make_predict_function()
  network3=tf.keras.models.load_model('ResNet152V2_Binary.hdf5')
  network3.make_predict_function()
  network4=tf.keras.models.load_model('ResNet50V2_MultiClass.hdf5')
  network4.make_predict_function()
  network5=tf.keras.models.load_model('ResNet101V2_MultiClass.hdf5')
  network5.make_predict_function()
  network6=tf.keras.models.load_model('ResNet152V2_MultiClass.hdf5')
  network6.make_predict_function()
  weights_binary=np.load('weights_binary.npy')
  weights_MultiClass=np.load('weights_MultiClass.npy')
  session = K.get_session()
  return network1, network2, network3, network4, network5, network6, weights_binary, weights_MultiClass, session


# Importing the Necessary Libraries for Loading the Input Image and Making Predicitons
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

# The Function of Importing the Image and Predicting Its Corresponding Class 
def import_and_predict_binary(image_data, classifier1, classifier2, classifier3, weights_binary):

        # Preparing the Image
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img=img/255.      
        img_reshape = img[np.newaxis,...]
        networks=[classifier1, classifier2, classifier3]

        preds = [network.predict(img_reshape) for network in networks]
        preds=np.array(preds)
        ideal_weights = weights_binary

#Use tensordot to sum the products of all elements over specified axes.
        ideal_weighted_preds = np.tensordot(preds, ideal_weights, axes=((0),(0)))
        ideal_weighted_ensemble_prediction = np.argmax(ideal_weighted_preds, axis=1)
        return  ideal_weighted_ensemble_prediction

def import_and_predict_MultiClass(image_data, classifier4, classifier5, classifier6, weights_MultiClass):

        # Preparing the Image
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img=img/255.      
        img_reshape = img[np.newaxis,...]
        networks=[classifier4, classifier5, classifier6]

        preds = [network.predict(img_reshape) for network in networks]
        preds=np.array(preds)
        ideal_weights = weights_MultiClass

        #Use tensordot to sum the products of all elements over specified axes.
        ideal_weighted_preds = np.tensordot(preds, ideal_weights, axes=((0),(0)))
        ideal_weighted_ensemble_prediction = np.argmax(ideal_weighted_preds, axis=1)

        return  ideal_weighted_ensemble_prediction
        
def MCDM(w, X, interval):
  import numpy as np
  import math
  X=np.array(X)
  X_a=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      if j>0:
        X_a[i][j]=X[i][j]/max(X[:,j])
      else:
        X_a[i][j]=min(X[:,j])/X[i][j]
  X_b=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      if j>0:
        X_b[i][j]=X[i][j]/sum(X[:,j])
      else:
        X_b[i][j]=1/X[i][j]/(sum(1/X[:,j])) 
  X_c=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      if j>0:
        X_c[i][j]=(X[i][j]-min(X[:,j]))/(max(X[:,j])-min(X[:,j]))
      else:
        X_c[i][j]=(max(X[:,j])-X[i][j])/(max(X[:,j])-min(X[:,j]))
  X_d=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      X_d[i][j]=math.log(X[i][j])/math.log(np.prod(X[:,j]))
  h=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      h[i][j]=0.25*(X_a[i][j]+X_b[i][j]+X_c[i][j]+X_d[i][j])
  x_min=np.zeros(len(X[0]))
  for j in range(len(X[0])):
    x_min[j]=min(X[:,j])
  x_max=np.zeros(len(X[0]))
  for j in range(len(X[0])):
    x_max[j]=max(X[:,j])
  # interval: m*2 Matrix

  f=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      if j>=1:   #benefit criterion
        if X[i][j]>=interval[j][0] and X[i][j]<=interval[j][1]:
          f[i][j]=1

        elif X[i][j]>=x_min[j] and X[i][j]<=interval[j][0]:
          f[i][j]=1-(interval[j][0]-X[i][j])/((max(interval[j][0]-x_min[j],x_max[j]-interval[j][1]))+1)
    
        elif X[i][j]>=interval[j][1] and X[i][j]<=x_max[j]:
          f[i][j]=1-(1-interval[j][1]+X[i][j])/((max(interval[j][0]-x_min[j],x_max[j]-interval[j][1]))+1)
    
      if j==0:   #cost criterion
        if X[i][j]>=interval[j][0] and X[i][j]<=interval[j][1]:
          f[i][j]=1/((max(interval[j][0]-x_min[j],x_max[j]-interval[j][1]))+1)

        elif X[i][j]>=x_min[j] and X[i][j]<=interval[j][0]:
          f[i][j]=(interval[j][0]-X[i][j])/(max(interval[j][0]-x_min[j],x_max[j]-interval[j][1]))
      
        elif X[i][j]>=interval[j][1] and X[i][j]<=x_max[j]:
          f[i][j]=(X[i][j]-interval[j][1])/(max(interval[j][0]-x_min[j],x_max[j]-interval[j][1]))
    
  y=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      y[i][j]=f[i][j]*h[i][j]
  g=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      g[i][j]=y[i][j]*w[j]
  m=np.zeros(len(X[0]))
  for j in range(len(X[0])):
    m[j]=min(g[:,j])

  E_dif=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      E_dif[i][j]=g[i][j]-m[j]

  E_dif=np.array(E_dif)
  E_dif=np.power(E_dif, 2)

  E=np.zeros(len(X))

  for i in range(len(X)):
    E[i]=math.sqrt(sum(E_dif[i,:]))

  T_dif=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      T_dif[i][j]=np.absolute(g[i][j]-m[j])

  T_dif=np.array(T_dif)

  T=np.zeros(len(X))

  for i in range(len(X)):
    T[i]=sum(T_dif[i,:])

  L_dif=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      L_dif[i][j]=1+np.absolute(g[i][j]-m[j]) 
      
  L_dif=np.array(L_dif)
  L_dif=np.log(L_dif)

  L=np.zeros(len(X))

  for i in range(len(X)):
    L[i]=sum(L_dif[i,:])
  
  P_dif=np.zeros((len(X),len(X[0])))
  for i in range(len(X)):
    for j in range(len(X[0])):
      P_dif[i][j]=E_dif[i][j]/(m[j]+0.01) ##Important

  P_dif=np.array(P_dif)

  P=np.zeros(len(X))

  for i in range(len(X)):
    P[i]=sum(P_dif[i,:])

  theta=np.zeros((len(X),len(X)))
  phi=np.zeros((len(X),len(X)))
  theta=np.array(theta)
  phi=np.array(phi)
  for i in range(len(X)):
    for k in range(len(X)):
      theta[i][k]=(E[i]-E[k])+((E[i]-E[k])*(T[i]-T[k]))
      phi[i][k]=(L[i]-L[k])+((L[i]-L[k])*(P[i]-P[k]))

  omega=np.zeros(len(X))
  for i in range(len(X)):
    omega[i]=0.5*sum(theta[i,:])+0.5*sum(phi[i,:])

  return omega

# Loading the Model
with st.spinner("The model is being loaded..."):
  classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, weights_binary, weights_MultiClass, session = load_classifiers()
  K.set_session(session)

# The Title and User Guide
st.markdown("<h1 style='text-align: center; color: black;'>A Clinical Decision Support for Psoriasis</h1>", unsafe_allow_html=True)
st.markdown('<style>body{background-color: black;}</style>',unsafe_allow_html=True)
#st.subheader('A Clinical Decision Support for Psoriasis')
st.image("Pso.png")
st.markdown("<h1 style='text-align: center; color: black;'>How this app works.</h1>", unsafe_allow_html=True)
st.image("UI.png", use_column_width = True)

files = st.file_uploader("Please Upload a skin Image (JPG, PNG, JPEG, or JFIF Format)", type=["jpg", "png", "jpeg", "jfif"])
st.session_state.files=files
if st.session_state.files is None:
  st.text("Please upload an image file")
else:
  image = Image.open(st.session_state.files)
  st.image(image, use_column_width=True)
  prediction_binary=import_and_predict_binary(image, classifier1, classifier2, classifier3, weights_binary)
  prediction_MultiClass=import_and_predict_MultiClass(image, classifier4, classifier5, classifier6, weights_MultiClass)
  result=''
  if prediction_binary==0:
    st.info('**The input image is not psoriatic. The possible skin conditions include eczema, seborrheic dermatitis, keratosis pilaris, irritant or allergic contact dermatitis, pityriasis rosea, ringworms, hives, acne, and rosacea.**')
  if prediction_binary==1:
    treatment=''
    X_erythrodermic=[[5, 4, 3, 2, 1], [5, 3, 2, 2, 1], [7, 8, 9, 2, 1], [5, 8, 3, 7, 5], [4, 9, 2, 2, 7], [5, 5, 3, 2, 9], [9, 8, 5, 7, 10]]
    X_guttate=[[5, 4, 3, 2, 1], [4, 2, 2, 5, 4], [7, 8, 9, 8, 5], [5, 4, 3, 9, 9], [4, 3, 9, 8, 7], [5, 8, 3, 7, 5], [5, 4, 3, 4, 3], [8, 3, 2, 2, 1], 
               [9, 4, 3, 1, 7], [2, 5, 3, 5, 6]]
    X_inverse=[[5, 4, 2, 1], [4, 3, 5, 4], [7, 8, 9, 1], [5, 4, 7, 8], [4, 3, 4, 5], [5, 5, 3, 2], [9, 8, 4, 3]]
    X_nail=[[5, 4, 3, 4, 5, 6, 7, 8, 9, 1, 1], [5, 8, 2, 1, 4, 6, 7, 8, 9, 1, 1], [6, 4, 3, 4, 5, 6, 7, 8, 9, 5, 1], 
            [5, 4, 3, 4, 5, 6, 2, 4, 9, 1, 1], [9, 7, 3, 4, 5, 6, 7, 2, 9, 4, 1], [5, 9, 2, 1, 5, 5, 8, 7, 8, 8, 8], 
            [6, 8, 1, 4, 5, 4, 7, 8, 9, 1, 1], [2, 4, 3, 1, 3, 6, 7, 8, 9, 8, 1], [2, 5, 3, 4, 5, 6, 7, 9, 9, 5, 7], 
            [4, 4, 2, 4, 1, 5, 1, 8, 9, 1, 10], [5, 5, 3, 1, 5, 1, 8, 4, 9, 8, 1], [3, 2, 3, 1, 5, 6, 7, 8, 9, 2, 7], 
            [5, 4, 3, 7, 2, 1, 2, 8, 9, 1, 1], [5, 4, 3, 4, 5, 6, 7, 8, 9, 1, 1], [4, 4, 3, 8, 9, 8, 7, 8, 9, 7, 1], 
            [9, 8, 7, 4, 2, 4, 4, 4, 4, 1, 1], [1, 8, 7, 4, 3, 4, 4, 5, 6, 7, 1]]
    X_pustular=[[8, 9, 3, 8, 5, 3, 8, 9, 1, 3], [5, 4, 3, 5, 5, 3, 8, 9, 4, 9], [2, 4, 3, 4, 5, 3, 8, 9, 7, 10], [5, 4, 3, 4, 3, 4, 8, 3, 5, 2], 
                [7, 4, 4, 8, 5, 7, 8, 9, 5, 2], [2, 5, 3, 4, 5, 3, 8, 9, 3, 9], [1, 4, 3, 4, 5, 3, 8, 9, 4, 1], [4, 8, 4, 9, 2, 3, 8, 9, 3, 4],
               [2, 4, 5, 5, 7, 3, 8, 9, 4, 8], [1, 4, 3, 5, 5, 3, 2, 9, 7, 8], [2, 9, 3, 4, 5, 9, 8, 9, 5, 1], [1, 4, 3, 4, 1, 3, 8, 4, 2, 3]]
    X_plaque=[[8, 9, 3, 8, 5, 3, 5, 8], [5, 4, 3, 5, 5, 3, 2, 9], [2, 4, 3, 4, 5, 6, 1, 3], [5, 4, 3, 4, 3, 4, 3, 4], 
                [7, 4, 4, 8, 5, 7, 4, 3], [2, 5, 3, 4, 5, 3, 1, 2], [1, 4, 3, 4, 5, 3, 1, 1], [4, 8, 4, 9, 2, 3, 2, 4],
               [2, 4, 5, 5, 7, 3, 2, 4], [1, 4, 3, 5, 5, 3, 3, 4], [2, 9, 3, 4, 5, 9, 8, 5], [1, 4, 3, 4, 1, 3, 4, 9],
               [4, 4, 8, 8, 7, 8, 4, 3], [2, 4, 3, 3, 5, 3, 5, 4], [1, 9, 2, 4, 3, 9, 9, 9], [1, 9, 2, 4, 3, 9, 1, 2],
             [1, 9, 2, 4, 3, 9, 5, 2]]
    X_arthritis=[[5, 4, 3, 4, 5, 6, 7, 8, 9, 1, 1], [5, 8, 2, 1, 4, 6, 7, 8, 9, 1, 1], [6, 4, 3, 4, 5, 6, 7, 8, 9, 5, 1], 
            [5, 4, 3, 4, 5, 6, 2, 4, 7, 1, 1], [9, 7, 1, 4, 1, 6, 7, 2, 9, 4, 1], [5, 9, 2, 1, 1, 5, 8, 7, 8, 8, 8], 
            [6, 8, 1, 4, 5, 4, 7, 8, 9, 1, 1], [2, 4, 1, 1, 3, 6, 7, 8, 9, 8, 1], [2, 5, 3, 4, 5, 6, 5, 5, 4, 5, 7], 
            [4, 4, 2, 4, 1, 5, 1, 8, 9, 1, 1], [5, 5, 2, 1, 5, 1, 8, 4, 9, 8, 1], [3, 2, 3, 1, 2, 6, 7, 8, 9, 2, 7], 
            [5, 4, 3, 7, 2, 1, 2, 8, 9, 1, 1], [5, 4, 3, 4, 5, 6, 7, 8, 9, 1, 1], [4, 4, 3, 8, 9, 8, 7, 8, 9, 7, 1]]
    interval_erythrodermic=[[2, 10],[2, 10],[2, 10],[2, 10], [2, 10]]
    interval_guttate=[[2, 10],[2, 10],[2, 10], [2, 10], [2, 10]]
    interval_inverse=[[2, 10],[2, 10], [2, 10], [2, 10]]
    interval_nail=[[2, 10], [2, 10],[2, 10],[2, 10],[2, 10], [2, 10],[2, 10],[2, 10], [2, 10], [2, 10], [2, 10]]
    interval_pustular=[[2, 10],[2, 10],[2, 10],[2, 10], [2, 10],[2, 10],[2, 10],[2, 10], [2, 10], [2, 10]]
    interval_plaque=[[2, 10],[2, 10],[2, 10],[2, 10], [2, 10],[2, 10], [2, 10], [2, 10]]
    interval_arthritis=[[2, 10], [2, 10],[2, 10],[2, 10],[2, 10], [2, 10],[2, 10],[2, 10],[2, 10], [2, 10], [2, 10]]
    if prediction_MultiClass==0:
      st.warning('The input skin image invloves **erythrodermic psoriasis**. Please specify the degree of your symptoms to get the treatments.')
      params = []
    #  cost = st.slider("How much can you spend?",0, 10, 1)
    # params.append(cost)      
      uveitis = st.slider("Redness and Pain of the Eye (Uveitis)",0, 4, 1)
      params.append(uveitis)
      vision= st.slider("Decline in Vision",0, 4, 1)
      params.append(vision)
      st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')
      col1_head, col2_head = st.columns(2)
          
      with col2_head:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("head.png", width=120) 
        
      with col1_head:
        percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
        if percent_head==0:
          ph=0
        elif percent_head>=1 and percent_head<=9:
          ph=1
        elif percent_head>=10 and percent_head<=29:
          ph=2
        elif percent_head>=30 and percent_head<=49:
          ph=3
        elif percent_head>=50 and percent_head<=69:
          ph=4
        elif percent_head>=70 and percent_head<=89:
          ph=5
        elif percent_head>=90 and percent_head<=100:
          ph=6
        erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
#        params.append(erythema_head)
        induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
 #       params.append(induration_head)
        desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
  #      params.append(desquamation_head)
      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      col1_arms, col2_arms = st.columns(2)

      with col2_arms:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("arms.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_arms:
        percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
        if percent_arms==0:
          pa=0
        elif percent_arms>=1 and percent_arms<=9:
          pa=1
        elif percent_arms>=10 and percent_arms<=29:
          pa=2
        elif percent_arms>=30 and percent_arms<=49:
          pa=3
        elif percent_arms>=50 and percent_arms<=69:
          pa=4
        elif percent_arms>=70 and percent_arms<=89:
          pa=5
        elif percent_arms>=90 and percent_arms<=100:
          pa=6
        erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
 #       params.append(erythema_arms)
        induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
  #      params.append(induration_arms)
        desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
   #     params.append(desquamation_arms)
        
      col1_trunk, col2_trunk = st.columns(2)
      with col2_trunk:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("trunk.png", width=120) 
      st.text(' ')  
      st.text(' ')  
      st.text(' ')        
      with col1_trunk:
        percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
        if percent_trunk==0:
          pt=0
        elif percent_trunk>=1 and percent_trunk<=9:
          pt=1
        elif percent_trunk>=10 and percent_trunk<=29:
          pt=2
        elif percent_trunk>=30 and percent_trunk<=49:
          pt=3
        elif percent_trunk>=50 and percent_trunk<=69:
          pt=4
        elif percent_trunk>=70 and percent_trunk<=89:
          pt=5
        elif percent_trunk>=90 and percent_trunk<=100:
          pt=6
        erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
  #      params.append(erythema_trunk)
        induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
   #     params.append(induration_trunk)
        desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
    #    params.append(desquamation_trunk)
               
      col1_legs, col2_legs = st.columns(2)
      with col2_legs:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("legs.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_legs:
        percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
        if percent_legs==0:
          pl=0
        elif percent_legs>=1 and percent_legs<=9:
          pl=1
        elif percent_legs>=10 and percent_legs<=29:
          pl=2
        elif percent_legs>=30 and percent_legs<=49:
          pl=3
        elif percent_legs>=50 and percent_legs<=69:
          pl=4
        elif percent_legs>=70 and percent_legs<=89:
          pl=5
        elif percent_legs>=90 and percent_legs<=100:
          pl=6
        erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
  #      params.append(erythema_legs)
        induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
   #     params.append(induration_legs)
        desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
    #    params.append(desquamation_legs)
      st.text('')
      erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
      params.append(erythema_avg)
      induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
      params.append(induration_avg)   
      desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_erythrodermic, interval_erythrodermic)
      treatment={'0':'**Methotrexate acitretin**', '1':'**Cyclosporine**', '2':'**Infliximab**', '3':'**Mild- to moderate-potency topical corticosteroids in combination with systemic treatments**',
                 '4':'**Guselkumab**', '5':'**Psoralen and Ultraviolet A (PUVA)**', '6':'**Psoralen and Combination of methotrexate and compound glycyrrhizin**'}
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
      PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PASI score and the best treatment options'):
        st.info('Your PASI score is ' +str(PASI))
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
               ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
              
    if prediction_MultiClass==1:
      st.warning('The input skin image involves **guttate psoriasis**. Please specify the degree of your symptoms to get the treatments.')
      treatment={'0':'**Phototherapy**', '1':'**Topical corticosteroids**', '2':'**Vitamin D analogs**', '3':'**Tar**',
                 '4':'**Cyclosporine**', '5':'**Adalimumab**', '6':'**Etanercept**', '7':'**Infliximab**', '8':'**Methotrexate**', '9':'**Acitretin**'}
      params = []
 #     cost = st.slider("How much can you spend?",0, 10, 1)
  #    params.append(cost)      
      lesions = st.slider("Teardrop-Shaped Lesions",0, 4, 1)
      params.append(lesions)
      plaques = st.slider("Round or Slightly Oval, Erythematous and Scaly Papules and Plaques",0, 4, 1)
      params.append(plaques)    
      st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')
      col1_head, col2_head = st.columns(2)
          
      with col2_head:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("head.png", width=120) 
        
      with col1_head:
        percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
        if percent_head==0:
          ph=0
        elif percent_head>=1 and percent_head<=9:
          ph=1
        elif percent_head>=10 and percent_head<=29:
          ph=2
        elif percent_head>=30 and percent_head<=49:
          ph=3
        elif percent_head>=50 and percent_head<=69:
          ph=4
        elif percent_head>=70 and percent_head<=89:
          ph=5
        elif percent_head>=90 and percent_head<=100:
          ph=6
        erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
#        params.append(erythema_head)
        induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
 #       params.append(induration_head)
        desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
  #      params.append(desquamation_head)
      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      col1_arms, col2_arms = st.columns(2)

      with col2_arms:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("arms.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_arms:
        percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
        if percent_arms==0:
          pa=0
        elif percent_arms>=1 and percent_arms<=9:
          pa=1
        elif percent_arms>=10 and percent_arms<=29:
          pa=2
        elif percent_arms>=30 and percent_arms<=49:
          pa=3
        elif percent_arms>=50 and percent_arms<=69:
          pa=4
        elif percent_arms>=70 and percent_arms<=89:
          pa=5
        elif percent_arms>=90 and percent_arms<=100:
          pa=6
        erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
   #     params.append(erythema_arms)
        induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
    #    params.append(induration_arms)
        desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
     #   params.append(desquamation_arms)
        
      col1_trunk, col2_trunk = st.columns(2)
      with col2_trunk:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("trunk.png", width=120) 
      st.text(' ')  
      st.text(' ')  
      st.text(' ')        
      with col1_trunk:
        percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
        if percent_trunk==0:
          pt=0
        elif percent_trunk>=1 and percent_trunk<=9:
          pt=1
        elif percent_trunk>=10 and percent_trunk<=29:
          pt=2
        elif percent_trunk>=30 and percent_trunk<=49:
          pt=3
        elif percent_trunk>=50 and percent_trunk<=69:
          pt=4
        elif percent_trunk>=70 and percent_trunk<=89:
          pt=5
        elif percent_trunk>=90 and percent_trunk<=100:
          pt=6
        erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
      #  params.append(erythema_trunk)
        induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
       # params.append(induration_trunk)
        desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
    #    params.append(desquamation_trunk)
               
      col1_legs, col2_legs = st.columns(2)
      with col2_legs:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("legs.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_legs:
        percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
        if percent_legs==0:
          pl=0
        elif percent_legs>=1 and percent_legs<=9:
          pl=1
        elif percent_legs>=10 and percent_legs<=29:
          pl=2
        elif percent_legs>=30 and percent_legs<=49:
          pl=3
        elif percent_legs>=50 and percent_legs<=69:
          pl=4
        elif percent_legs>=70 and percent_legs<=89:
          pl=5
        elif percent_legs>=90 and percent_legs<=100:
          pl=6
        erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
     #   params.append(erythema_legs)
        induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
      #  params.append(induration_legs)
        desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
       # params.append(desquamation_legs)
      st.text('')
      erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
      params.append(erythema_avg)
      induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
      params.append(induration_avg)   
      desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_guttate, interval_guttate)
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
      PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PASI score and the best treatment options'):
        st.info('Your PASI score is ' +str(PASI))
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
               ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
             
                 
    if prediction_MultiClass==2:
      st.warning('The input skin image involves **inverse psoriasis**. Please specify the degree of your symptoms to get the treatments.')
      treatment={'0':'**Low-potency corticosteroids**', '1':'**Vitamin D3 analogues**', '2':'**Emollients**', '3':'**Topical Tar-Based Products**',
                 '4':'**Topical Antimicrobials/Antiseptics**', '5':'**Cyclosporine**', '6':'**Methotrexate**'}
      params = []
 #     cost = st.slider("How much can you spend?",0, 10, 1)
  #    params.append(cost)      
      lesions = st.slider("Red Shiny Lesions Devoid of Scale in the Inframammary, Perineal, and Axillary Areas ",0, 4, 1)
      params.append(lesions)  
      st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')
      col1_head, col2_head = st.columns(2)
          
      with col2_head:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("head.png", width=120) 
        
      with col1_head:
        percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
        if percent_head==0:
          ph=0
        elif percent_head>=1 and percent_head<=9:
          ph=1
        elif percent_head>=10 and percent_head<=29:
          ph=2
        elif percent_head>=30 and percent_head<=49:
          ph=3
        elif percent_head>=50 and percent_head<=69:
          ph=4
        elif percent_head>=70 and percent_head<=89:
          ph=5
        elif percent_head>=90 and percent_head<=100:
          ph=6
        erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
#        params.append(erythema_head)
        induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
 #       params.append(induration_head)
        desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
  #      params.append(desquamation_head)
      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      col1_arms, col2_arms = st.columns(2)

      with col2_arms:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("arms.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_arms:
        percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
        if percent_arms==0:
          pa=0
        elif percent_arms>=1 and percent_arms<=9:
          pa=1
        elif percent_arms>=10 and percent_arms<=29:
          pa=2
        elif percent_arms>=30 and percent_arms<=49:
          pa=3
        elif percent_arms>=50 and percent_arms<=69:
          pa=4
        elif percent_arms>=70 and percent_arms<=89:
          pa=5
        elif percent_arms>=90 and percent_arms<=100:
          pa=6
        erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
 #       params.append(erythema_arms)
        induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
     #   params.append(induration_arms)
        desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
      #  params.append(desquamation_arms)
        
      col1_trunk, col2_trunk = st.columns(2)
      with col2_trunk:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("trunk.png", width=120) 
      st.text(' ')  
      st.text(' ')  
      st.text(' ')        
      with col1_trunk:
        percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
        if percent_trunk==0:
          pt=0
        elif percent_trunk>=1 and percent_trunk<=9:
          pt=1
        elif percent_trunk>=10 and percent_trunk<=29:
          pt=2
        elif percent_trunk>=30 and percent_trunk<=49:
          pt=3
        elif percent_trunk>=50 and percent_trunk<=69:
          pt=4
        elif percent_trunk>=70 and percent_trunk<=89:
          pt=5
        elif percent_trunk>=90 and percent_trunk<=100:
          pt=6
        erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
   #     params.append(erythema_trunk)
        induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
    #    params.append(induration_trunk)
        desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
     #   params.append(desquamation_trunk)
               
      col1_legs, col2_legs = st.columns(2)
      with col2_legs:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("legs.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_legs:
        percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
        if percent_legs==0:
          pl=0
        elif percent_legs>=1 and percent_legs<=9:
          pl=1
        elif percent_legs>=10 and percent_legs<=29:
          pl=2
        elif percent_legs>=30 and percent_legs<=49:
          pl=3
        elif percent_legs>=50 and percent_legs<=69:
          pl=4
        elif percent_legs>=70 and percent_legs<=89:
          pl=5
        elif percent_legs>=90 and percent_legs<=100:
          pl=6
        erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
   #     params.append(erythema_legs)
        induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
    #    params.append(induration_legs)
        desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
     #   params.append(desquamation_legs)
      st.text('')
      erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
      params.append(erythema_avg)
      induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
      params.append(induration_avg)   
      desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_inverse, interval_inverse)
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
      PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PASI score and the best treatment options'):
        st.info('Your PASI score is ' +str(PASI))
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
               ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
               
    if prediction_MultiClass==3:
      st.warning('The input skin image involves **nail psoriasis**. Please specify the degree of your symptoms to get the treatments.')
      treatment={'0':'**Topical corticosteroids alone or in combination with calcipotriol around the nail fold**', 
                 '1':'**Ustekinumab**', '2':'**Vitamin D3 analogues**', '3':'**Narrow-band UVB UVA**',
                 '4':'**Acitretin**', '5':'**Cyclosporine**', '6':'**Methotrexate**', '7':'**Infliximab**', '8':'**Etanercept**', '9':'**Adalimumab**',
                 '10':'**Certolizumab**', '11':'**Golimumab**', '12':'**Secukinumab**', '13':'**Ixekizumab**', '14':'**Apremilast**', '15':'**Tofacitinib**',
                 '16':'**5-fluorouracil**'}
      params = []
 #     cost = st.slider("How much can you spend?",0, 10, 1)
  #    params.append(cost)      
      pitting = st.slider("Pitting",0, 4, 1)
      params.append(pitting)  
      crumbling = st.slider("Crumbling",0, 4, 1)
      params.append(crumbling)      
      onycholysis = st.slider("Onycholysis (The painless separation of the nail from the nail bed)",0, 4, 1)
      params.append(onycholysis) 
      sh = st.slider("Subungual Hyperkeratosis (This happens when a chalky substance builds up under your nails)",0, 4, 1)
      params.append(sh)
      scaling = st.slider("Scaling Under Nail",0, 4, 1)
      params.append(scaling)      
      oil = st.slider("Oil Drop Yellow-Red Discoloration in the Nail Bed",0, 4, 1)
      params.append(oil) 
      splinter = st.slider("Splinter Hemorrhages (Tiny blood spots that appear underneath the nail)",0, 4, 1)
      params.append(splinter)  
      transverse = st.slider("Transverse Lines and Ridges",0, 4, 1)
      params.append(transverse) 
      leukonychia = st.slider("Leukonychia (White lines or dots that appear on the fingernails or toenails)",0, 4, 1)
      params.append(leukonychia) 
      dyschro = st.slider("Dyschromia (A change in color of the skin or nails)",0, 4, 1)
      params.append(dyschro)    
      lunula = st.slider("Red Lunula",0, 4, 1)
      params.append(lunula)      
    #  erythema = st.slider("Erythema (Redness)",0, 4, 1)        
     # params.append(erythema)
      #induration = st.slider("Induration (Thickness)", 0, 4, 1)
     # params.append(induration)   
      #desquamation = st.slider("Desquamation (Scaling)", 0, 4, 1)
     # params.append(desquamation)
 #     st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')

  #    col1_head, col2_head = st.columns(2)
          
   #   with col2_head:
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
       # st.text(' ')
  #      st.text(' ')
   #     st.image("head.png", width=120) 
        
#      with col1_head:
 #       percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
  #      if percent_head==0:
   #       ph=0
    #    elif percent_head>=1 and percent_head<=9:
     #     ph=1
      #  elif percent_head>=10 and percent_head<=29:
       #   ph=2
  #      elif percent_head>=30 and percent_head<=49:
   #       ph=3
    #    elif percent_head>=50 and percent_head<=69:
     #     ph=4
      #  elif percent_head>=70 and percent_head<=89:
 #         ph=5
  #      elif percent_head>=90 and percent_head<=100:
   #       ph=6
    #    erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
 #       params.append(erythema_head)
  #      induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
  #      params.append(induration_head)
   #     desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
   #     params.append(desquamation_head)
    #  st.text(' ')  
     # st.text(' ')  
      #st.text(' ')  
    #  col1_arms, col2_arms = st.columns(2)

     # with col2_arms:
   #     st.text(' ')
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
       # st.text(' ')
#        st.image("arms.png", width=120) 

 #     st.text(' ')  
  #    st.text(' ')  
   #   st.text(' ')  
    #  with col1_arms:
 #       percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
 #       if percent_arms==0:
  #        pa=0
   #     elif percent_arms>=1 and percent_arms<=9:
    #      pa=1
#        elif percent_arms>=10 and percent_arms<=29:
 #         pa=2
  #      elif percent_arms>=30 and percent_arms<=49:
   #       pa=3
    #    elif percent_arms>=50 and percent_arms<=69:
     #     pa=4
      #  elif percent_arms>=70 and percent_arms<=89:
       #   pa=5
        #elif percent_arms>=90 and percent_arms<=100:
    #      pa=6
     #   erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
    #    params.append(erythema_arms)
    #    induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
     #   params.append(induration_arms)
     #   desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
      #  params.append(desquamation_arms)
        
 #     col1_trunk, col2_trunk = st.columns(2)
  #    with col2_trunk:
   #     st.text(' ')
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
 #       st.text(' ')
  #      st.image("trunk.png", width=120) 
   #   st.text(' ')  
    #  st.text(' ')  
     # st.text(' ')        
  #    with col1_trunk:
   #     percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
    #    if percent_trunk==0:
     #     pt=0
      #  elif percent_trunk>=1 and percent_trunk<=9:
 #         pt=1
  #      elif percent_trunk>=10 and percent_trunk<=29:
   #       pt=2
    #    elif percent_trunk>=30 and percent_trunk<=49:
     #     pt=3
      #  elif percent_trunk>=50 and percent_trunk<=69:
#          pt=4
 #       elif percent_trunk>=70 and percent_trunk<=89:
  #        pt=5
   #     elif percent_trunk>=90 and percent_trunk<=100:
    #      pt=6
     #   erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
  #      params.append(erythema_trunk)
   #     induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
   #     params.append(induration_trunk)
    #    desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
    #    params.append(desquamation_trunk)
               
#      col1_legs, col2_legs = st.columns(2)
 #     with col2_legs:
  #      st.text(' ')
   #     st.text(' ')
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
       # st.image("legs.png", width=120) 

  #    st.text(' ')  

   #   with col1_legs:
    #    percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
     #   if percent_legs==0:
      #    pl=0
  #      elif percent_legs>=1 and percent_legs<=9:
   #       pl=1
    #    elif percent_legs>=10 and percent_legs<=29:
     #     pl=2
      #  elif percent_legs>=30 and percent_legs<=49:
       #   pl=3
        #elif percent_legs>=50 and percent_legs<=69:
  #        pl=4
   #     elif percent_legs>=70 and percent_legs<=89:
    #      pl=5
     #   elif percent_legs>=90 and percent_legs<=100:
#          pl=6
 #       erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
   #     params.append(erythema_legs)
  #      induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
    #    params.append(induration_legs)
   #     desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
     #   params.append(desquamation_legs)

 #     erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
  #    params.append(erythema_avg)
   #   induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
    #  params.append(induration_avg)   
     # desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      #params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_nail, interval_nail)
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
    #  PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PASI score and the best treatment options'):
  #      st.info('Your PASI score is ' +str(PASI))
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
               ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
        
         
    if prediction_MultiClass==4:
      st.warning('The input skin image involves **plaque psoriasis**. Please specify the degree of your symptoms to get the treatments.')
      treatment={'0':'**Topical tacrolimus (0.1%) ointment combined with 6% salicylic acid gel**', 
                 '1':'**Corticosteroids**', '2':'**Calcipotriol**', '3':'**Calcitriol**',
                 '4':'**Tacalcitol**', '5':'**Retinoids**', '6':'**Excimer Laser Therapy**', '7':'**Keratolytic and Tar-Based Shampoos**', '8':'**Infliximab**', '9':'**Adalimumab**',
                 '10':'**Etanercept**', '11':'**Ustekinumab**', '12':'**Ixekizumab**', '13':'**Secukinumab**', '14':'**Brodalumab**', '15':'**Guselkumab**',
                 '16':'**Tonsillectomy**'}
      params = []
 #     cost = st.slider("How much can you spend?",0, 10, 1)
  #    params.append(cost)  
      lesion = st.slider("Annular Lesions Comprising an Erythematous Base and Thick Silvery Scale",0, 4, 1)
      params.append(lesion)  
      plaques = st.slider("Sharply Circumscribed, Round-Oval, or Nummular (Coin-Sized) Plaques",0, 4, 1)
      params.append(plaques)  
      itching = st.slider("Itching",0, 4, 1)
      params.append(itching)      
      dryness = st.slider("Dryness",0, 4, 1)
      params.append(dryness) 
      tenderness = st.slider("Tenderness",0, 4, 1)
      params.append(tenderness)      
 #     erythema = st.slider("Erythema",0, 10, 1)
  #    params.append(erythema)
      st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')
      col1_head, col2_head = st.columns(2)
          
      with col2_head:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("head.png", width=120) 
        
      with col1_head:
        percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
        if percent_head==0:
          ph=0
        elif percent_head>=1 and percent_head<=9:
          ph=1
        elif percent_head>=10 and percent_head<=29:
          ph=2
        elif percent_head>=30 and percent_head<=49:
          ph=3
        elif percent_head>=50 and percent_head<=69:
          ph=4
        elif percent_head>=70 and percent_head<=89:
          ph=5
        elif percent_head>=90 and percent_head<=100:
          ph=6
        erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
#        params.append(erythema_head)
        induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
 #       params.append(induration_head)
        desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
  #      params.append(desquamation_head)
      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      col1_arms, col2_arms = st.columns(2)

      with col2_arms:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("arms.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_arms:
        percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
        if percent_arms==0:
          pa=0
        elif percent_arms>=1 and percent_arms<=9:
          pa=1
        elif percent_arms>=10 and percent_arms<=29:
          pa=2
        elif percent_arms>=30 and percent_arms<=49:
          pa=3
        elif percent_arms>=50 and percent_arms<=69:
          pa=4
        elif percent_arms>=70 and percent_arms<=89:
          pa=5
        elif percent_arms>=90 and percent_arms<=100:
          pa=6
        erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
   #     params.append(erythema_arms)
        induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
    #    params.append(induration_arms)
        desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
     #   params.append(desquamation_arms)
        
      col1_trunk, col2_trunk = st.columns(2)
      with col2_trunk:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("trunk.png", width=120) 
      st.text(' ')  
      st.text(' ')  
      st.text(' ')        
      with col1_trunk:
        percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
        if percent_trunk==0:
          pt=0
        elif percent_trunk>=1 and percent_trunk<=9:
          pt=1
        elif percent_trunk>=10 and percent_trunk<=29:
          pt=2
        elif percent_trunk>=30 and percent_trunk<=49:
          pt=3
        elif percent_trunk>=50 and percent_trunk<=69:
          pt=4
        elif percent_trunk>=70 and percent_trunk<=89:
          pt=5
        elif percent_trunk>=90 and percent_trunk<=100:
          pt=6
        erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
   #     params.append(erythema_trunk)
        induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
    #    params.append(induration_trunk)
        desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
     #   params.append(desquamation_trunk)
               
      col1_legs, col2_legs = st.columns(2)
      with col2_legs:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("legs.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_legs:
        percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
        if percent_legs==0:
          pl=0
        elif percent_legs>=1 and percent_legs<=9:
          pl=1
        elif percent_legs>=10 and percent_legs<=29:
          pl=2
        elif percent_legs>=30 and percent_legs<=49:
          pl=3
        elif percent_legs>=50 and percent_legs<=69:
          pl=4
        elif percent_legs>=70 and percent_legs<=89:
          pl=5
        elif percent_legs>=90 and percent_legs<=100:
          pl=6
        erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
#        params.append(erythema_legs)
        induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
 #       params.append(induration_legs)
        desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
  #      params.append(desquamation_legs)
      st.text('')
      erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
      params.append(erythema_avg)
      induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
      params.append(induration_avg)   
      desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_plaque, interval_plaque)
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
      PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PASI score and the best treatment options'):
        st.info('Your PASI score is ' +str(PASI))
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
               ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
        
    
    if prediction_MultiClass==5:
      st.warning('The input skin image involves **psoriatic arthritis**. Please specify the degree of your symptoms to get the treatments.')
      treatment={'0':'**Secukinumab**', 
                 '1':'**Ibuprofen**', '2':'**Naproxen**', '3':'**Diclofenac**',
                 '4':'**Celecoxib**', '5':'**Etoricoxib**', '6':'**Corticosteroids**', '7':'**Leflunomide**', '8':'**Sulfasalazine**', '9':'**Methotrexate**',
                 '10':'**Adalimumab**', '11':'**Apremilast**', '12':'**Certolizumab**', '13':'**Etanercept**', '14':'**Tofacitinib**'}
      params = []
  #    cost = st.slider("How much can you spend?",0, 10, 1)
   #   params.append(cost)      
      nail = st.slider("Nail Lesions",0, 4, 1)
      params.append(nail)  
      Dactylitis = st.slider("Dactylitis (A global swelling of a finger or a toe giving it a clinical sausage-shape presentation)",0, 4, 1)
      params.append(Dactylitis)      
      Elbow = st.slider("Elbow Pain",0, 4, 1)
      params.append(Elbow) 
      Fatigue = st.slider("Fatigue",0, 4, 1)
      params.append(Fatigue)         
      Trouble = st.slider("Trouble Moving Hands and Fingers",0, 4, 1)        
      params.append(Trouble)   
      pain = st.slider("Pain and Swelling Over Tendons",0, 4, 1)        
      params.append(pain)  
      Stiffness = st.slider("Stiffness, Pain, Throbbing, Swelling and Tenderness in One or More Joints",0, 4, 1)
      params.append(Stiffness)  
      Reduced = st.slider("Reduced Range of Motion",0, 4, 1)
      params.append(Reduced)      
      Morning = st.slider("Morning Stiffness and Tiredness",0, 4, 1)
      params.append(Morning) 
      bed = st.slider("Nail Changes Such As Pitting or Separation from the Nail Bed",0, 4, 1)
      params.append(bed)         
      Redness = st.slider("Redness and pain of the eye (Uveitis)",0, 4, 1)        
      params.append(Redness) 
#      erythema = st.slider("Erythema (Redness)",0, 4, 1)        
 #     params.append(erythema)
  #    induration = st.slider("Induration (Thickness)", 0, 4, 1)
   #   params.append(induration)   
    #  desquamation = st.slider("Desquamation (Scaling)", 0, 4, 1)
     # params.append(desquamation)
 #     st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')

  #    col1_head, col2_head = st.columns(2)
          
   #   with col2_head:
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
       # st.text(' ')
  #      st.text(' ')
   #     st.image("head.png", width=120) 
        
#      with col1_head:
 #       percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
  #      if percent_head==0:
   #       ph=0
    #    elif percent_head>=1 and percent_head<=9:
     #     ph=1
      #  elif percent_head>=10 and percent_head<=29:
       #   ph=2
  #      elif percent_head>=30 and percent_head<=49:
   #       ph=3
    #    elif percent_head>=50 and percent_head<=69:
     #     ph=4
      #  elif percent_head>=70 and percent_head<=89:
 #         ph=5
  #      elif percent_head>=90 and percent_head<=100:
   #       ph=6
    #    erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
 #       params.append(erythema_head)
  #      induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
  #      params.append(induration_head)
   #     desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
   #     params.append(desquamation_head)
    #  st.text(' ')  
     # st.text(' ')  
      #st.text(' ')  
    #  col1_arms, col2_arms = st.columns(2)

     # with col2_arms:
   #     st.text(' ')
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
       # st.text(' ')
#        st.image("arms.png", width=120) 

 #     st.text(' ')  
  #    st.text(' ')  
   #   st.text(' ')  
    #  with col1_arms:
 #       percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
 #       if percent_arms==0:
  #        pa=0
   #     elif percent_arms>=1 and percent_arms<=9:
    #      pa=1
#        elif percent_arms>=10 and percent_arms<=29:
 #         pa=2
  #      elif percent_arms>=30 and percent_arms<=49:
   #       pa=3
    #    elif percent_arms>=50 and percent_arms<=69:
     #     pa=4
      #  elif percent_arms>=70 and percent_arms<=89:
       #   pa=5
        #elif percent_arms>=90 and percent_arms<=100:
    #      pa=6
     #   erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
    #    params.append(erythema_arms)
    #    induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
     #   params.append(induration_arms)
     #   desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
      #  params.append(desquamation_arms)
        
 #     col1_trunk, col2_trunk = st.columns(2)
  #    with col2_trunk:
   #     st.text(' ')
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
 #       st.text(' ')
  #      st.image("trunk.png", width=120) 
   #   st.text(' ')  
    #  st.text(' ')  
     # st.text(' ')        
  #    with col1_trunk:
   #     percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
    #    if percent_trunk==0:
     #     pt=0
      #  elif percent_trunk>=1 and percent_trunk<=9:
 #         pt=1
  #      elif percent_trunk>=10 and percent_trunk<=29:
   #       pt=2
    #    elif percent_trunk>=30 and percent_trunk<=49:
     #     pt=3
      #  elif percent_trunk>=50 and percent_trunk<=69:
#          pt=4
 #       elif percent_trunk>=70 and percent_trunk<=89:
  #        pt=5
   #     elif percent_trunk>=90 and percent_trunk<=100:
    #      pt=6
     #   erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
  #      params.append(erythema_trunk)
   #     induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
   #     params.append(induration_trunk)
    #    desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
    #    params.append(desquamation_trunk)
               
#      col1_legs, col2_legs = st.columns(2)
 #     with col2_legs:
  #      st.text(' ')
   #     st.text(' ')
    #    st.text(' ')
     #   st.text(' ')
      #  st.text(' ')
       # st.image("legs.png", width=120) 

  #    st.text(' ')  

   #   with col1_legs:
    #    percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
     #   if percent_legs==0:
      #    pl=0
  #      elif percent_legs>=1 and percent_legs<=9:
   #       pl=1
    #    elif percent_legs>=10 and percent_legs<=29:
     #     pl=2
      #  elif percent_legs>=30 and percent_legs<=49:
       #   pl=3
        #elif percent_legs>=50 and percent_legs<=69:
  #        pl=4
   #     elif percent_legs>=70 and percent_legs<=89:
    #      pl=5
     #   elif percent_legs>=90 and percent_legs<=100:
#          pl=6
 #       erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
   #     params.append(erythema_legs)
  #      induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
    #    params.append(induration_legs)
   #     desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
     #   params.append(desquamation_legs)

 #     erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
  #    params.append(erythema_avg)
   #   induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
    #  params.append(induration_avg)   
     # desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      #params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_arthritis, interval_arthritis)
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
      st.info("The following questions are according to Psoriasis Epidemiology Screening Tool (PEST). Please check the box, if you have any of the below conditions.")
      pest=[]
      pest_joint=st.radio("Have you ever had a swollen joint (or joints)?",('Yes','No'))
      if pest_joint=='Yes':
       pj=1
      else:
       pj=0
      pest.append(pj)
      pest_doctor=st.radio("Has a doctor ever told you that you have arthritis?",('Yes','No'))
      if pest_doctor=='Yes':
       pd=1
      else:
       pd=0
      pest.append(pd)
      pest_finger=st.radio("Do your finger nails or toenails have holes or pits?",('Yes','No'))
      if pest_finger=='Yes':
       pf=1
      else:
       pf=0
      pest.append(pf)
      pest_pain=st.radio("Have you had pain in your heel?",('Yes','No'))
      if pest_pain=='Yes':
       pp=1
      else:
       pp=0
      pest.append(pp)
      pest_reason=st.radio("Have you had a finger or toe that was completely swollen and painful for no apparent reason?",('Yes','No'))
      if pest_reason=='Yes':
       pr=1
      else:
       pr=0
      pest.append(pr)
      pest_sum=sum(pest)
#      PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PEST score and the best treatment options'):
#        st.info('Your PASI score is ' +str(PASI))
        if pest_sum>=3:
          st.info('Your PEST score is equal to ' + str(pest_sum) +'. You most probably have Psoriatic Arthritis (PsA) according to the PEST score, which confirms the result of analyzing your skin image by our system.')
        else:
          st.info('Your PEST score is equal to ' + str(pest_sum) +'. It is possible that you do not have Psoriatic Arthritis (PsA) according to the PEST score. However, the results of analyzing skin image indicate you have PsA. Therefore, you have to be cautious about your condition.')
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
                 ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')

    if prediction_MultiClass==6:
      st.warning('The input skin image involves **pustular psoriasis**. Please specify the degree of your symptoms to get the treatments.')
      treatment={'0':'**Acitretin**', 
                 '1':'**Cyclosporine**', '2':'**Methotrexate**', '3':'**Infliximab**',
                 '4':'**Adalimumab**', '5':'**Etanercept**', '6':'**Psoralen and Ultraviolet A (PUVA)**', '7':'**Basiliximab**', '8':'**Tocilizumab**', '9':'**Gueselkumab**',
                 '10':'**Risankizumab**', '11':'**Vitamin D derivatives**'}
      params = []
 #     cost = st.slider("How much can you spend?",0, 10, 1)
  #    params.append(cost)      
      toxicity = st.slider("Toxicity",0, 4, 1)
      params.append(toxicity)  
      fever = st.slider("Fever",0, 4, 1)
      params.append(fever)      
      leucocytosis = st.slider("Leucocytosis (High White Blood Cell Count)",0, 4, 1)
      params.append(leucocytosis) 
      pustules = st.slider("Multiple Uniform Sterile Pustules on the Body",0, 4, 1)
      params.append(pustules)         
      pain = st.slider("Pain",0, 4, 1)        
      params.append(pain)   
      arthralgia = st.slider("Arthralgia",0, 4, 1)        
      params.append(arthralgia)  
      fatigue = st.slider("Fatigue",0, 4, 1)
      params.append(fatigue)  
      st.info('In this part, a survey is performed to claculate your Psoriasis Area and Severity Index (PASI) score.')
      col1_head, col2_head = st.columns(2)
          
      with col2_head:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("head.png", width=120) 
        
      with col1_head:
        percent_head=st.slider("What percentage of your head is involved?",0, 100, 1)
        if percent_head==0:
          ph=0
        elif percent_head>=1 and percent_head<=9:
          ph=1
        elif percent_head>=10 and percent_head<=29:
          ph=2
        elif percent_head>=30 and percent_head<=49:
          ph=3
        elif percent_head>=50 and percent_head<=69:
          ph=4
        elif percent_head>=70 and percent_head<=89:
          ph=5
        elif percent_head>=90 and percent_head<=100:
          ph=6
        erythema_head=st.slider("Erythema (Redness) of the head lesion",0, 4, 1)
 #       params.append(erythema_head)
        induration_head=st.slider("Induration (Thickness) of the head lesion",0, 4, 1)
  #      params.append(induration_head)
        desquamation_head=st.slider("Desquamation (Scaling) of the head lesion",0, 4, 1)
   #     params.append(desquamation_head)
      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      col1_arms, col2_arms = st.columns(2)

      with col2_arms:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("arms.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_arms:
        percent_arms=st.slider("What percentage of your arms is involved?",0, 100, 1)
        if percent_arms==0:
          pa=0
        elif percent_arms>=1 and percent_arms<=9:
          pa=1
        elif percent_arms>=10 and percent_arms<=29:
          pa=2
        elif percent_arms>=30 and percent_arms<=49:
          pa=3
        elif percent_arms>=50 and percent_arms<=69:
          pa=4
        elif percent_arms>=70 and percent_arms<=89:
          pa=5
        elif percent_arms>=90 and percent_arms<=100:
          pa=6
        erythema_arms=st.slider("Erythema (Redness) of the arms lesion",0, 4, 1)
  #      params.append(erythema_arms)
        induration_arms=st.slider("Induration (Thickness) of the arms lesion",0, 4, 1)
   #     params.append(induration_arms)
        desquamation_arms=st.slider("Desquamation (Scaling) of the arms lesion",0, 4, 1)
    #    params.append(desquamation_arms)
        
      col1_trunk, col2_trunk = st.columns(2)
      with col2_trunk:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("trunk.png", width=120) 
      st.text(' ')  
      st.text(' ')  
      st.text(' ')        
      with col1_trunk:
        percent_trunk=st.slider("What percentage of your trunk is involved?",0, 100, 1)
        if percent_trunk==0:
          pt=0
        elif percent_trunk>=1 and percent_trunk<=9:
          pt=1
        elif percent_trunk>=10 and percent_trunk<=29:
          pt=2
        elif percent_trunk>=30 and percent_trunk<=49:
          pt=3
        elif percent_trunk>=50 and percent_trunk<=69:
          pt=4
        elif percent_trunk>=70 and percent_trunk<=89:
          pt=5
        elif percent_trunk>=90 and percent_trunk<=100:
          pt=6
        erythema_trunk=st.slider("Erythema (Redness) of the trunk lesion",0, 4, 1)
 #       params.append(erythema_trunk)
        induration_trunk=st.slider("Induration (Thickness) of the trunk lesion",0, 4, 1)
  #      params.append(induration_trunk)
        desquamation_trunk=st.slider("Desquamation (Scaling) of the trunk lesion",0, 4, 1)
   #     params.append(desquamation_trunk)
               
      col1_legs, col2_legs = st.columns(2)
      with col2_legs:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.image("legs.png", width=120) 

      st.text(' ')  
      st.text(' ')  
      st.text(' ')  
      with col1_legs:
        percent_legs=st.slider("What percentage of your legs is involved?",0, 100, 1)
        if percent_legs==0:
          pl=0
        elif percent_legs>=1 and percent_legs<=9:
          pl=1
        elif percent_legs>=10 and percent_legs<=29:
          pl=2
        elif percent_legs>=30 and percent_legs<=49:
          pl=3
        elif percent_legs>=50 and percent_legs<=69:
          pl=4
        elif percent_legs>=70 and percent_legs<=89:
          pl=5
        elif percent_legs>=90 and percent_legs<=100:
          pl=6
        erythema_legs=st.slider("Erythema (Redness) of the legs lesion",0, 4, 1)
  #      params.append(erythema_legs)
        induration_legs=st.slider("Induration (Thickness) of the legs lesion",0, 4, 1)
   #     params.append(induration_legs)
        desquamation_legs=st.slider("Desquamation (Scaling) of the legs lesion",0, 4, 1)
    #    params.append(desquamation_legs)
      st.text('')
      erythema_avg=(erythema_head+erythema_arms+erythema_trunk+erythema_legs)/4
      params.append(erythema_avg)
      induration_avg=(induration_head+induration_arms+induration_trunk+induration_legs)/4
      params.append(induration_avg)   
      desquamation_avg=(desquamation_head+desquamation_arms+desquamation_trunk+desquamation_legs)/4
      params.append(desquamation_avg)
      
      params1=[element/sum(params) for element in params]
      omega=MCDM(params1, X_pustular, interval_pustular)
      max_index1 = np.argmax(omega)
      omega[max_index1]=float('-inf')
      max_index2 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index3 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index4 = np.argmax(omega)
      omega[np.argmax(omega)]=float('-inf')
      max_index5 = np.argmax(omega)
      PASI=0.1*(erythema_head+induration_head+desquamation_head)*ph+0.2*(erythema_arms+induration_arms+desquamation_arms)*pa+0.3*(erythema_trunk+induration_trunk+desquamation_trunk)*pt+0.4*(erythema_legs+induration_legs+desquamation_legs)*pl
      if st.button('Get your PASI score and the best treatment options'):
        st.info('Your PASI score is ' +str(PASI))
        st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
               ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
