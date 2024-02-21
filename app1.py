# Importing the Necessary Libraries for Loading the Models 

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import requests
import tensorflow.compat.v1.keras.backend as K
import pandas as pd
from streamlit_option_menu import option_menu

st.set_page_config(layout="centered")

@st.cache(allow_output_mutation=True)
#@st.cache_data(allow_output_mutation=True)
# The Function of Loading the Hybrid CNN Model for Feature Extraction

def load_classifiers():
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

    weights_binary = np.load('weights_binary.npy')
    weights_MultiClass = np.load('weights_MultiClass.npy')

    session = K.get_session()

    return network1, network2, network3, network4, network5, network6, weights_binary, weights_MultiClass, session

#def load_classifiers():
 # url1 = 'https://drive.google.com/u/0/uc?id=19yQXM-v_Q0h9sQGdqmGgiDjkqXAj7pQG&export=download'
  #r1 = requests.get(url1, allow_redirects=True)
  #url2 = 'https://drive.google.com/u/0/uc?id=1-2oglfOHsJ8BsJ98q5b8xLBiytGM0ez3&export=download'
  #r2 = requests.get(url2, allow_redirects=True)
 # url3 = 'https://drive.google.com/u/0/uc?id=1-5r031ZBkGVitnuV4Xu8z0Hx2ZMwqWYT&export=download&confirm=t'
 # r3 = requests.get(url3, allow_redirects=True)
 # url4 = 'https://drive.google.com/u/0/uc?id=1-3SVUDHrKmV-dUz1uppdLrIoV_favOZt&export=download&confirm=t'
 # r4 = requests.get(url4, allow_redirects=True)
  #url5 = 'https://drive.google.com/u/0/uc?id=1-64EcLv277pWm4pfrlAeXLkygT2JIvKj&export=download&confirm=t'
 # r5 = requests.get(url5, allow_redirects=True)
 # url6 = 'https://drive.google.com/u/0/uc?id=1-4E8kBAFuoToexf_Nh1-i3MxQeQ7wC4D&export=download&confirm=t'
 # r6 = requests.get(url6, allow_redirects=True)
 # url7 = 'https://drive.google.com/u/0/uc?id=1C2X4worq9sGrBlURKLzqucDtDYQCvBA-&export=download'
 # r7 = requests.get(url7, allow_redirects=True)
 # url8 = 'https://drive.google.com/u/0/uc?id=1-5zYystBU_xRA4iihAcRUqsU3EC5QUnm&export=download'
 # r8 = requests.get(url8, allow_redirects=True)
 # with open('ResNet50V2_Binary.hdf5', 'wb') as f:
 #   f.write(r1.content)
 # with open('ResNet50V2_MultiClass.hdf5', 'wb') as f:
  #  f.write(r2.content)
#  with open('ResNet101V2_Binary.hdf5', 'wb') as f:
 #   f.write(r3.content)
 # with open('ResNet101V2_MultiClass.hdf5', 'wb') as f:
 #   f.write(r4.content)
 # with open('ResNet152V2_Binary.hdf5', 'wb') as f:
 #   f.write(r5.content)
 # with open('ResNet152V2_MultiClass.hdf5', 'wb') as f:
  #  f.write(r6.content)
 # with open('weights_binary.npy', 'wb') as f:
  #  f.write(r7.content)
 # with open('weights_MultiClass.npy', 'wb') as f:
 #   f.write(r8.content)
 # network1=tf.keras.models.load_model('ResNet50V2_Binary.hdf5')
 # network1.make_predict_function()
 # network2=tf.keras.models.load_model('ResNet101V2_Binary.hdf5')
 # network2.make_predict_function()
 # network3=tf.keras.models.load_model('ResNet152V2_Binary.hdf5')
 # network3.make_predict_function()
 # network4=tf.keras.models.load_model('ResNet50V2_MultiClass.hdf5')
  #network4.make_predict_function()
  #network5=tf.keras.models.load_model('ResNet101V2_MultiClass.hdf5')
 # network5.make_predict_function()
 # network6=tf.keras.models.load_model('ResNet152V2_MultiClass.hdf5')
 # network6.make_predict_function()
 # weights_binary=np.load('weights_binary.npy')
 # weights_MultiClass=np.load('weights_MultiClass.npy')
 # session = K.get_session()
 # return network1, network2, network3, network4, network5, network6, weights_binary, weights_MultiClass, session


# Importing the Necessary Libraries for Loading the Input Image and Making Predicitons
import cv2
from PIL import Image, ImageOps
import numpy as np
# st.set_option('deprecation.showfileUploaderEncoding', False)

# The Function of Importing the Image and Predicting Its Corresponding Class 
def import_and_predict_binary(image_data, classifier1, classifier2, classifier3, weights_binary):

        # Preparing the Image
        size = (224,224)   
        image = image_data.resize(size, Image.Resampling.LANCZOS)
  #      image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
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
        image = image_data.resize(size, Image.Resampling.LANCZOS)
  #      image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
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
          f[i][j]=1-(interval[j][0]-X[i][j])/((max(interval[j][0]-x_min[j],x_max[j]-interval[j][1])))
    
        elif X[i][j]>=interval[j][1] and X[i][j]<=x_max[j]:
          f[i][j]=1-(1-interval[j][1]+X[i][j])/((max(interval[j][0]-x_min[j],x_max[j]-interval[j][1])))
    
      if j==0:   #cost criterion
        if X[i][j]>=interval[j][0] and X[i][j]<=interval[j][1]:
          f[i][j]=1/((max(interval[j][0]-x_min[j],x_max[j]-interval[j][1])))

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

with st.sidebar:
#  st.info('For technical details, please refer to: ... .doi: ...')
  st.info('For technical details, please refer to: https://doi.org/10.1016/j.cie.2023.109754')
  st.header('Disclaimer:')
  st.warning('This app works as a decision support tool for clinicians. Although this program has been tested thoroughly, the accuracy of the information cannot be guaranteed and the authors shall not be liable for any claim, damages or other liability.')
  
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
    import pandas as pd
    X = pd.read_excel('Matrix1.xlsx')
    X=np.array(X, dtype=np.float64)
    treatment=''

    interval=[[0, 10],[0, 10],[0, 10],[0, 10], [0, 10],[0, 10],[0, 10],[0, 10],[0, 10], [0, 10],[0, 10],[0, 10],[0, 10],[0, 10], [0, 10], [0,10]]

    if prediction_MultiClass==0:
      st.warning('The input skin image involves **Erythrodermic Psoriasis**.')
    if prediction_MultiClass==1:
     st.warning('The input skin image involves **Guttate Psoriasis**.')
    if prediction_MultiClass==2:
     st.warning('The input skin image involves **Inverse Psoriasis**.')
    if prediction_MultiClass==3:
     st.warning('The input skin image involves **Nail Psoriasis**.')
    if prediction_MultiClass==4:
     st.warning('The input skin image involves **Plaque Psoriasis**.')
    if prediction_MultiClass==5:
     st.warning('The input skin image involves **Psoriatic Arthritis**.')
    if prediction_MultiClass==6:
     st.warning('The input skin image involves **Pustular Psoriasis**.')
    st.warning('Please specify the degree of symptoms on a 0-9 scale to get the best treatment options. A higher amount indicates a higher severity level')
    params = []
    params.append(5)
    erythema = st.slider("Erythema",0, 9, 1)
    params.append(erythema)      
    induration = st.slider("Induration",0, 9, 1)
    params.append(induration)
    desquamation= st.slider("Desquamation",0, 9, 1)
    params.append(desquamation)
    itching= st.slider("Itching",0, 9, 1)
    params.append(itching)
    dryness= st.slider("Dryness",0, 9, 1)
    params.append(dryness) 
    tenderness= st.slider("Tenderness",0, 9, 1)
    params.append(tenderness)
    uveitis= st.slider("Redness of the Eye (Uveitis)",0, 9, 1)
    params.append(uveitis) 
    decline= st.slider("Decline in Vision",0, 9, 1)
    params.append(decline)
    fever= st.slider("Fever",0, 9, 1)
    params.append(fever) 
    pain= st.slider("Pain",0, 9, 1)
    params.append(pain) 
    pitting= st.slider("Pitting",0, 9, 1)
    params.append(pitting) 
    crumbling= st.slider("Crumbling",0, 9, 1)
    params.append(crumbling)
    swelling= st.slider("Swelling",0, 9, 1)
    params.append(swelling)
    tiredness= st.slider("Tiredness",0, 9, 1)
    params.append(tiredness) 
    params.append(5)
    params1=[element/sum(params) for element in params]
    omega=MCDM(params1, X, interval)
    treatment={'0':'**Tacrolimus**', '1':'**Corticosteroids**', '2':'**Calcipotriol**', 
               '3':'**Retinoids**', 
               '4':'**Excimer Laser Therapy**', '5':'**Infliximab**', '6':'**Adalimumab**', '7':'**Etanercept**',
               '8':'**Phototherapy**', '9':'**Ustekinumab**', '10':'**Ixekizumab**', '11':'**Secukinumab**',
               '12':'**Brodalumab**', '13':'**Guselkumab**', '14':'**Tonsillectomy**', '15':'**Vitamin D analogs**', '16':'**Tar**', '17':'**Cyclosprine**',
               '18':'**Methotrexate**', '19':'**Acitretin**', '20':'**Emollients**',
               '21':'**Narrow-band Ultraviolet B **', '22':'**Psoralen and Ultraviolet A (PUVA)**', '23':'**Basiliximab**', 
               '24':'**Tocilizumab**','25':'**Risankizumab**', '26':'**Certolizumab**', '27':'**Golimumab**', '28':'**Apremilast**',
               '29':'**Tofacitinib**', '30':'**5-Fluorouracil**', '31':'**Ibuprofen**', '32':'**Naproxen**',
               '33':'**Diclofenac**', '34':'**Celecoxib**', '35':'**Etoricoxib**', '36':'**Leflunomide**',
               '37':'**Sulfasalazine**'}
      
    max_index1 = np.argmax(omega)
    omega[max_index1]=float('-inf')
    max_index2 = np.argmax(omega)
    omega[np.argmax(omega)]=float('-inf')
    max_index3 = np.argmax(omega)
    omega[np.argmax(omega)]=float('-inf')
    max_index4 = np.argmax(omega)
    omega[np.argmax(omega)]=float('-inf')
    max_index5 = np.argmax(omega)
      
    if st.button('Get the best treatment options'):
      st.info('The best treatment options according to your conditions are ' + treatment[str(max_index1)] + ', ' + treatment[str(max_index2)] + ', ' + treatment[str(max_index3)]+
             ', ' + treatment[str(max_index4)] +' and '+treatment[str(max_index5)] + '.')
