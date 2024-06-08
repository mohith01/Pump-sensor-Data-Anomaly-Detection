import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.tree import DecisionTreeClassifier


with open(r'dict_for_pickling.pkl', 'rb') as f:
    dict_for_unpickling = pickle.load(f)

st.write("# Pump Sensor Anomaly Detection")


import streamlit as st

name_dict = dict_for_unpickling['d1']

for k, v in name_dict.items():
    name_dict[k] = st.number_input(k, min_value= v[0] , max_value=v[1])
    st.write(name_dict[k])

if st.button('Submit'):
  sc = dict_for_unpickling['sc']
  xgb = dict_for_unpickling['reg']
  isf_tree = dict_for_unpickling['isf_tree']
  output = sc.transform([list(name_dict.values())])
  xgb_output = xgb.predict(output)[0]
  isf_tree_output = isf_tree.predict(output)[0]
  if isf_tree_output==1:
    isf_tree_output = 0
  else:
    isf_tree_output = 1

  df = pd.DataFrame(
    [
        {"Classifier": "XG Boost", "output": bool(int(xgb_output))},
        {"Classifier": "Isolation Trees",  "output": bool(int(isf_tree_output))},
    ]
  )
  st.write(df)









