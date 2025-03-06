import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
scal=StandardScaler()
clfr = joblib.load("heartmodel.pkl")
 
def preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal ):   
 
    if sex=="male":
        sex=1 
    else: sex=0 
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=3
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
    if slope=="Upsloping: better heart rate with excercise(uncommon)":
        slope=0
    elif slope=="Flatsloping: minimal change(typical healthy heart)":
          slope=1
    elif slope=="Downsloping: signs of unhealthy heart":
        slope=2  
    if thal=="fixed defect: used to be defect but ok now":
        thal=6
    elif thal=="reversable defect: no proper blood movement when excercising":
        thal=7
    elif thal=="normal":
        thal=2.31
    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2
    user_input=[age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal]
    df=pd.read_csv('heart.csv')
    y = df["target"]
    X = df.drop('target',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    user_input=np.array(user_input)
    user_input=user_input.reshape(1,-1)
    user_input=scaler.transform(user_input)
    print(user_input)
    prediction = clfr.predict(user_input)
    print(int(prediction))

    return int(prediction)

# # if __name__ == '__main__':
# #     t=preprocess(39,"male","Non-anginal pain",130,"ST-T Wave abnormality",250,"Yes",187,"Yes",2,"Downsloping: signs of unhealthy heart",2,"normal")
# #     print(int(t))
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler

# # Load the pre-trained model
# try:
#     clfr = joblib.load("heartmodel.pkl")
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# # Load dataset for feature scaling
# df = pd.read_csv('heart.csv')

# # Extract features and target
# y = df["target"]
# X = df.drop(columns=['target'])

# # Fit the scaler on full dataset (NOT re-fitting in function)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# def preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):   
#     # Encoding categorical inputs
#     sex = 1 if sex == "male" else 0
#     cp_mapping = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
#     cp = cp_mapping.get(cp, 0)

#     exang = 1 if exang == "Yes" else 0
#     fbs = 1 if fbs == "Yes" else 0

#     slope_mapping = {
#         "Upsloping: better heart rate with exercise(uncommon)": 0,
#         "Flatsloping: minimal change(typical healthy heart)": 1,
#         "Downsloping: signs of unhealthy heart": 2
#     }
#     slope = slope_mapping.get(slope, 1)

#     thal_mapping = {
#         "fixed defect: used to be defect but ok now": 6,
#         "reversable defect: no proper blood movement when exercising": 7,
#         "normal": 2.31
#     }
#     thal = thal_mapping.get(thal, 2.31)

#     restecg_mapping = {
#         "Nothing to note": 0,
#         "ST-T Wave abnormality": 1,
#         "Possible or definite left ventricular hypertrophy": 2
#     }
#     restecg = restecg_mapping.get(restecg, 0)

#     # Create input array
#     user_input = np.array([[age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal]])

#     # Standardize input using pre-fitted scaler
#     user_input_scaled = scaler.transform(user_input)

#     # Predict
#     prediction = clfr.predict(user_input_scaled)[0]
#     print(f"Prediction: {prediction}")
    
#     return int(prediction)

# Example usage (Uncomment below line to test)
# t = preprocess(39, "male", "Non-anginal pain", 130, "ST-T Wave abnormality", 250, "Yes", 187, "Yes", 2, "Downsloping: signs of unhealthy heart", 2, "normal")
# print(f"Predicted class: {t}")
