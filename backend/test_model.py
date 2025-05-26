import pickle

model = pickle.load(open("churn_model.pkl", "rb"))
sample = [[0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 70.70, 151.65]]
result = model.predict(sample)
print("Prediction:", result[0])
