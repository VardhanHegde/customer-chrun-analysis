import pickle
filename = 'artifacts\model_trainer\model.pkl'
temp = pickle.load(open(filename, 'rb'))
print(temp)

filename = 'artifacts\data_transformation\preprocessor.pkl'
temp = pickle.load(open(filename, 'rb'))
print(temp)

filename = 'model.pkl'
temp = pickle.load(open(filename, 'rb'))
print(temp)
