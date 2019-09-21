import numpy as np
from sklearn.preprocessing import OneHotEncoder


enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)


print("categries_" + str(enc.categories_))
print("transform: " + str(enc.transform([['Female', 1], ['Male', 4]]).toarray()))
#
# coordinates = np.reshape(np.arange(1, 33), newshape=(32, 1))
# move1 = np.full(fill_value=1, shape=(32, 1))
# move2 = np.full(fill_value=2, shape=(32, 1))
# move3 = np.full(fill_value=3, shape=(32, 1))
# move4 = np.full(fill_value=4, shape=(32, 1))
#
# part1 = np.append(coordinates, move1, axis=1)
# part2 = np.append(coordinates, move2, axis=1)
# part3 = np.append(coordinates, move3, axis=1)
# part4 = np.append(coordinates, move4, axis=1)
#
# table = np.append(coordinates * 1, coordinates * 2, axis=0)
# table = np.append(table, coordinates * 3, axis=0)
# table = np.append(table, coordinates * 4, axis=0)

enc = OneHotEncoder()
enc.fit(np.reshape(np.arange(1, 129), newshape=(128, 1)))

print("categries_" + str(enc.categories_))
print("transform: " + str(enc.transform([[2]]).toarray()))
print("transform: " + str(enc.transform([[127]]).toarray()))

print("")