from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()

print(digits.keys())

# ['data', 'target', 'frame', 'feature_names', 'target_names', 'images', 'DESCR']

n = len(digits.images)

image_data = digits.images.reshape((n, -1))
image_data.shape

labels = digits.target
labels


pca_transformer = PCA(n_components=0.8)
pca_images = pca_transformer.fit_transform(image_data)

pca_transformer.explained_variance_ratio_

pca_transformer.explained_variance_ratio_[:3].sum()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(100):
    ax.scatter(pca_images[i, 0], pca_images[i, 1],
               pca_images[i, 2], marker=r'{}$'.format(labels[i]), s=64)

ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.set_zlabel('Principal component 3')

plt.show()
