import face_alignment
from skimage import io
from pylab import plot
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

input = io.imread('G:\aflw-test.jpg')
preds = fa.get_landmarks(input)

plt.imshow(input)
for pred in preds:
	for pre in pred:
		plot(pre[0], pre[1], "r*")

plt.show()
