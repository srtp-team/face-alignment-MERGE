import face_alignment
from skimage import io
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

input = io.imread('../face_alignment/test/assets/aflw-test.jpg')
preds = fa.get_landmarks(input)