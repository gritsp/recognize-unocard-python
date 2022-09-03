from copyreg import constructor
from turtle import st
import cv2

# img = cv2.imread("./uno_deck3.jpg")
# (y,x) = img.shape[:2]
# print(y,x)
# t = 11
# img2 = img[0+t+3:y-t-3, 0+t:x-t]
# cv2.imwrite("uno.jpg",img2)
# print(img2.shape[:2])
# cv2.imshow('img',img2)

img = cv2.imread("./uno.jpg")
w = 175
h = 263
dic = {10:"stop",11:"revert",12:"plus",13:"plus4"}

for i in range(14):
  # print(i*h,i*h+h)
  img2 = img[h*3:h*3+h, i*w:i*w+w]
  # print(i,img2.shape[:2])
  # cv2.imshow("img"+str(i),img2)
  if i>9:
    cv2.imwrite("./data/red_{}.jpg".format(dic[i]),img2)
  else:
    cv2.imwrite("./data/red_{}.jpg".format(i),img2)
# cv2.waitKey()
# cv2.destroyAllWindows()
