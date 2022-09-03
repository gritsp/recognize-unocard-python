import glob
import cv2
import numpy as np
from nonmax import non_max_suppression

def isColor(pixel):
  if (pixel[0]<10 or 175<=pixel[0]) and 25<=pixel[1] and 25<=pixel[2]:
    return 'Red'        
  elif 20<=pixel[0]<35 and 25<=pixel[1] and 25<=pixel[2]:
    return 'Yellow'        
  elif 35<=pixel[0]<90 and 25<=pixel[1]<=240 and 25<=pixel[2]:
    return 'Green'
  elif 90<=pixel[0]<130 and 25<=pixel[1] and 25<=pixel[2]:
    return 'Blue'
  else:
    return ''

def loadImg(path):
	data = {i.split("blue_")[1][0]:cv2.imread(i,cv2.IMREAD_GRAYSCALE) for i in glob.glob(path+'blue_*.jpg')}
	return data

def findCard(img,tpl,text):
	card = {'s':'stop','p':'plus 2','r':'revert','c':'color','w':'plus 4'}
	if not (text >= '0' and text <= '9'):
		text = card[text]
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	(ih,iw) = img_gray.shape[:2]
	(th,tw) = tpl.shape[:2]

	location = {}
	for scale in np.linspace(0.1, 4.0, 100)[::-1]:
		w = int(tw*scale)
		h = int(th*scale)
		resized = cv2.resize(tpl, (w,h))
		if resized.shape[0] > ih or resized.shape[1] > iw or resized.shape[0] < 5 or resized.shape[1] < 5:
			continue
		
		result = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
		threshold = 0.86
		(ys,xs) = np.where(result >= threshold)
		rects = []
		for (x,y) in zip(xs,ys):
			rects.append((x, y, x+w, y+h))
		
		picks = non_max_suppression(np.array(rects))
		for (x1, y1, x2, y2) in picks:
			tmp = img[y1:y2,x1:x2]
			w = int((x2-x1)/2)
			h = int((y2-y1)/8)
			color = isColor(cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV)[w,h])
			location[x1] = "{} {}".format(text,color)
			cv2.rectangle(img, (x1,y1), (x2, y2), (0,0,255), 3)
			cv2.putText(img, "{} {}".format(text,color), (x1-5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	return img,location

def main(data,img):
	location = {}
	for i in data:
		img,loc = findCard(img,data[i],i)
		for j in loc:
			location[j] = loc[j]
	sortLoc = [location[i] for i in sorted(location)]
	print(sortLoc)
	# return img
	cv2.imshow("Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def loadImgTest(path):
	data = [cv2.imread(i) for i in glob.glob(path+'card*')]
	return data

data = loadImg('./dataset/')
img = cv2.imread('./datatest/card_4_4.png')
main(data,img)
# dataTest = loadImgTest('./datatest/')
# name = 0
# print(dataTest)
# for i in dataTest:
# 	img = main(data,i)
	# print(img)
	# cv2.imwrite("./result/card_{}.jpg".format(name),img)
	# print("save ./result/card_{}.jpg".format(name))
	# name+=1