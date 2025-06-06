import cv2
import matplotlib.pylab as plt
import pytesseract

#Uploading the image
image = cv2.imread('gallery/ocr-photo.png')

#Applying Tesseract OCR, this will get the text from the image and appear in terminal
# text = pytesseract.image_to_string(image, lang='por')
# print(text)

#Resizing the image
# image_resize = cv2.resize(image, (300,300))
# cv2.imshow('resize', image_resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Applying Tesseract OCR in resized image
# text1 = pytesseract.image_to_string(image_resize, lang='por')
# print(text1)

#Gray scale for the image
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray image', image_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Border Detection with Canny
border = cv2.Canny(image_gray, 100,200)
cv2.imshow('border',border)
cv2.waitKey(0)
cv2.destroyAllWindows()