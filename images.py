import cv2
import matplotlib.pyplot as plt
#Uploading an image
image = cv2.imread('gallery/catphoto.jpg')

# Showing the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows

# Converting to grayscale
# image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# #Displaying gray image
# plt.imshow(image_gray, cmap='gray')
# plt.axis('off') #disables the axis
# plt.show()

#Smothering the image
# smoth_image = cv2.GaussianBlur(image,(15,15),0)
# smoth_image_rgb = cv2.cvtColor(smoth_image, cv2.COLOR_BGR2RGB)
# plt.imshow(smoth_image_rgb)
# plt.axis('off') #disables the axis
# plt.show()

#Border detecction
# image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# #Detect borders
# border = cv2.Canny(image_gray,100,200)
# plt.imshow(border, cmap='gray')
# plt.axis('off') #disables the axis
# plt.show()