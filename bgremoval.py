#! /usr/bin/env python3

#
#
# Image Background Remover
# Version: 001-20180528
# For: Removing the background from Lauren's work photos at scale
#
# Uses Image Resizer and the instructions from StackOverflow:
# https://stackoverflow.com/questions/29313667/how-do-i-remove-the-background-from-this-kind-of-image
#
#


# ==========IMPORTS==========
import cv2
import numpy as np
import glob
import os

#== Parameters =======================================================================
BLUR = 11
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100
MASK_DILATE_ITER = 180
MASK_ERODE_ITER = 180
MASK_COLOR = (0.0,0.0,1.0) # In BGR format
# picture needs to be 368 pixels tall

#==best results
#BLUR = 11
#CANNY_THRESH_1 = 10
#CANNY_THRESH_2 = 100
#MASK_DILATE_ITER = 180
#MASK_ERODE_ITER = 180
#MASK_COLOR = (0.0,0.0,1.0) # In BGR format

#== Image Resizer ====================================================================
for pic in glob.glob("*.png"):
    file, ext = os.path.splitext(pic)
    img = cv2.imread(pic)
    width = 368
    height = int(img.shape[0] * (width/int(img.shape[1])))
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(file + '_resized.png', resized)

#==Background Removal=================================================================

#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
for pic in glob.glob("*.png"):
	file, ext = os.path.splitext(pic)
	img = cv2.imread(pic)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#-- Edge detection -------------------------------------------------------------------
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)

#-- Find contours in edges, sort by area ---------------------------------------------
	contour_info = []
	_, contours, _= cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in contours:
		contour_info.append((
			c,
			cv2.isContourConvex(c),
			cv2.contourArea(c),
		))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
	mask = np.zeros(edges.shape)
	cv2.fillConvexPoly(mask, max_contour[0], (255))

#-- Smooth mask, then blur it --------------------------------------------------------
	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
	mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#-- Blend masked img into MASK_COLOR background --------------------------------------
	mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
	img         = img.astype('float32') / 255.0                 #  for easy blending

	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
	masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

	#cv2.imshow('img', masked)                                   # Display
	#cv2.waitKey()

#cv2.imwrite('C:/Temp/person-masked.jpg', masked)           # Save

# split image into channels
	c_red, c_green, c_blue = cv2.split(img)

# merge with mask got on one of a previous steps
	img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

# show on screen (optional in jupiter)
# %matplotlib inline
# plt.imshow(img_a)
# plt.show()

# save to disk
	cv2.imwrite(file + '_edited.png', img_a*255)

# or the same using plt
#	plt.imsave('girl_2.png', img_a)
