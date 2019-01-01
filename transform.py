#!/usr/bin/env python

""" 
Original code for rotation belongs to Alex & Gareth Rees, see https://codereview.stackexchange.com/questions/41688/rotating-greyscale-images.
Code for scaling and shear written by L. Zuehsow (Oktober13)
The transform matrices act on individual cells from the original source image, which are treated as "points."
It's important to only act on cells in the source image and determine where they should be placed in the destination image. 
The reverse operation is prone to indexing issues when the destination image is a different size from the original.
"""

import cv2
import math
import numpy as np

class Transform(object):
	def __init__(self):
		pass

	@staticmethod
	def rotate_coords(coord_list, theta, origin):
		"""Rotate arrays of coordinates x and y by theta radians about the origin point.
		"""
		(ox, oy) = origin
		x, y = coord_list[:][0], coord_list[:][1] # Vector of all x and all y coordinates for corners.

		sin, cos = np.sin(theta), np.cos(theta)
		x, y = np.asarray(x) - ox, np.asarray(y) - oy
		return x * cos - y * sin + ox, x * sin + y * cos + oy

	@staticmethod
	def rotate_image(src, theta, origin, fill=255):
		"""Rotate the image src by theta radians about (ox, oy).
		Pixels in the result that don't correspond to pixels in src are
		replaced by the value fill (White).
		We try to find corresponding pixels from the destination image,
		and match them to the source image to eliminate attempts to access nonexistent pixels in source.
		"""

		t = Transform()

		# Images have origin at the top left, so the angle is negated. (Clockwise rotation)
		theta = -theta
		# Dimensions of source image. Note that scipy.misc.imread loads
		# images in row-major order, so src.shape gives (height, width).
		height, width, channels = src.shape

		# Positions of the corners of the source image. (Vector)
		corners = np.transpose([[0,0],[width,0],[width,height],[0,height]])
		cx, cy = t.rotate_coords(corners, theta, origin)

		# Determine dimensions of destination image by finding extremes.
		newWidth, newHeight = (int(np.ceil(c.max() - c.min())) for c in (cx, cy))

		# Iterate over image to form grid of coordinates of pixels in destination image.
		dx, dy = np.meshgrid(np.arange(newWidth), np.arange(newHeight))
		pixels = [dx + cx.min(), dy + cy.min()]

		# Corresponding coordinates in source image. Since we are
		# transforming dest-to-src here, the rotation is negated.
		sx, sy = t.rotate_coords(pixels, theta, origin)
		# Select nearest neighbour pixel. Index must be integer.
		sx, sy = sx.round().astype(int), sy.round().astype(int)

		# Mask for valid coordinates.
		mask = (0 <= sx) & (sx < width) & (0 <= sy) & (sy < height)
		# Create destination image.
		dest = np.empty(shape=(newHeight, newWidth, channels), dtype=src.dtype)

		# Copy valid coordinates from source image.
		dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
		# Fill invalid coordinates. Fill inverted mask with white.
		dest[dy[~mask], dx[~mask]] = fill
		return dest

	@staticmethod
	def scaleCoords(coord_list, scale):
		""" Scale the coordinates of the four corners of the image. """
		scaleX, scaleY = scale[0][0], scale[1][1]
		x, y = coord_list[:][0], coord_list[:][1] # Vector of all x and all y coordinates for corners.
		sx, sy = np.dot(x,1.0/scaleX), np.dot(y, 1.0/scaleY)
		return sx, sy

	@staticmethod
	def scale(src, scale, fill=255):
		""" Scale an image horizontally or vertically. """

		height, width, channels = src.shape

		# Positions of the corners of the source image. (Vector)
		corners = np.transpose([[0,0],[width,0],[width,height],[0,height]])

		# Determine dimensions of destination image by finding extremes.
		newWidth = int(scale[0][0] * (int(np.ceil(corners[0].max() - corners[0].min()))))
		newHeight = int(scale[1][1] * (int(np.ceil(corners[1].max() - corners[1].min()))))

		# Iterate over image to form grid of coordinates of pixels in destination image.
		dx, dy = np.meshgrid(np.arange(newWidth), np.arange(newHeight))
		pixels = [dx + corners[0].min(), dy + corners[1].min()] # Moves into frame

		sx, sy = t.scaleCoords(pixels, scale)

		# Select nearest neighbour pixel. Index must be integer.
		sx, sy = sx.round().astype(int), sy.round().astype(int)

		# Mask for valid coordinates.
		mask = (0 <= sx) & (sx < width) & (0 <= sy) & (sy < height)

		# Create destination image
		dest = np.empty(shape=(newHeight, newWidth, channels), dtype=src.dtype)

		# Copy valid coordinates from source image into destination image.
		dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
		
		# # Fill invalid coordinates. Fill inverted mask with white.
		dest[dy[~mask], dx[~mask]] = fill
		return dest


	@staticmethod
	def shearCoords(coord_list, scale, topRight = [[0][0]]):
		""" Shear the coordinates of the four corners of the image. """
		shearX, shearY = scale[0][1], scale[1][0]
		x, y = coord_list[:][0], coord_list[:][1] # Vector of all x and all y coordinates for corners.
		sx, sy = x + y * shearX, y + x * shearY
		if shearX > 0:
			sx = np.add(sx,-topRight[0])
		if shearY > 0:
			sy = np.add(sy,-topRight[1])
		return sx, sy

	@staticmethod
	def shear(src, scale, fill=255):
		""" Scale an image horizontally or vertically. """

		height, width, channels = src.shape

		# Positions of the corners of the source image. (Vector)
		corners = np.transpose([[0,0],[width,0],[width,height],[0,height]])

		# Determine dimensions of destination image by finding extremes.
		if shear[0][1] != 0:
			print shear[1][0]
			newWidth = int(np.ceil(corners[0].max() - corners[0].min())) + int(shear[0][1] * (int(np.ceil(corners[1].max() - corners[1].min()))))
		else:
			newWidth = int(np.ceil(corners[0].max() - corners[0].min()))
		if shear[1][0] != 0:
			newHeight = int(np.ceil(corners[1].max() - corners[1].min())) + int(shear[1][0] * (int(np.ceil(corners[0].max() - corners[0].min()))))
		else:
			newHeight = int(np.ceil(corners[1].max() - corners[1].min()))

		print newWidth, newHeight, shear[0][1], shear[1][0]

		# Iterate over image to form grid of coordinates of pixels in destination image.
		dx, dy = np.meshgrid(np.arange(newWidth), np.arange(newHeight))
		pixels = [dx + corners[0].min(), dy + corners[1].min()] # Moves into frame

		sx, sy = t.shearCoords(pixels, shear, [newWidth-width,newHeight-height])

		# Select nearest neighbour pixel. Index must be integer.
		sx, sy = sx.round().astype(int), sy.round().astype(int)

		# Mask for valid coordinates.
		mask = (0 <= sx) & (sx < width) & (0 <= sy) & (sy < height)

		# Create destination image
		dest = np.empty(shape=(newHeight, newWidth, channels), dtype=src.dtype)

		# Copy valid coordinates from source image into destination image.
		dest[dy[mask], dx[mask]] = src[sy[mask], sx[mask]]
		
		# # Fill invalid coordinates. Fill inverted mask with white.
		dest[dy[~mask], dx[~mask]] = fill
		return dest

	@staticmethod
	def transform(src, angle = 0, scale = None, shear = None):
		""" Wrapper function, performs requested transforms. """
		dst = src

		if scale is not None or np.identity(2):
			if scale[0][0] > 0 and scale[1][1] > 0:
				dst = t.scale(dst, scale)
			else:
				print "INVALID INPUT: Negative or zero scale is not allowed."
		if shear is not None or np.identity(2):
			if shear[0][0] > 0 and shear[1][1] > 0:
				dst = t.shear(dst,shear)
			else:
				print "INVALID INPUT: Negative or zero shear is not allowed."
		if angle is not 0:
			origin = (int(dst.shape[1] / 2), int(dst.shape[0] / 2)) # y, x
			dst = t.rotate_image(dst, angle * np.pi / 180, origin)
		return dst

	@staticmethod
	def display(old, new):
		cv2.imshow("old", old)
		cv2.imshow("new", new)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	angle = 45
	scale = np.array([[2,0],[0,1]])
	shear = np.array([[1,0.5],[0,1]])

	t = Transform()

	src = cv2.imread("frisk.jpg")

	fin = t.transform(src, angle, scale, shear)
	# dst = t.rotate_image(src, angle * np.pi / 180, (100, 100))
	t.display(src, fin)
