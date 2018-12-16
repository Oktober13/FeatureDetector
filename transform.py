#!/usr/bin/env python

""" Original code belongs to Alex & Gareth Rees, see https://codereview.stackexchange.com/questions/41688/rotating-greyscale-images"""

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
		sx, sy = t.rotate_coords(pixels, -theta, origin)
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
	def scale(src):
		height, width, channels = src.shape
		# Positions of the corners of the source image. (Vector)
		corners = np.transpose([[0,0],[width,0],[width,height],[0,height]])
		# Determine dimensions of destination image by finding extremes.
		newWidth, newHeight = scale*(int(np.ceil(corners.max() - corners.min())) for c in (cx, cy))

		# Iterate over image to form grid of coordinates of pixels in destination image.
		dx, dy = np.meshgrid(np.arange(newWidth), np.arange(newHeight))
		pixels = [dx + cx.min(), dy + cy.min()] # Moves into frame

		(ox, oy) = origin
		x, y = coord_list[:][0], coord_list[:][1] # Vector of all x and all y coordinates for corners.

		sin, cos = np.sin(theta), np.cos(theta)
		x, y = np.asarray(x) - ox, np.asarray(y) - oy
		# return x * cos - y * sin + ox, x * sin + y * cos + oy

		# Corresponding coordinates in source image. Since we are
		# transforming dest-to-src here, the rotation is negated.
		# sx, sy = 
		# Select nearest neighbour pixel. Index must be integer.
		sx, sy = sx.round().astype(int), sy.round().astype(int)

		pass

	@staticmethod
	def transform(src, angle, scale):
		if angle is not 0:
			dst = t.rotate_image(src, angle * np.pi / 180, (100, 100))
		if scale is not 1:
			# dst = 
			pass
		return dst

	@staticmethod
	def display(old, new):
		cv2.imshow("old", old)
		cv2.imshow("new", new)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	angle = 45
	t = Transform()

	src = cv2.imread("frisk.jpg")

	height, width, channels = src.shape
	cx, cy = (width / 2.0, height / 2.0)

	fin = t.transform(src, angle, 1.0)
	# dst = t.rotate_image(src, angle * np.pi / 180, (100, 100))
	t.display(src, fin)