#!/usr/bin/env python

import cv2
import math
import numpy as np

class Transform(object):
	def __init__(self):
		pass

	def getRMat(self, (cx, cy), angle, scale):
		"""
		Calculates rotational matrix
		"""
		a = scale*math.cos(angle*np.pi/180)
		b = scale*(math.sin(angle*np.pi/180))
		# u = (1-a)*cx-b*cy
		# v = b*cx+(1-a)*cy
		u = cx * a + cy * b
		v = cx * b + cy * a
		# round (x*cos(anrad)+ y*sin(anrad));
  #   h= round (x*sin(anrad)+ y*cos(anrad));
		print u,v
		return np.array([[a,b,u], [-b,a,v]]) 

	@staticmethod
	def warpAffine2D(image, matrix, width, height):
		"""
		For each cell in destination matrix (index u,v),
		the corresponding (transformed) source matrix indices (x,y) are calculated for each index.
		We need to work backwards from the destination matrix because the alternative is
		potentially having cells in dst that don't correspond to any source cell.
		"""
		dst = np.zeros((height, width, 3), dtype=np.uint8) # Create destination np array
		oldh, oldw = image.shape[:2]
		for u in range(width):
			for v in range(height):
				x = u*matrix[0,0]+v*matrix[0,1]+matrix[0,2]
				y = u*matrix[1,0]+v*matrix[1,1]+matrix[1,2]
				intx, inty = int(x), int(y)
				if 0 < intx < oldw and 0 < inty < oldh: # Only copy the src cell to dst cell if it exists.
					pix = image[inty, intx]
					dst[v, u] = pix
		return dst

	@staticmethod
	def display(old, new):
		cv2.imshow("old", old)
		cv2.imshow("new", new)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	# orig = np.identity(3)
	isoScale = np.array([[0.5,0,0],[0,0.5,0],[0,0,0.5]])
	# angle = math.pi
	angle = 45
	# rot = np.array([[1,0,0],[0, math.cos(angle), math.sin(angle)],[0, math.sin(angle), math.cos(angle)]])
	t = Transform()

	src = cv2.imread("frisk.jpg")

	h, w, channels = src.shape
	cx, cy = (h / 2, w / 2)
	mat = t.getRMat((cx, cy), int(angle), 1)
	# print mat
	cos = np.abs(mat[0,0])
	sin  = np.abs(mat[0,1])
	newWidth = int((h * sin) + (w * cos))
	newHeight = int((h * cos) + (w * sin))
	print newHeight, newWidth
	mat[0,2] += cx - (newWidth / 2)
	mat[1,2] += cy - (newHeight / 2)

	# print mat[:,2]
	# print cx, cy

	# newHeight, newWidth, channels = src.shape
	dst = t.warpAffine2D(src, mat, newWidth, newHeight)
	t.display(src, dst)