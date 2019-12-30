{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "flat_chess=cv2.imread('../DATA/flat_chessboard.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x280ce95e320>"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(flat_chess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "found,corners = cv2.findChessboardCorners(flat_chess,(7,7))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "found # TO check if there is no blur"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 43.500004,  43.500004]],\n",
       "\n",
       "       [[ 87.5     ,  43.5     ]],\n",
       "\n",
       "       [[131.5     ,  43.5     ]],\n",
       "\n",
       "       [[175.5     ,  43.5     ]],\n",
       "\n",
       "       [[219.5     ,  43.5     ]],\n",
       "\n",
       "       [[263.5     ,  43.5     ]],\n",
       "\n",
       "       [[307.5     ,  43.5     ]],\n",
       "\n",
       "       [[ 43.499996,  87.50001 ]],\n",
       "\n",
       "       [[ 87.5     ,  87.5     ]],\n",
       "\n",
       "       [[131.5     ,  87.5     ]],\n",
       "\n",
       "       [[175.5     ,  87.5     ]],\n",
       "\n",
       "       [[219.5     ,  87.5     ]],\n",
       "\n",
       "       [[263.5     ,  87.5     ]],\n",
       "\n",
       "       [[307.5     ,  87.49999 ]],\n",
       "\n",
       "       [[ 43.500004, 131.5     ]],\n",
       "\n",
       "       [[ 87.5     , 131.5     ]],\n",
       "\n",
       "       [[131.5     , 131.5     ]],\n",
       "\n",
       "       [[175.5     , 131.5     ]],\n",
       "\n",
       "       [[219.5     , 131.5     ]],\n",
       "\n",
       "       [[263.5     , 131.5     ]],\n",
       "\n",
       "       [[307.5     , 131.5     ]],\n",
       "\n",
       "       [[ 43.499996, 175.5     ]],\n",
       "\n",
       "       [[ 87.5     , 175.5     ]],\n",
       "\n",
       "       [[131.5     , 175.5     ]],\n",
       "\n",
       "       [[175.5     , 175.5     ]],\n",
       "\n",
       "       [[219.5     , 175.5     ]],\n",
       "\n",
       "       [[263.5     , 175.5     ]],\n",
       "\n",
       "       [[307.5     , 175.5     ]],\n",
       "\n",
       "       [[ 43.500004, 219.5     ]],\n",
       "\n",
       "       [[ 87.5     , 219.5     ]],\n",
       "\n",
       "       [[131.5     , 219.5     ]],\n",
       "\n",
       "       [[175.5     , 219.5     ]],\n",
       "\n",
       "       [[219.5     , 219.5     ]],\n",
       "\n",
       "       [[263.5     , 219.5     ]],\n",
       "\n",
       "       [[307.5     , 219.5     ]],\n",
       "\n",
       "       [[ 43.499996, 263.5     ]],\n",
       "\n",
       "       [[ 87.5     , 263.5     ]],\n",
       "\n",
       "       [[131.5     , 263.5     ]],\n",
       "\n",
       "       [[175.5     , 263.5     ]],\n",
       "\n",
       "       [[219.5     , 263.5     ]],\n",
       "\n",
       "       [[263.5     , 263.5     ]],\n",
       "\n",
       "       [[307.5     , 263.5     ]],\n",
       "\n",
       "       [[ 43.5     , 307.5     ]],\n",
       "\n",
       "       [[ 87.5     , 307.5     ]],\n",
       "\n",
       "       [[131.5     , 307.5     ]],\n",
       "\n",
       "       [[175.5     , 307.5     ]],\n",
       "\n",
       "       [[219.5     , 307.5     ]],\n",
       "\n",
       "       [[263.5     , 307.5     ]],\n",
       "\n",
       "       [[307.5     , 307.5     ]]], dtype=float32)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corners"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        ...,\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209]],\n",
       "\n",
       "       [[158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        ...,\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209]],\n",
       "\n",
       "       [[158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        ...,\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        ...,\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255]],\n",
       "\n",
       "       [[ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        ...,\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255]],\n",
       "\n",
       "       [[ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        [ 71, 139, 209],\n",
       "        ...,\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255],\n",
       "        [158, 206, 255]]], dtype=uint8)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv2.drawChessboardCorners(flat_chess,(7,7),corners,found)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x280cea10e48>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(flat_chess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "dots = cv2.imread('../DATA/dot_grid.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x280cea7a7b8>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(dots)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "found,corners=cv2.findCirclesGrid(dots,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "found"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 29.5,  29.5]],\n",
       "\n",
       "       [[ 89.5,  29.5]],\n",
       "\n",
       "       [[149.5,  29.5]],\n",
       "\n",
       "       [[209.5,  29.5]],\n",
       "\n",
       "       [[269.5,  29.5]],\n",
       "\n",
       "       [[329.5,  29.5]],\n",
       "\n",
       "       [[389.5,  29.5]],\n",
       "\n",
       "       [[449.5,  29.5]],\n",
       "\n",
       "       [[509.5,  29.5]],\n",
       "\n",
       "       [[569.5,  29.5]],\n",
       "\n",
       "       [[ 29.5,  89.5]],\n",
       "\n",
       "       [[ 89.5,  89.5]],\n",
       "\n",
       "       [[149.5,  89.5]],\n",
       "\n",
       "       [[209.5,  89.5]],\n",
       "\n",
       "       [[269.5,  89.5]],\n",
       "\n",
       "       [[329.5,  89.5]],\n",
       "\n",
       "       [[389.5,  89.5]],\n",
       "\n",
       "       [[449.5,  89.5]],\n",
       "\n",
       "       [[509.5,  89.5]],\n",
       "\n",
       "       [[569.5,  89.5]],\n",
       "\n",
       "       [[ 29.5, 149.5]],\n",
       "\n",
       "       [[ 89.5, 149.5]],\n",
       "\n",
       "       [[149.5, 149.5]],\n",
       "\n",
       "       [[209.5, 149.5]],\n",
       "\n",
       "       [[269.5, 149.5]],\n",
       "\n",
       "       [[329.5, 149.5]],\n",
       "\n",
       "       [[389.5, 149.5]],\n",
       "\n",
       "       [[449.5, 149.5]],\n",
       "\n",
       "       [[509.5, 149.5]],\n",
       "\n",
       "       [[569.5, 149.5]],\n",
       "\n",
       "       [[ 29.5, 209.5]],\n",
       "\n",
       "       [[ 89.5, 209.5]],\n",
       "\n",
       "       [[149.5, 209.5]],\n",
       "\n",
       "       [[209.5, 209.5]],\n",
       "\n",
       "       [[269.5, 209.5]],\n",
       "\n",
       "       [[329.5, 209.5]],\n",
       "\n",
       "       [[389.5, 209.5]],\n",
       "\n",
       "       [[449.5, 209.5]],\n",
       "\n",
       "       [[509.5, 209.5]],\n",
       "\n",
       "       [[569.5, 209.5]],\n",
       "\n",
       "       [[ 29.5, 269.5]],\n",
       "\n",
       "       [[ 89.5, 269.5]],\n",
       "\n",
       "       [[149.5, 269.5]],\n",
       "\n",
       "       [[209.5, 269.5]],\n",
       "\n",
       "       [[269.5, 269.5]],\n",
       "\n",
       "       [[329.5, 269.5]],\n",
       "\n",
       "       [[389.5, 269.5]],\n",
       "\n",
       "       [[449.5, 269.5]],\n",
       "\n",
       "       [[509.5, 269.5]],\n",
       "\n",
       "       [[569.5, 269.5]],\n",
       "\n",
       "       [[ 29.5, 329.5]],\n",
       "\n",
       "       [[ 89.5, 329.5]],\n",
       "\n",
       "       [[149.5, 329.5]],\n",
       "\n",
       "       [[209.5, 329.5]],\n",
       "\n",
       "       [[269.5, 329.5]],\n",
       "\n",
       "       [[329.5, 329.5]],\n",
       "\n",
       "       [[389.5, 329.5]],\n",
       "\n",
       "       [[449.5, 329.5]],\n",
       "\n",
       "       [[509.5, 329.5]],\n",
       "\n",
       "       [[569.5, 329.5]],\n",
       "\n",
       "       [[ 29.5, 389.5]],\n",
       "\n",
       "       [[ 89.5, 389.5]],\n",
       "\n",
       "       [[149.5, 389.5]],\n",
       "\n",
       "       [[209.5, 389.5]],\n",
       "\n",
       "       [[269.5, 389.5]],\n",
       "\n",
       "       [[329.5, 389.5]],\n",
       "\n",
       "       [[389.5, 389.5]],\n",
       "\n",
       "       [[449.5, 389.5]],\n",
       "\n",
       "       [[509.5, 389.5]],\n",
       "\n",
       "       [[569.5, 389.5]],\n",
       "\n",
       "       [[ 29.5, 449.5]],\n",
       "\n",
       "       [[ 89.5, 449.5]],\n",
       "\n",
       "       [[149.5, 449.5]],\n",
       "\n",
       "       [[209.5, 449.5]],\n",
       "\n",
       "       [[269.5, 449.5]],\n",
       "\n",
       "       [[329.5, 449.5]],\n",
       "\n",
       "       [[389.5, 449.5]],\n",
       "\n",
       "       [[449.5, 449.5]],\n",
       "\n",
       "       [[509.5, 449.5]],\n",
       "\n",
       "       [[569.5, 449.5]],\n",
       "\n",
       "       [[ 29.5, 509.5]],\n",
       "\n",
       "       [[ 89.5, 509.5]],\n",
       "\n",
       "       [[149.5, 509.5]],\n",
       "\n",
       "       [[209.5, 509.5]],\n",
       "\n",
       "       [[269.5, 509.5]],\n",
       "\n",
       "       [[329.5, 509.5]],\n",
       "\n",
       "       [[389.5, 509.5]],\n",
       "\n",
       "       [[449.5, 509.5]],\n",
       "\n",
       "       [[509.5, 509.5]],\n",
       "\n",
       "       [[569.5, 509.5]],\n",
       "\n",
       "       [[ 29.5, 569.5]],\n",
       "\n",
       "       [[ 89.5, 569.5]],\n",
       "\n",
       "       [[149.5, 569.5]],\n",
       "\n",
       "       [[209.5, 569.5]],\n",
       "\n",
       "       [[269.5, 569.5]],\n",
       "\n",
       "       [[329.5, 569.5]],\n",
       "\n",
       "       [[389.5, 569.5]],\n",
       "\n",
       "       [[449.5, 569.5]],\n",
       "\n",
       "       [[509.5, 569.5]],\n",
       "\n",
       "       [[569.5, 569.5]]], dtype=float32)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corners"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        ...,\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255]],\n",
       "\n",
       "       [[255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        ...,\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255]],\n",
       "\n",
       "       [[255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        ...,\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        ...,\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255]],\n",
       "\n",
       "       [[255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        ...,\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255]],\n",
       "\n",
       "       [[255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        ...,\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255],\n",
       "        [255, 255, 255]]], dtype=uint8)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv2.drawChessboardCorners(dots,(7,7),corners,found)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x280ce9eedd8>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.imshow(dots)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
