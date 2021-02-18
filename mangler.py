from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
from skimage.transform import PiecewiseAffineTransform, warp
import random

def getpicture():
    img_transfer = ski.io.imread('romano2.jpg')
    image_name = "output.jpg"
    ski.io.imsave(image_name,img_transfer)

    for x in range(1):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        try:
            w = rects[0].width()
            h = rects[0].height()
        except IndexError:
            print("No face found")
        face_mesh = np.empty((0,2))

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            face_mesh = np.concatenate((face_mesh, shape))
            # loop over the face parts individually
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                for (x, y) in shape[i:j]:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        
        # visualize all facial landmarks with a transparent overlay
        # output = face_utils.visualize_facial_landmarks(image, shape)
        # cv2.imshow("Image", output)
        # cv2.waitKey(0)
                
        img = ski.io.imread(image_name)
        rows, cols = img.shape[0], img.shape[1]
        colspace = np.linspace(0, cols, 20)
        src_cols=np.empty((0,2))
        rowspace = np.linspace(0, rows, 20)
        src_rows=np.empty((0,2))
        for i in range(colspace.shape[0]):
            src_cols = np.append(src_cols,[[colspace[i],0]],axis=0)
        for i in range(colspace.shape[0]):
            src_cols = np.append(src_cols,[[colspace[i],rows]],axis=0)
        for i in range(rowspace.shape[0]):
            src_rows = np.append(src_rows,[[0,rowspace[i]]],axis=0)
        for i in range(rowspace.shape[0]):
            src_rows = np.append(src_rows,[[cols,rowspace[i]]],axis=0)
        src = np.concatenate((src_rows,src_cols))
        scale_factor = 25

        src_mesh = face_mesh
        src_mesh = np.concatenate((src_mesh,src))
        face_proportion = np.mean([w/(img.shape[0]*1),h/(img.shape[1]*1)])
        max = (face_proportion)*scale_factor

        def stretchfeatures(mesh, gmax):
            regions = [range(0,68),range(0,17),range(36,42),range(42,48),range(48,60),range(60,68)]
            regions = np.random.choice(regions,size=2)
            max = 1*gmax
            scale =  np.random.uniform(0,high=.5)
            for region in regions:
                centerx = np.mean(mesh[region,0])
                centery = np.mean(mesh[region,1])
                direction = 1 if random.random() < 0.5 else -1
                for i in region:
                    x = mesh[i,0]
                    y = mesh[i,1]
                    xcomp = x-centerx
                    ycomp = y-centery
                    if ycomp == 0:
                        ycomp = .1
                    distance = np.sqrt(np.square(xcomp)+np.square(ycomp))
                    dst_distance = distance+(max*scale*direction)
                    angle = np.arcsin(xcomp/distance)
                    dstx = centerx + dst_distance*np.sin(angle)
                    angle = np.arccos(ycomp/distance) 
                    dsty = centery + dst_distance*np.cos(angle)
                    mesh[i] = [dstx,dsty]
            return mesh

        def wigglefeatures(mesh):
            regions = [range(0,17),range(17,22),range(22,27),range(27,31),range(31,36),range(36,42),range(42,48),range(48,68)]
            # regions = np.random.choice(regions,size=5)
            for i in regions:
                if max >=1:
                    rm = (max*.5)+np.random.randint(.5*max,high=max)
                    ra = np.random.uniform(0.0,high=2.0*(np.pi))
                    rx = np.cos(ra)*rm
                    ry = np.sin(ra)*rm
                else:
                    rx = 0
                    ry = 0
                for j in i:
                    mesh[j] = [mesh[j,0]+rx,mesh[j,1]+ry]
            return mesh
        
        dst_mesh = face_mesh
        dst_mesh = stretchfeatures(dst_mesh, max)
        dst_mesh = wigglefeatures(dst_mesh)
        dst_mesh = np.concatenate((dst_mesh, src))

        tform = PiecewiseAffineTransform()
        tform.estimate(src_mesh, dst_mesh)

        out_rows = img.shape[0]
        out_cols = img.shape[1]
        out = warp(img, tform)
        ski.io.imsave(image_name,out)

        # fig, ax = plt.subplots()
        # ax.imshow(out)
        # ax.plot(tform.inverse(src_mesh)[:, 0], tform.inverse(src_mesh)[:, 1], '.b')
        # ax.axis((0, out_cols, out_rows, 0))
        # plt.show()