import sys
import cv2
import imutils
from itertools import combinations
import numpy as np

def corner_detection(im):
    orig_imsize = np.shape(im)[0:2]
    # Resize and change color to greys.
    im = imutils.resize(im, height=500)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imsize = np.shape(im)
    
    # Find Hough lines in the image.
    img = cv2.GaussianBlur(img, (3, 3), 0) 
    edges = cv2.Canny(img, 55, 175)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
    lines = lines.reshape(np.shape(lines)[0], 2)
    
    # Find the document's corners. We select as corners those defined as the intersection
    # of four lines, which are parallel two by two, and have maximal area.
    max_area = -1.0
    doc_corners = []
    # Iterate through every pair of lines. 
    for comb in combinations(range(np.shape(lines)[0]), 4):
        ρ = lines[comb, 0]
        θ = lines[comb, 1]
        
        # Check that pairs of lines are parallel.
        def are_parallel(θ1, θ2):
            return np.abs(θ1-θ2) < np.pi/8 or np.abs(np.abs(θ1-θ2)-np.pi) < np.pi/8
                                                     
        if (not (are_parallel(θ[0], θ[1]) and are_parallel(θ[2], θ[3]))
            and not (are_parallel(θ[0], θ[2]) and are_parallel(θ[1], θ[3]))
            and not (are_parallel(θ[0], θ[3]) and are_parallel(θ[1], θ[2]))
        ):
            continue
        
        # Find all 4 corners.
        def line_intersection(ρ1, ρ2, θ1, θ2):
            x = (ρ1/np.sin(θ1) - ρ2/np.sin(θ2))/(1/np.tan(θ1) - 1/np.tan(θ2))
            y = (ρ1 - x*np.cos(θ1))/np.sin(θ1)
            return [x, y]
        
        corners = []
        if are_parallel(θ[0], θ[1]) and are_parallel(θ[2], θ[3]):
            corners.append(line_intersection(ρ[0], ρ[2], θ[0], θ[2]))
            corners.append(line_intersection(ρ[0], ρ[3], θ[0], θ[3]))
            corners.append(line_intersection(ρ[1], ρ[2], θ[1], θ[2]))
            corners.append(line_intersection(ρ[1], ρ[3], θ[1], θ[3]))
        if are_parallel(θ[0], θ[2]) and are_parallel(θ[1], θ[3]):
            corners.append(line_intersection(ρ[0], ρ[1], θ[0], θ[1]))
            corners.append(line_intersection(ρ[0], ρ[3], θ[0], θ[3]))
            corners.append(line_intersection(ρ[2], ρ[1], θ[2], θ[1]))
            corners.append(line_intersection(ρ[2], ρ[3], θ[2], θ[3]))
        if are_parallel(θ[0], θ[3]) and are_parallel(θ[1], θ[2]):
            corners.append(line_intersection(ρ[0], ρ[1], θ[0], θ[1]))
            corners.append(line_intersection(ρ[0], ρ[2], θ[0], θ[2]))
            corners.append(line_intersection(ρ[3], ρ[1], θ[3], θ[1]))
            corners.append(line_intersection(ρ[3], ρ[2], θ[3], θ[2]))
        corners = np.array(corners)
        
        # Compute area inside lines.
        def triangle_area(x, y):
            return np.abs((x[0]-x[2])*(y[1]-y[0]) - (x[0]-x[1])*(y[2]-y[0]))/2
        area = (triangle_area(corners[[0, 1, 2], 0], corners[[0, 1, 2], 1])
             + triangle_area(corners[[0, 1, 3], 0], corners[[0, 1, 3], 1]))

        # Choose if they are candidate for document's corners.
        if area > max_area:
            doc_corners = corners
            max_area = area
    
    # Rearrange document's corners.
    ix_upleft_corner = np.argwhere(np.logical_and(doc_corners[:, 0] < imsize[0]/2, 
                                                  doc_corners[:, 1] < imsize[1]/2))[0, 0]
    ix_upright_corner = np.argwhere(np.logical_and(doc_corners[:, 0] < imsize[0]/2, 
                                                   doc_corners[:, 1] > imsize[1]/2))[0, 0]
    ix_doleft_corner = np.argwhere(np.logical_and(doc_corners[:, 0] > imsize[0]/2, 
                                                  doc_corners[:, 1] < imsize[1]/2))[0, 0]
    ix_doright_corner = np.argwhere(np.logical_and(doc_corners[:, 0] > imsize[0]/2, 
                                                   doc_corners[:, 1] > imsize[1]/2))[0, 0]
    doc_corners = doc_corners[[ix_upleft_corner, ix_upright_corner, ix_doleft_corner, ix_doright_corner]]
    
    # Rescale document's corners to original size.
    doc_corners[:, 0] *= orig_imsize[0]/imsize[0]
    doc_corners[:, 1] *= orig_imsize[1]/imsize[1]
    
    return doc_corners

def perspective_transformation(im, doc_corners):
    imsize = np.shape(im)
    pts1 = np.float32(doc_corners)
    pts2 = np.float32([[0, 0], [0, imsize[0]], [imsize[1], 0], [imsize[1], imsize[0]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    imt = cv2.warpPerspective(im, M, (imsize[1], imsize[0]))
    return imt

def do_image_thresholding(im):
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    return img

def scanner_main(image_path, dest_name):
    im = cv2.imread(image_path)
    orig = im.copy()
    
    doc_corners = corner_detection(im)
    
    im = perspective_transformation(im, doc_corners)
    
    im = do_image_thresholding(im)
    cv2.imwrite(dest_name, im)
    
    return

if len(sys.argv) >= 3:
    scanner_main(sys.argv[1], sys.argv[2])
else:
    print("Please, introduce the path of the original image and the path of the destination image.")
    sys.exit()

