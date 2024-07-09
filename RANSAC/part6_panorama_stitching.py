import numpy as np
import cv2

class PanoramaStitcher:
    def __init__(self):
        pass

    def detect_feature_and_keypoints(self, image):
        descriptors = cv2.SIFT_create()
        (keypoints, features) = descriptors.detectAndCompute(image, None)
        return keypoints, features

    def match_keypoints(self, keypointsA, keypointsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        matches = []

        for m, n in rawMatches:
            if m.distance < n.distance * ratio:
                matches.append((m.trainIdx, m.queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([keypointsA[i].pt for (_, i) in matches])
            ptsB = np.float32([keypointsB[i].pt for (i, _) in matches])

            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return matches, H, status
        return None

    def image_stitch(self, images, ratio=0.75, reprojThresh=4.0, match_status=False):
        (imageB, imageA) = images
        (keypointsA, featuresA) = self.detect_feature_and_keypoints(imageA)
        (keypointsB, featuresB) = self.detect_feature_and_keypoints(imageB)

        M = self.match_keypoints(keypointsA, keypointsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None

        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        if match_status:
            vis = self.draw_matches(imageA, imageB, keypointsA, keypointsB, matches, status)
            return (result, vis)

        return result, None

    def draw_matches(self, imageA, imageB, keypointsA, keypointsB, matches, status):
        pass

def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized