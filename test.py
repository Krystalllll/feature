"""
Convention: append an integer to the end of the test, for multiple versions of
the same test at different difficulties.  Higher numbers are more difficult
(lower thresholds or accept fewer mistakes).  Example:

    test_all_equal1(self):
        ...
    test_all_equal2(self):
        ...
"""

import argparse
import json
import os
import math
import random
import unittest

import cv2
import numpy as np
import scipy

import benchmark
import features


class TestHarrisKeypointDetector_computeLocalMaxima(unittest.TestCase):
    def setUp(self):
        self.height   = 20
        self.width    = 20
        self.imgsize  = (self.height, self.width)
        self.detector = features.HarrisKeypointDetector()
        self.img      = np.zeros(self.imgsize)

    def tearDown(self):
        pass

    def test_allSame(self):
        '''
        Check if computeLocalMaxima returns all True for a constant zero image.
        '''
        locmaximg = self.detector.computeLocalMaxima(self.img)
        # We accept either all True or all False, both are valid interpretations
        self.assertTrue(np.all(locmaximg == True) or np.all(locmaximg == False))

    def test_everySecond(self):
        '''
        Check if computeLocalMaxima returns alternating True/False for an
        image which contains alternating 0, 1 values
        '''
        self.img[::2] = 1
        locmaximg = self.detector.computeLocalMaxima(self.img)
        self.assertTrue(np.all(locmaximg[::2, ::2] == True))
        self.assertTrue(np.all(locmaximg[1::2, 1::2] == False))

    def sparse_increase_init(self, gap, axis):
        # place increasing values at every 6th position, so if
        # the filtering uses smaller kernel than 7x7, we get an error
        for h in range(self.height):
            for w in range(self.width):
                axisval = [h, w][axis]
                if h % gap == 0 and w % gap == 0:
                    self.img[h, w] = axisval + 1

    def test_7x7_smaller_gap(self):
        '''
        We create a grid of increasing numbers on one particular axis (like
        1000200030004 etc.) with just small enough gaps that no point is
        local maximum except nonzero elements in the last row/column
        (depending on the axis).
        '''
        kernelsize = 7
        gap = kernelsize // 2
        for axis in range(2):
            self.sparse_increase_init(gap, axis)
            locmaximg = self.detector.computeLocalMaxima(self.img)

            # the last selected row/column has True values!
            inds = np.zeros_like(self.img, np.bool)
            if axis == 0:
                lastind = (self.height - 1) // gap * gap
                inds[lastind, ::gap] = True
            elif axis == 1:
                lastind = (self.width - 1) // gap * gap
                inds[::gap, lastind] = True
            else:
                self.fail('Invalid axis specified')

            self.assertTrue(np.all(locmaximg[inds] == True))
            self.assertTrue(np.all(locmaximg[~inds] == False))

    def test_7x7_bigger_gap(self):
        '''
        We create a grid of increasing numbers on one particular axis (like
        1000020000300004 etc.) with just big enough gaps that all nonzero
        points are local maxima.
        '''
        kernelsize = 7
        gap = kernelsize // 2 + 1
        for axis in range(2):
            self.sparse_increase_init(gap, axis)
            locmaximg = self.detector.computeLocalMaxima(self.img)
            inds = np.zeros_like(self.img, np.bool)
            inds[::gap, ::gap] = True
            self.assertTrue(np.all(locmaximg[inds] == True))
            self.assertTrue(np.all(locmaximg[~inds] == False))


class TestHarrisKeypointDetector_computeHarrisValues(unittest.TestCase):
    def setUp(self):
        self.height   = 100
        self.width    = 100
        self.imgsize  = (self.height, self.width)
        self.detector = features.HarrisKeypointDetector()
        self.img      = np.zeros(self.imgsize)

    def tearDown(self):
        pass

    def test_grid_points1(self):
        '''
        Check if computeHarrisValues has high values at the 'cross' points
        in a grid image. (difficulty 1)
        '''
        self._grid_points(0.7, 7)

    def test_grid_points2(self):
        '''
        Check if computeHarrisValues has high values at the 'cross' points
        in a grid image. (difficulty 2)
        '''
        self._grid_points(0.8, 5)

    def test_grid_points3(self):
        '''
        Check if computeHarrisValues has high values at the 'cross' points
        in a grid image. (difficulty 3)
        '''
        self._grid_points(0.9, 3)

    def _grid_points(self, ratio_threshold, radius):
        gap       = 20
        thickness = 1
        for t in range(thickness):
            self.img[:, t::gap + thickness-1] = 1
            self.img[t::gap + thickness-1, :] = 1

        harrisImage, orientationImage = self.detector.computeHarrisValues(
            self.img)

        # the grid points should have high score, the other points should have
        # small
        maxval = np.max(harrisImage)
        inds   = np.zeros_like(self.img, np.bool)
        inds[thickness // 2 + gap + thickness - 1 :: gap + thickness - 1,
             thickness // 2 + gap + thickness - 1 :: gap + thickness - 1] = True

        # we should find a 'good' point somewhere in the neighborhood
        maxImg = scipy.ndimage.filters.maximum_filter(
            harrisImage, size=radius, mode='constant', cval=-1e100)

        self.assertTrue(np.all(maxImg[inds] > maxval * ratio_threshold))

    def test_edges(self):
        '''
        Check if the computed Harris scores are negative along a simple edge.
        '''
        # Bottom half of image is 1 and the top half is zero
        midpoint = self.height/2
        self.img[midpoint:,:] = 1
        harrisImage, orientationImage = self.detector.computeHarrisValues(
            self.img)

        radius = 5
        regionOfInterest = harrisImage[
            (midpoint-radius):(midpoint+radius),
            20:(self.width-20)]
        minImage = scipy.ndimage.filters.minimum_filter(
            regionOfInterest, size=radius*2)
        maxImage = scipy.ndimage.filters.maximum_filter(
            regionOfInterest, size=radius*2)

        self.assertTrue(np.all(minImage <= -10))
        self.assertTrue(np.all(maxImage <= 1e-2))


class TestHarrisKeypointDetector_detectKeypoints(unittest.TestCase):
    def setUp(self):
        self.height   = 100
        self.width    = 100
        self.imgsize  = (self.height, self.width)
        self.detector = features.HarrisKeypointDetector()
        self.img      = np.zeros(self.imgsize)

    def tearDown(self):
        pass

    def test_grid_points1(self):
        '''
        Check if detectKeypoints detects the 'cross' points in a grid image. (difficulty 1)
        '''
        self._grid_points(7)

    def test_grid_points2(self):
        '''
        Check if detectKeypoints detects the 'cross' points in a grid image. (difficulty 2)
        '''
        self._grid_points(5)

    def test_grid_points3(self):
        '''
        Check if detectKeypoints detects the 'cross' points in a grid image. (difficulty 3)
        '''
        self._grid_points(3)

    def _grid_points(self, radius):
        gap       = 20
        thickness = 1
        for t in range(thickness):
            self.img[:, t :: gap + thickness - 1] = 1
            self.img[t::gap + thickness - 1, :]   = 1

        colorimg  = np.tile(self.img[:, :, np.newaxis], (1, 1, 3))
        kps       = self.detector.detectKeypoints(colorimg)

        # the grid points should have high score, the other points should have
        # small
        inds = np.zeros_like(self.img, np.bool)
        inds[thickness // 2 :: gap + thickness - 1,
             thickness // 2 :: gap + thickness - 1] = True

        # we also want to accept points around the grid points in the 'radius'
        # radius
        coords = np.transpose(np.nonzero(inds))
        for h, w in coords:
            minh = max(h - radius // 2, 0)
            maxh = min(h + radius // 2, self.height) + 1
            minw = max(w - radius // 2, 0)
            maxw = min(w + radius // 2, self.width) + 1
            inds[minh : maxh, minw : maxw] = True

        kps = sorted(kps, key = lambda x : x.response, reverse=True)
        count = (self.height // (gap + thickness - 1)) * \
                (self.width  // (gap + thickness - 1))

        # get the best keypoints only
        kps = kps[:count]
        kpcoords = np.array([kp.pt for kp in kps], np.int)

        # mark the position of the detected keypoints
        indskp = np.zeros_like(self.img, np.bool)
        for y, x in kpcoords:
            indskp[y, x] = True

        self.assertTrue(np.all(inds[indskp] == True))


class TestSimpleFeatureDescriptor(unittest.TestCase):
    def setUp(self):
        self.height     = 100
        self.width      = 100
        self.imgsize    = (self.height, self.width)
        self.descriptor = features.SimpleFeatureDescriptor()
        self.img = np.random.random(self.imgsize + (3,)) * 255.0

    def tearDown(self):
        pass

    def test_random_image2(self):
        '''
        For all random patch in the randomized image, we should get back
        the original values. This is a simple descriptor, with no
        transformations applied.  In this test, patches near the edge
        are not tested.
        '''
        self._do_test_random_image(True)

    def test_random_image1(self):
        '''
        For all random patch in the randomized image, we should get back
        the original values. This is a simple descriptor, with no
        transformations applied.
        '''
        self._do_test_random_image(False)

    def _do_test_random_image(self, test_zero_pad):
        patchSize   = 5
        patchRadius = patchSize // 2

        image = self.img.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # zero padding
        grayImage = np.pad(grayImage, patchRadius, mode='constant')

        testCount = 100
        kps = []
        expDesc = np.zeros((testCount, patchSize * patchSize))
        for iPatch in range(testCount):
            if test_zero_pad:
                ry    = random.randint(0, self.height - 1)
                rx    = random.randint(0, self.width  - 1)
            else:
                ry    = random.randint(patchSize, self.height - 1 - patchSize)
                rx    = random.randint(patchSize, self.width  - 1 - patchSize)

            # we add patchRadius because of np.pad above
            miny  = ry - patchRadius + patchRadius
            minx  = rx - patchRadius + patchRadius
            patch = grayImage[miny : miny + patchSize, minx : minx + patchSize]
            expDesc[iPatch] = np.ravel(patch)

            kp          = cv2.KeyPoint()
            kp.pt       = (rx, ry)
            kp.angle    = random.random() * 360
            kp.response = random.random()
            kp.size     = random.random() * 20

            kps.append(kp)

        desc = self.descriptor.describeFeatures(self.img, kps)
        self.assertTrue(np.array_equal(expDesc, desc))


class TestMOPSFeatureDescriptor(unittest.TestCase):
    def setUp(self):
        self.height     = 100
        self.width      = 100
        self.imgsize    = (self.height, self.width)
        self.descriptor = features.MOPSFeatureDescriptor()
        testdatadir     = TestMOPSFeatureDescriptor.testdatadir

        self.img        = np.zeros(self.imgsize + (3,))

        self.arrowImg   = cv2.imread(os.path.join(testdatadir,
            'whitearrow-40x40.jpg'))

        self.angleCount = 6

        smallArrowImg   = cv2.imread(os.path.join(testdatadir,
            'whitearrow-8x8.jpg'))
        smallArrowImg   = cv2.cvtColor(smallArrowImg, cv2.COLOR_BGR2GRAY)

        normsmallarrowimg = np.ravel(smallArrowImg).astype(np.float) / 255.
        normsmallarrowimg = (normsmallarrowimg - np.mean(normsmallarrowimg)) / \
            np.std(normsmallarrowimg)
        self.expectedDesc = normsmallarrowimg

    def tearDown(self):
        pass

    def _paste_random_img(self, angle, pasteimg):
        # Paste arrow image into default image
        h, w = pasteimg.shape[:2]
        windowdiag = np.ceil(np.hypot(h, w))
        newimg = self.img.copy()
        centerx = random.randint(
            windowdiag // 2, self.height - windowdiag // 2 - 1)
        centery = random.randint(
            windowdiag // 2, self.width - windowdiag // 2 - 1)

        rx = centerx - w // 2
        ry = centery - h // 2
        newimg[ry:ry+h, rx:rx+w] = pasteimg

        rotmx = cv2.getRotationMatrix2D((centerx, centery), angle, 1.)

        # rotate the whole image around the center, the new pixels will be
        # padded with zeros...
        newimg = cv2.warpAffine(newimg, rotmx, newimg.shape[:2],
            flags=cv2.INTER_LINEAR)

        return newimg, centerx, centery

    def test_const_img(self):
        '''
        Check if they take care of the case where the standard deviation is 0.
        We give constant images as input.
        '''
        testcount = 100
        for i in range(testcount):
            rval = random.random() * 255
            newimg = np.full_like(self.img, rval)
            # get patches from the middle where blurring can't hurt us
            centery = random.randint(self.height//3, self.height*2//3)
            centerx = random.randint(self.width//3, self.width*2//3)

            # position to the middle of the const image
            kp          = cv2.KeyPoint()
            kp.pt       = (centerx, centery)
            kp.angle    = 0
            kp.response = random.random()
            kp.size     = random.random() * 20

            desc        = self.descriptor.describeFeatures(newimg, [kp])

            self.assertTrue(len(desc) > 0)
            self.assertTrue(np.allclose(desc[0], 0))

    def test_angles(self):
        '''
        We should get approximately the smaller rotated version of the
        input arrow in the image. The keypoint always points to the middle
        of the arrow.
        '''
        for i in range(self.angleCount):
            angle   = int(360.0 / self.angleCount * i)
            newimg, centerx, centery = self._paste_random_img(
                angle, self.arrowImg)

            # position to the middle of the arrow
            # try with both angles, because the writeup was ambiguous
            mindist = 1000
            for mult in [1., -1.]:
                kp          = cv2.KeyPoint()
                kp.pt       = (centerx, centery)
                kp.angle    = angle * mult
                kp.response = random.random()
                kp.size     = random.random() * 20

                desc        = self.descriptor.describeFeatures(newimg, [kp])

                self.assertTrue(len(desc) > 0)

                dist        = np.linalg.norm(self.expectedDesc - desc[0])
                mindist     = min(mindist, dist)

            self.assertLess(mindist, 6)

    def test_angles_double(self):
        '''
        This is similar to the previous test, but is less restrictive, since we
        generate two random rotated arrows and check if they have the same
        descriptors. The students' code can pass this if they consequently make
        rotation mistakes.
        '''
        testcount = 100
        for i in range(testcount):
            # Generate two random angles and rotated arrows
            descs = [[] for j in range(2)]
            for j in range(2):
                angle   = int(360.0 * random.random())
                newimg, centerx, centery = self._paste_random_img(
                    angle, self.arrowImg)

                # position to the middle of the arrow
                # try with both angles, because the writeup was ambiguous
                mindist = 1000
                for mult in [1., -1.]:
                    kp          = cv2.KeyPoint()
                    kp.pt       = (centerx, centery)
                    kp.angle    = angle * mult
                    kp.response = random.random()
                    kp.size     = random.random() * 20

                    desc        = self.descriptor.describeFeatures(newimg, [kp])
                    self.assertTrue(len(desc) > 0)
                    descs[j].append(desc[0])

            # compute the pairwise distance between the computed descriptors,
            # we get 4 distances
            dists = scipy.spatial.distance.cdist(
                np.array(descs[0]), np.array(descs[1]))
            mindist = np.min(dists)

            self.assertLess(mindist, 4)


class TestSSDFeatureMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = features.SSDFeatureMatcher()

    def tearDown(self):
        pass

    #def test_equals(self):
    #    '''
    #    If all descriptors are equal, the matches' matched index should be 0
    #    and all the distances should be approximately 0.
    #    '''
    #    num         = 10
    #    fetDim      = 10
    #    testCount   = 50

    #    for t in range(testCount):
    #        val     = random.random()
    #        desc1   = np.full((num, fetDim), val)
    #        desc2   = np.full((num, fetDim), val)

    #        matches = self.matcher.matchFeatures(desc1, desc2)

    #        for i, m in enumerate(matches):
    #            self.assertEqual(m.queryIdx, i)
    #            self.assertEqual(m.trainIdx, 0)
    #            self.assertAlmostEqual(m.distance, 0)

    def test_sequence(self):
        '''
        If all descriptors have dimension=1, and have increasing values, they
        should be matched to the same value. Like 0 to 0, 1 to 1, etc.
        '''
        num     = 100
        seq     = range(num)

        desc1   = np.transpose(np.array(seq)[np.newaxis])
        desc2   = np.transpose(np.array(seq[ :: -1])[np.newaxis])

        matches = self.matcher.matchFeatures(desc1, desc2)

        for i, m in enumerate(matches):
            self.assertEqual(m.queryIdx, i)
            self.assertEqual(m.trainIdx, num - i - 1)
            self.assertAlmostEqual(m.distance, 0)

    def test_multiplechoice(self):
        '''
        If all descriptors have dimension=1, and have increasing values,
        but the second descriptor array has x+0.1, and x+0.2 values, the
        first descriptors should be matched with the x+0.1 ones.
        '''
        num     = 100
        seq     = range(num)

        desc1   = np.transpose(np.array(seq)[np.newaxis])
        revarr  = np.array(seq[::-1])
        revarr  = np.concatenate((revarr + 0.2, revarr + 0.1))
        desc2   = np.transpose(revarr[np.newaxis])

        matches = self.matcher.matchFeatures(desc1, desc2)

        for i, m in enumerate(matches):
            self.assertEqual(m.queryIdx, i)
            self.assertEqual(m.trainIdx, 2 * num - i - 1)
            closenormal = np.allclose(m.distance, 0.1)
            closesquared = np.allclose(m.distance, 0.1**2)
            # it should compute either the squared or the normal distance
            self.assertTrue(closenormal or closesquared)

    def test_varyingdimensions(self, dim=64):
        '''
        If all the descriptors have dimension=dim, and have increasing values,
        but the second descriptor array has x+0.1, and x+0.2 values, the
        first descriptors should be matched first with the x+0.1
        '''
        num     = 100
        seq     = range(num)

        desc1 = [np.transpose(np.array(seq)[np.newaxis]) for i in range(dim)]
        desc1 = np.hstack(desc1)
        revarr = [np.transpose(np.array(seq[::-1])[np.newaxis]) for i in range(dim)]
        revarr = np.hstack(revarr)
        desc2  = np.concatenate((revarr + 0.2, revarr + 0.1))

        matches = self.matcher.matchFeatures(desc1, desc2)

        for i, m in enumerate(matches):
            self.assertEqual(m.queryIdx, i)
            self.assertEqual(m.trainIdx, 2 * num - i - 1)
            ssd = (0.1**2) * dim
            closenormal = np.allclose(m.distance, math.sqrt(ssd))
            closesquared = np.allclose(m.distance, ssd)
            # it should compute either the squared or the normal distance
            self.assertTrue(closenormal or closesquared)


class TestRatioFeatureMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = features.RatioFeatureMatcher()

    def tearDown(self):
        pass

    #def test_equals(self):
    #    '''
    #    If all descriptors are equal, the matches' matched index should be 0
    #    and all the distances should be approximately 0.
    #    '''
    #    num         = 10
    #    fetDim      = 10
    #    testCount   = 50

    #    for t in range(testCount):
    #        val     = random.random()
    #        desc1   = np.full((num, fetDim), val)
    #        desc2   = np.full((num, fetDim), val)

    #        matches = self.matcher.matchFeatures(desc1, desc2)

    #        for i, m in enumerate(matches):
    #            self.assertEqual(m.queryIdx, i)
    #            self.assertEqual(m.trainIdx, 0)
    #            # this is a pathological case, so we only check if it is nan
    #            self.assertFalse(np.isnan(m.distance))

    def test_sequence(self):
        '''
        If all descriptors have dimension=1, and have increasing values, they
        should be matched to the same value. Like 0 to 0, 1 to 1, etc.
        '''
        num = 100
        seq = range(num)

        desc1   = np.transpose(np.array(seq)[np.newaxis])
        desc2   = np.transpose(np.array(seq[::-1])[np.newaxis])

        matches = self.matcher.matchFeatures(desc1, desc2)

        for i, m in enumerate(matches):
            self.assertEqual(m.queryIdx, i)
            self.assertEqual(m.trainIdx, num-i-1)
            self.assertAlmostEqual(m.distance, 0)

    def test_multiplechoice(self):
        '''
        If all descriptors have dimension=1, and have increasing values,
        but the second descriptor array has x+0.1, and x+0.2 values, the
        first descriptors should be matched with the x+0.1 ones. (difficulty 1)
        '''
        num     = 100
        seq     = range(num)

        desc1   = np.transpose(np.array(seq)[np.newaxis])
        revarr  = np.array(seq[::-1])
        revarr  = np.concatenate((revarr + 0.2, revarr + 0.1))
        desc2   = np.transpose(revarr[np.newaxis])

        matches = self.matcher.matchFeatures(desc1, desc2)

        for i, m in enumerate(matches):
            self.assertEqual(m.queryIdx, i)
            self.assertEqual(m.trainIdx, 2 * num - i - 1)

            # it should compute either the squared or the normal distance
            closenormal = np.allclose(m.distance, 0.1 / 0.2)
            closesquared = np.allclose(m.distance, 0.1**2 / 0.2**2)
            self.assertTrue(closenormal or closesquared)

    def test_varyingdimensions(self, dim=64):
        '''
        If all the descriptors have dimension=dim, and have increasing values,
        but the second descriptor array has x+0.1, and x+0.2 values, the
        first descriptors should be matched first with the x+0.1 and second with
        x+0.2
        '''
        num     = 100
        seq     = range(num)

        desc1 = [np.transpose(np.array(seq)[np.newaxis]) for i in range(dim)]
        desc1 = np.hstack(desc1)
        revarr = [np.transpose(np.array(seq[::-1])[np.newaxis]) for i in range(dim)]
        revarr = np.hstack(revarr)
        desc2  = np.concatenate((revarr + 0.2, revarr + 0.1))

        matches = self.matcher.matchFeatures(desc1, desc2)

        for i, m in enumerate(matches):
            self.assertEqual(m.queryIdx, i)
            self.assertEqual(m.trainIdx, 2 * num - i - 1)
            ssd_first = (0.1**2) * dim
            ssd_second = (0.2**2) * dim
            closenormal = np.allclose(m.distance, math.sqrt(ssd_first / ssd_second))
            closesquared = np.allclose(m.distance, ssd_first / ssd_second)
            # it should compute either the squared or the normal distance
            self.assertTrue(closenormal or closesquared)


class TestBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(clz):
        """ Run the benchmark once (independent of thresholds) """

        # our solution gets these scores; it's nearly the same with euclidean
        # or squared euclidean in the ratio test
        clz.expected_auc = {
            'bikes': 0.3294,
            'graf': 0.759,
            'leuven': 0.863,
            'wall': 0.7927,
            'yosemite': 0.904,
        }

        clz.partial_thresholds = [0.70, 0.85, 0.98]

        clz.auc_results = {}
        for datasetname in ['bikes', 'graf', 'leuven', 'wall', 'yosemite']:
            detector        = features.HarrisKeypointDetector()
            descriptor      = features.MOPSFeatureDescriptor()
            matcher         = features.RatioFeatureMatcher()
            kpThreshold     = 1e-2
            matchThreshold  = 5
            directory       = os.path.join(TestBenchmark.datasetsrootpath, datasetname)

            ds, aucs, roc_img = benchmark.benchmark_dir(
                directory, detector, descriptor, matcher, kpThreshold, matchThreshold)

            auc_mean = np.mean(aucs)
            clz.auc_results[datasetname] = auc_mean

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def dataset_tester(self, datasetname, idx):
        self.assertGreaterEqual(
            self.auc_results[datasetname],
            self.expected_auc[datasetname] *
            self.partial_thresholds[idx - 1]
        )

    def test_bikes1(self):
        ''' Run benchmark on the bikes dataset (difficulty 1) '''
        self.dataset_tester('bikes', 1)

    def test_bikes2(self):
        ''' Run benchmark on the bikes dataset (difficulty 2) '''
        self.dataset_tester('bikes', 2)

    def test_bikes3(self):
        ''' Run benchmark on the bikes dataset (difficulty 3) '''
        self.dataset_tester('bikes', 3)

    def test_graf1(self):
        ''' Run benchmark on the graf dataset (difficulty 1) '''
        self.dataset_tester('graf', 1)

    def test_graf2(self):
        ''' Run benchmark on the graf dataset (difficulty 2) '''
        self.dataset_tester('graf', 2)

    def test_graf3(self):
        ''' Run benchmark on the graf dataset (difficulty 3) '''
        self.dataset_tester('graf', 3)

    def test_leuven1(self):
        ''' Run benchmark on the leuven dataset (difficulty 1) '''
        self.dataset_tester('leuven', 1)

    def test_leuven2(self):
        ''' Run benchmark on the leuven dataset (difficulty 2) '''
        self.dataset_tester('leuven', 2)

    def test_leuven3(self):
        ''' Run benchmark on the leuven dataset (difficulty 3) '''
        self.dataset_tester('leuven', 3)

    def test_wall1(self):
        ''' Run benchmark on the wall dataset (difficulty 1) '''
        self.dataset_tester('wall', 1)

    def test_wall2(self):
        ''' Run benchmark on the wall dataset (difficulty 2) '''
        self.dataset_tester('wall', 2)

    def test_wall3(self):
        ''' Run benchmark on the wall dataset (difficulty 3) '''
        self.dataset_tester('wall', 3)

    def test_yosemite1(self):
        ''' Run benchmark on the yosemite dataset (difficulty 1) '''
        self.dataset_tester('yosemite', 1)

    def test_yosemite2(self):
        ''' Run benchmark on the yosemite dataset (difficulty 2) '''
        self.dataset_tester('yosemite', 2)

    def test_yosemite3(self):
        ''' Run benchmark on the yosemite dataset (difficulty 3) '''
        self.dataset_tester('yosemite', 3)


TestBenchmark.datasetsrootpath        = '.'
TestMOPSFeatureDescriptor.testdatadir = 'testdata'


if __name__ == '__main__':
    np.random.seed(4670)
    unittest.main()

