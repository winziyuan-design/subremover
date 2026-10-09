package com.chealert.subremover.engine;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Background plate carried through time: last-seen clean value per pixel (P), how many frames ago it was
 * seen (A, 0..255) and whether it is valid (V). Warped frame to frame with the band flow.
 */
final class Plate {
    final Mat P, A, V;
    private final Mat tmp = new Mat();

    Plate(int bw, int bh) {
        P = Mat.zeros(bh, bw, CvType.CV_8UC3);
        A = new Mat(bh, bw, CvType.CV_8U, new Scalar(255));
        V = Mat.zeros(bh, bw, CvType.CV_8U);
    }

    void reset() { A.setTo(new Scalar(255)); V.setTo(new Scalar(0)); }

    /** Move plate into the next frame's coordinates. map == null means static. ok = consistency mask. */
    void warp(Mat map, Mat ok) {
        if (map != null) {
            Imgproc.remap(P, tmp, map, new Mat(), Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, new Scalar(0));
            tmp.copyTo(P);
            Imgproc.remap(A, tmp, map, new Mat(), Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(255));
            tmp.copyTo(A);
            Imgproc.remap(V, tmp, map, new Mat(), Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));
            tmp.copyTo(V);
        }
        Core.add(A, new Scalar(1), A);
        Core.bitwise_and(V, ok, V);
    }

    /** Pixels not covered by a subtitle in this frame are real background. */
    void update(Mat band, Mat mask) {
        Core.bitwise_not(mask, tmp);
        band.copyTo(P, tmp);
        A.setTo(new Scalar(0), tmp);
        V.setTo(new Scalar(255), tmp);
    }

    void release() { P.release(); A.release(); V.release(); tmp.release(); }
}
