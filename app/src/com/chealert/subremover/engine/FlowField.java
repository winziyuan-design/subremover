package com.chealert.subremover.engine;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.DISOpticalFlow;

import java.util.ArrayList;
import java.util.List;

/**
 * Dense flow between consecutive band frames at reduced resolution. Flow inside subtitle holes is completed
 * from surrounding valid flow (normalised convolution); a forward-backward check marks occlusions.
 */
final class FlowField {
    static final float S = 0.5f;
    final Size small, full;
    private final DISOpticalFlow dis = DISOpticalFlow.create(DISOpticalFlow.PRESET_FAST);
    private final Mat gridS, gridF, holeK;
    private final double sigma;
    long ns;

    FlowField(int bw, int bh, float lineH) {
        full = new Size(bw, bh);
        small = new Size(Math.max(8, Math.round(bw * S)), Math.max(8, Math.round(bh * S)));
        gridS = grid((int) small.width, (int) small.height);
        gridF = grid(bw, bh);
        holeK = Mat.ones(5, 5, CvType.CV_8U);
        sigma = Math.max(4.0, lineH * S * 0.8);
    }

    private static Mat grid(int w, int h) {
        Mat g = new Mat(h, w, CvType.CV_32FC2);
        float[] row = new float[w * 2];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) { row[2 * x] = x; row[2 * x + 1] = y; }
            g.put(y, 0, row);
        }
        return g;
    }

    Mat smallGray(Mat band) {
        Mat g = new Mat(), s = new Mat();
        Imgproc.cvtColor(band, g, Imgproc.COLOR_BGR2GRAY);
        Imgproc.resize(g, s, small, 0, 0, Imgproc.INTER_AREA);
        g.release();
        return s;
    }

    Mat smallMask(Mat mask) {
        Mat s = new Mat();
        Imgproc.resize(mask, s, small, 0, 0, Imgproc.INTER_NEAREST);
        return s;
    }

    /** Flow for pixels of a pointing into b, completed in holes; null means "static" (zero flow). */
    Mat flow(Mat ga, Mat gb, Mat ma, Mat mb) {
        long t0 = System.nanoTime();
        Mat f = new Mat();
        dis.calc(ga, gb, f);
        Mat hole = new Mat(), validU8 = new Mat(), valid = new Mat(), den = new Mat();
        Core.bitwise_or(ma, mb, hole);
        Imgproc.dilate(hole, hole, holeK);
        Core.bitwise_not(hole, validU8);
        validU8.convertTo(valid, CvType.CV_32F, 1.0 / 255);
        Imgproc.GaussianBlur(valid, den, new Size(0, 0), sigma);
        Core.max(den, new Scalar(1e-4), den);
        List<Mat> ch = new ArrayList<>(2);
        Core.split(f, ch);
        for (int c = 0; c < 2; c++) {
            Mat x = ch.get(c), num = new Mat(), fill = new Mat();
            Core.multiply(x, valid, num);
            Imgproc.GaussianBlur(num, num, new Size(0, 0), sigma);
            Core.divide(num, den, fill);
            x.copyTo(fill, validU8);
            x.release(); num.release();
            ch.set(c, fill);
        }
        Core.merge(ch, f);
        Mat mag = new Mat();
        Core.magnitude(ch.get(0), ch.get(1), mag);
        double mean = Core.mean(mag).val[0], max = Core.minMaxLoc(mag).maxVal;
        for (Mat m : ch) m.release();
        mag.release(); hole.release(); validU8.release(); valid.release(); den.release();
        ns += System.nanoTime() - t0;
        if (mean < 0.15 && max < 0.6) { f.release(); return null; }   // static: no resampling blur
        return f;
    }

    /** 255 where a→b→a returns close to the start (full band size). Either flow may be null (= zero). */
    Mat consistent(Mat fab, Mat fba) {
        Mat ok = new Mat();
        if (fab == null && fba == null) return new Mat((int) full.height, (int) full.width, CvType.CV_8U, new Scalar(255));
        Mat a = fab != null ? fab : Mat.zeros(small, CvType.CV_32FC2);
        Mat b = fba != null ? fba : Mat.zeros(small, CvType.CV_32FC2);
        Mat map = new Mat(), back = new Mat(), sum = new Mat(), err = new Mat(), magA = new Mat(), thr = new Mat();
        Core.add(gridS, a, map);
        Imgproc.remap(b, back, map, new Mat(), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(99, 99));
        Core.add(a, back, sum);
        List<Mat> c = new ArrayList<>(2);
        Core.split(sum, c);
        Core.magnitude(c.get(0), c.get(1), err);
        for (Mat m : c) m.release();
        Core.split(a, c);
        Core.magnitude(c.get(0), c.get(1), magA);
        for (Mat m : c) m.release();
        Core.multiply(magA, new Scalar(0.05), thr);
        Core.add(thr, new Scalar(0.5), thr);
        Mat okS = new Mat();
        Core.compare(err, thr, okS, Core.CMP_LT);
        Imgproc.resize(okS, ok, full, 0, 0, Imgproc.INTER_NEAREST);
        if (fab == null) a.release();
        if (fba == null) b.release();
        map.release(); back.release(); sum.release(); err.release(); magA.release(); thr.release(); okS.release();
        return ok;
    }

    /** Full-resolution remap field from small flow. */
    Mat fullMap(Mat f) {
        Mat fu = new Mat();
        Imgproc.resize(f, fu, full, 0, 0, Imgproc.INTER_LINEAR);
        Core.multiply(fu, new Scalar(1 / S, 1 / S), fu);
        Core.add(gridF, fu, fu);
        return fu;
    }
}
