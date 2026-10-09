package com.chealert.subremover.engine;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stroke-shaped subtitle masks. Inside the detected subtitle boxes, keep pixels that are bright, unsaturated,
 * thin (white top-hat) and that stay put across neighbouring frames while the background moves; then dilate
 * a little to cover the outline / shadow.
 */
final class StrokeMasker {
    static final int K = 6;   // detector stride (frames)
    private final int bw, bh, pad, n;
    private final Mat hatK, dilK, closeK;
    private final List<byte[]> raws;          // PNG of per-frame raw stroke candidates
    private final Map<Integer, List<Rect>> dets;
    private final Map<Integer, Mat> rawCache = new LinkedHashMap<Integer, Mat>(16, 0.75f, true) {
        @Override protected boolean removeEldestEntry(Map.Entry<Integer, Mat> e) {
            if (size() > 12) { e.getValue().release(); return true; }
            return false;
        }
    };

    StrokeMasker(int bw, int bh, float lineH, List<byte[]> raws, Map<Integer, List<Rect>> dets, int n) {
        this.bw = bw; this.bh = bh; this.raws = raws; this.dets = dets; this.n = n;
        pad = (int) (lineH * 0.15f) + 2;
        int hk = ((int) (lineH * 0.5f)) | 1;
        hatK = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(hk, hk));
        int dk = Math.max(3, Math.round(lineH * 0.12f));
        dilK = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2 * dk + 1, 2 * dk + 1));
        closeK = Mat.ones(3, 3, CvType.CV_8U);
    }

    /** Raw candidates for one band (computed once in the decode pass). */
    static byte[] rawPng(Mat band, float lineH) {
        int hk = ((int) (lineH * 0.5f)) | 1;
        Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(hk, hk));
        Mat hsv = new Mat();
        Imgproc.cvtColor(band, hsv, Imgproc.COLOR_BGR2HSV);
        List<Mat> ch = new ArrayList<>(3);
        Core.split(hsv, ch);
        Mat th = new Mat(), a = new Mat(), b = new Mat(), c = new Mat();
        Imgproc.morphologyEx(ch.get(2), th, Imgproc.MORPH_TOPHAT, k);
        Imgproc.threshold(th, a, 35, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(ch.get(2), b, 170, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(ch.get(1), c, 79, 255, Imgproc.THRESH_BINARY_INV);
        Core.bitwise_and(a, b, a);
        Core.bitwise_and(a, c, a);
        MatOfByte out = new MatOfByte();
        Imgcodecs.imencode(".png", a, out);
        byte[] r = out.toArray();
        for (Mat m : ch) m.release();
        hsv.release(); th.release(); a.release(); b.release(); c.release(); out.release(); k.release();
        return r;
    }

    private Mat raw(int i) {
        i = Math.max(0, Math.min(n - 1, i));
        Mat m = rawCache.get(i);
        if (m == null) {
            m = Imgcodecs.imdecode(new MatOfByte(raws.get(i)), Imgcodecs.IMREAD_GRAYSCALE);
            rawCache.put(i, m);
        }
        return m;
    }

    Mat region(int i) {
        Mat m = Mat.zeros(bh, bw, CvType.CV_8U);
        int a = i / K * K;
        for (int j = a - K; j <= a + K; j += K) {
            List<Rect> rs = dets.get(j);
            if (rs == null) continue;
            for (Rect r : rs)
                Imgproc.rectangle(m, new Point(r.x - pad, r.y - pad), new Point(r.x + r.width + pad, r.y + r.height + pad), new Scalar(255), -1);
        }
        return m;
    }

    /** Final stroke mask for frame i (new Mat). */
    Mat mask(int i) {
        Mat reg = region(i);
        if (Core.countNonZero(reg) == 0) return reg;
        Mat r0 = raw(i), a = new Mat(), b = new Mat();
        Core.bitwise_and(r0, raw(i - 2), a);
        Core.bitwise_and(r0, raw(i + 2), b);
        Core.bitwise_or(a, b, a);
        Imgproc.morphologyEx(a, a, Imgproc.MORPH_CLOSE, closeK);
        Imgproc.dilate(a, a, dilK);
        Imgproc.dilate(reg, reg, dilK);
        Core.bitwise_and(a, reg, a);
        reg.release(); b.release();
        return a;
    }
}
