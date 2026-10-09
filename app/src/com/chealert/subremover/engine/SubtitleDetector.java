package com.chealert.subremover.engine;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.TextDetectionModel_DB;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/** PP-OCRv3 mobile text detector (DB head) run through OpenCV DNN, fully offline. */
public final class SubtitleDetector {
    private static final Scalar MEAN = new Scalar(122.67891434, 116.66876762, 104.00698793);
    private final TextDetectionModel_DB model;
    public long ns;
    public int calls;

    public SubtitleDetector(String onnxPath) {
        model = new TextDetectionModel_DB(onnxPath);
        model.setBinaryThreshold(0.3f);
        model.setPolygonThreshold(0.5f);
        model.setMaxCandidates(200);
        model.setUnclipRatio(2.0);
    }

    /** Axis-aligned text boxes in the coordinates of {@code bgr}. */
    public List<Rect> detect(Mat bgr, int maxW) {
        long t0 = System.nanoTime();
        int w = bgr.cols(), h = bgr.rows();
        double s = Math.min(1.0, maxW / (double) w);
        int iw = Math.max(32, (int) Math.round(w * s / 32.0) * 32);
        int ih = Math.max(32, (int) Math.round(h * s / 32.0) * 32);
        model.setInputParams(1.0 / 255, new Size(iw, ih), MEAN);
        List<MatOfPoint> polys = new ArrayList<>();
        model.detect(bgr, polys);
        List<Rect> out = new ArrayList<>(polys.size());
        for (MatOfPoint p : polys) {
            out.add(Imgproc.boundingRect(p));
            p.release();
        }
        ns += System.nanoTime() - t0;
        calls++;
        return out;
    }
}
