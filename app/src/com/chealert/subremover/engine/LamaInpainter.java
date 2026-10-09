package com.chealert.subremover.engine;

import android.content.Context;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/** LaMa (big-lama, Apache-2.0, Carve/LaMa-ONNX export, int8 weights) on a 512x512 crop via ONNX Runtime. */
final class LamaInpainter {
    private static final int N = 512;
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final float[] img = new float[3 * N * N], msk = new float[N * N];
    long ns;
    int calls;

    LamaInpainter(Context ctx) throws Exception {
        File f = new File(ctx.getFilesDir(), "lama_w8.onnx");
        if (!f.exists() || f.length() == 0) {
            File tmp = new File(ctx.getFilesDir(), "lama_w8.onnx.part");
            try (InputStream in = ctx.getAssets().open("lama_w8.onnx"); OutputStream out = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 20];
                int n;
                while ((n = in.read(b)) > 0) out.write(b, 0, n);
            }
            tmp.renameTo(f);
        }
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(4);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        session = env.createSession(f.getAbsolutePath(), o);
    }

    /**
     * Fills {@code hole} (CV_8U, frame size) in {@code frame} (BGR) in place, using a square context crop around
     * the hole scaled to 512x512.
     */
    void inpaint(Mat frame, Mat hole) throws Exception {
        long t0 = System.nanoTime();
        Rect bb = Imgproc.boundingRect(hole);
        int W = frame.cols(), H = frame.rows();
        int side = (int) Math.max(N, Math.max(bb.width * 1.3, bb.height * 3.0));
        int cw = Math.min(side, W), ch = Math.min(side, H);
        int cx = bb.x + bb.width / 2, cy = bb.y + bb.height / 2;
        int lx = Math.max(0, Math.min(W - cw, cx - cw / 2)), ly = Math.max(0, Math.min(H - ch, cy - ch / 2));
        Rect cr = new Rect(lx, ly, cw, ch);
        Mat crop = frame.submat(cr), hm = hole.submat(cr);
        Mat rs = new Mat(), rgb = new Mat(), fl = new Mat(), mr = new Mat(), md = new Mat();
        Imgproc.resize(crop, rs, new Size(N, N), 0, 0, Imgproc.INTER_AREA);
        Imgproc.cvtColor(rs, rgb, Imgproc.COLOR_BGR2RGB);
        rgb.convertTo(fl, CvType.CV_32FC3, 1.0 / 255);
        float[] hwc = new float[3 * N * N];
        fl.get(0, 0, hwc);
        for (int i = 0, p = 0; i < N * N; i++, p += 3) {
            img[i] = hwc[p]; img[N * N + i] = hwc[p + 1]; img[2 * N * N + i] = hwc[p + 2];
        }
        Imgproc.dilate(hm, md, Mat.ones(5, 5, CvType.CV_8U));
        Imgproc.resize(md, mr, new Size(N, N), 0, 0, Imgproc.INTER_LINEAR);
        byte[] mb = new byte[N * N];
        mr.get(0, 0, mb);
        for (int i = 0; i < N * N; i++) msk[i] = mb[i] != 0 ? 1f : 0f;
        Map<String, OnnxTensor> in = new HashMap<>();
        try (OnnxTensor ti = OnnxTensor.createTensor(env, FloatBuffer.wrap(img), new long[]{1, 3, N, N});
             OnnxTensor tm = OnnxTensor.createTensor(env, FloatBuffer.wrap(msk), new long[]{1, 1, N, N})) {
            in.put("image", ti);
            in.put("mask", tm);
            try (OrtSession.Result r = session.run(in)) {
                FloatBuffer ob = ((OnnxTensor) r.get(0)).getFloatBuffer();
                float[] chw = new float[3 * N * N];
                ob.get(chw);
                for (int i = 0, p = 0; i < N * N; i++, p += 3) {
                    hwc[p] = chw[2 * N * N + i]; hwc[p + 1] = chw[N * N + i]; hwc[p + 2] = chw[i];   // -> BGR
                }
            }
        }
        Mat out = new Mat(N, N, CvType.CV_32FC3), o8 = new Mat(), up = new Mat();
        out.put(0, 0, hwc);
        out.convertTo(o8, CvType.CV_8UC3);   // saturating 0..255
        Imgproc.resize(o8, up, new Size(cw, ch), 0, 0, Imgproc.INTER_CUBIC);
        up.copyTo(crop, hm);
        rs.release(); rgb.release(); fl.release(); mr.release(); md.release(); out.release(); o8.release(); up.release();
        ns += System.nanoTime() - t0;
        calls++;
    }

    void close() {
        try { session.close(); } catch (Exception ignored) { }
    }
}
