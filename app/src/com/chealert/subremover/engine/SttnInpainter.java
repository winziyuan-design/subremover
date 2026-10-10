package com.chealert.subremover.engine;

import android.content.Context;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;

/**
 * Official STTN (MIT) at native resolution: the hole is covered by 432x240 tiles with overlap, not scaled down.
 * Input layout is fixed by {@code sttn/train/export_phone.py --t 8}: frames [8,3,240,432] RGB 0..1,
 * masks [8,1,240,432] 1=hole. NNAPI FP16 when the device has it, otherwise CPU.
 */
final class SttnInpainter {
    static final int T = 8, TW = 432, TH = 240, OV = 64, CTX = 32;
    private static final int PIX = TH * TW;

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private OrtSession session;
    private final float[] inF = new float[T * 3 * PIX], inM = new float[T * PIX], outF = new float[T * 3 * PIX];
    private final Mat dilK = Mat.ones(3, 3, CvType.CV_8U);
    long ns;
    int calls;

    static boolean assetExists(Context ctx) {
        try (InputStream in = ctx.getAssets().open("sttn.onnx")) {
            return in.read() >= 0;
        } catch (Exception e) {
            return false;
        }
    }

    SttnInpainter(Context ctx) throws Exception {
        File f = new File(ctx.getFilesDir(), "sttn.onnx");
        // Re-extract after every app update so new weights replace the cached copy.
        File stamp = new File(ctx.getFilesDir(), "sttn.onnx.stamp");
        String want = String.valueOf(ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0).lastUpdateTime);
        String have = null;
        if (stamp.isFile()) {
            try (java.io.FileInputStream si = new java.io.FileInputStream(stamp)) {
                byte[] sb = new byte[64];
                int sn = si.read(sb);
                have = sn > 0 ? new String(sb, 0, sn, "UTF-8") : null;
            }
        }
        if (!f.exists() || f.length() == 0 || !want.equals(have)) {
            File tmp = new File(ctx.getFilesDir(), "sttn.onnx.part");
            try (InputStream in = ctx.getAssets().open("sttn.onnx"); OutputStream out = new FileOutputStream(tmp)) {
                byte[] b = new byte[1 << 20];
                int n;
                while ((n = in.read(b)) > 0) out.write(b, 0, n);
            }
            f.delete();
            tmp.renameTo(f);
            try (OutputStream so = new FileOutputStream(stamp)) { so.write(want.getBytes("UTF-8")); }
        }
        try {
            session = open(f, true);
        } catch (Throwable e) {
            Log.w(SubtitleRemover.TAG, "nnapi session failed, CPU", e);
            session = open(f, false);
        }
    }

    private OrtSession open(File f, boolean nnapi) throws Exception {
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(4);
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if (nnapi) o.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
        try {
            o.addXnnpack(Collections.emptyMap());
        } catch (Throwable ignored) { }
        return env.createSession(f.getAbsolutePath(), o);
    }

    /**
     * Writes the inpainted target frame back into {@code frames[target]} on {@code holes[target]}.
     * The other frames are reference only. All mats share one size; holes are CV_8U, 255 = hole.
     */
    void inpaint(Mat[] frames, Mat[] holes, int target) throws Exception {
        if (frames.length != T || holes.length != T) throw new IllegalArgumentException("need " + T + " frames");
        long t0 = System.nanoTime();
        int H = frames[0].rows(), W = frames[0].cols();
        Rect bb = Imgproc.boundingRect(holes[target]);
        if (bb.width <= 0 || bb.height <= 0) return;
        int x0 = Math.max(0, bb.x - CTX), y0 = Math.max(0, bb.y - CTX);
        int x1 = Math.min(W, bb.x + bb.width + CTX), y1 = Math.min(H, bb.y + bb.height + CTX);
        int[] xs = starts(x0, x1, W, TW, OV), ys = starts(y0, y1, H, TH, OV);
        float[] num = new float[H * W * 3], den = new float[H * W];
        byte[] hb = new byte[TH * TW];
        for (int y : ys) {
            for (int x : xs) {
                Mat ht = tile(holes[target], x, y, true);
                ht.get(0, 0, hb);
                int nz = 0;
                for (byte v : hb) if (v != 0) nz++;
                ht.release();
                if (nz == 0) continue;
                Mat[] crops = new Mat[T], hm = new Mat[T];
                try {
                    for (int i = 0; i < T; i++) {
                        crops[i] = tile(frames[i], x, y, false);
                        hm[i] = tile(holes[i], x, y, true);
                        Imgproc.dilate(hm[i], hm[i], dilK);
                    }
                    runTile(crops, hm);
                } finally {
                    for (int i = 0; i < T; i++) {
                        if (crops[i] != null) crops[i].release();
                        if (hm[i] != null) hm[i].release();
                    }
                }
                float[] w = ramp(x, y, W, H);
                int yLo = Math.max(0, y), yHi = Math.min(H, y + TH);
                int xLo = Math.max(0, x), xHi = Math.min(W, x + TW);
                for (int yy = yLo; yy < yHi; yy++) {
                    int ty = yy - y;
                    for (int xx = xLo; xx < xHi; xx++) {
                        float ww = w[ty * TW + (xx - x)];
                        if (ww <= 0) continue;
                        int dst = yy * W + xx;
                        int src = (target * 3) * PIX + ty * TW + (xx - x);
                        den[dst] += ww;
                        num[dst * 3] += outF[src + 2 * PIX] * ww;       // B
                        num[dst * 3 + 1] += outF[src + PIX] * ww;       // G
                        num[dst * 3 + 2] += outF[src] * ww;             // R
                    }
                }
                calls++;
            }
        }
        byte[] pix = new byte[H * W * 3], hole = new byte[H * W];
        frames[target].get(0, 0, pix);
        holes[target].get(0, 0, hole);
        for (int i = 0; i < H * W; i++) {
            if (hole[i] == 0 || den[i] < 1e-3f) continue;
            int p = i * 3;
            pix[p] = sat(num[p] / den[i]);
            pix[p + 1] = sat(num[p + 1] / den[i]);
            pix[p + 2] = sat(num[p + 2] / den[i]);
        }
        frames[target].put(0, 0, pix);
        ns += System.nanoTime() - t0;
    }

    /** One STTN call. Overwrites {@code inF} with RGB 0..1 output in frames-major NCHW. */
    private void runTile(Mat[] bgr, Mat[] mask) throws Exception {
        float[] hwc = new float[3 * PIX];
        byte[] mb = new byte[PIX];
        Mat rgb = new Mat(), fl = new Mat();
        try {
            for (int f = 0; f < T; f++) {
                Imgproc.cvtColor(bgr[f], rgb, Imgproc.COLOR_BGR2RGB);
                rgb.convertTo(fl, CvType.CV_32FC3, 1.0 / 255);
                fl.get(0, 0, hwc);
                int base = f * 3 * PIX;
                for (int i = 0, p = 0; i < PIX; i++, p += 3) {
                    inF[base + i] = hwc[p];
                    inF[base + PIX + i] = hwc[p + 1];
                    inF[base + 2 * PIX + i] = hwc[p + 2];
                }
                mask[f].get(0, 0, mb);
                int mb0 = f * PIX;
                for (int i = 0; i < PIX; i++) inM[mb0 + i] = mb[i] != 0 ? 1f : 0f;
            }
        } finally {
            rgb.release();
            fl.release();
        }
        Map<String, OnnxTensor> in = new HashMap<>();
        try (OnnxTensor ti = OnnxTensor.createTensor(env, FloatBuffer.wrap(inF), new long[]{T, 3, TH, TW});
             OnnxTensor tm = OnnxTensor.createTensor(env, FloatBuffer.wrap(inM), new long[]{T, 1, TH, TW})) {
            in.put("frames", ti);
            in.put("masks", tm);
            try (OrtSession.Result r = session.run(in)) {
                ((OnnxTensor) r.get(0)).getFloatBuffer().get(outF);
            }
        }
    }

    /** 432x240 view of a tile. Image borders are replicated; mask borders stay empty. */
    private static Mat tile(Mat src, int x, int y, boolean mask) {
        int H = src.rows(), W = src.cols();
        int x0 = Math.max(0, x), y0 = Math.max(0, y);
        int x1 = Math.min(W, x + TW), y1 = Math.min(H, y + TH);
        Mat sub = src.submat(y0, y1, x0, x1);
        int top = y0 - y, left = x0 - x, bottom = y + TH - y1, right = x + TW - x1;
        if (top == 0 && bottom == 0 && left == 0 && right == 0) {
            Mat c = sub.clone();
            sub.release();
            return c;
        }
        Mat dst = new Mat();
        Core.copyMakeBorder(sub, dst, top, bottom, left, right,
                mask ? Core.BORDER_CONSTANT : Core.BORDER_REPLICATE, new Scalar(0));
        sub.release();
        return dst;
    }

    private static int[] starts(int lo, int hi, int n, int tile, int ov) {
        if (n <= tile) return new int[]{0};
        int span = Math.max(1, hi - lo);
        if (span <= tile) {
            int c = lo + span / 2 - tile / 2;
            return new int[]{Math.max(0, Math.min(c, n - tile))};
        }
        int k = (int) Math.ceil((span - ov) / (double) (tile - ov));
        if (k < 2) return new int[]{Math.max(0, Math.min(lo, n - tile))};
        int[] s = new int[k];
        for (int i = 0; i < k; i++) {
            int v = (int) Math.round(lo + i * (span - tile) / (double) (k - 1));
            s[i] = Math.max(0, Math.min(v, n - tile));
        }
        return s;
    }

    private static float[] ramp(int x, int y, int W, int H) {
        float[] wx = new float[TW], wy = new float[TH];
        for (int i = 0; i < TW; i++) wx[i] = 1f;
        for (int i = 0; i < TH; i++) wy[i] = 1f;
        for (int i = 0; i < OV; i++) {
            float r = (i + 1f) / (OV + 1f);
            if (x > 0) wx[i] = r;
            if (x + TW < W) wx[TW - 1 - i] = r;
            if (i < TH && y > 0) wy[i] = r;
            if (i < TH && y + TH < H) wy[TH - 1 - i] = r;
        }
        float[] w = new float[TH * TW];
        for (int yy = 0; yy < TH; yy++)
            for (int xx = 0; xx < TW; xx++) w[yy * TW + xx] = wy[yy] * wx[xx];
        return w;
    }

    private static byte sat(float v) {
        int i = Math.round(v * 255f);
        return (byte) (i < 0 ? 0 : i > 255 ? 255 : i);
    }

    void close() {
        dilK.release();
        try { session.close(); } catch (Exception ignored) { }
    }
}
