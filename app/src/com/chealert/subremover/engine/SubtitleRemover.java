package com.chealert.subremover.engine;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.util.Log;

import com.chealert.subremover.media.FrameReader;
import com.chealert.subremover.media.GallerySaver;
import com.chealert.subremover.media.Mp4Writer;
import com.chealert.subremover.media.VideoInfo;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.photo.Photo;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.Locale;

/** Pipeline: locate subtitle band → stream decode (look-ahead + output) → erase → encode → gallery. */
public final class SubtitleRemover {
    private long fillNs;

    public static final String TAG = "SubRm";

    public interface Listener {
        void onStage(int stage);           // 0 = locating, 1 = erasing, 2 = saving
        void onProgress(int done, int total);
    }

    public static final class NoSubtitleException extends Exception { }

    public static final class Result {
        public Uri uri;
        public long ms;
        public int frames, width, height;
    }

    private volatile boolean cancelled;

    public void cancel() { cancelled = true; }

    static {
        System.loadLibrary("opencv_java4");
    }

    public static String modelPath(Context ctx) throws Exception {
        File f = new File(ctx.getFilesDir(), "ppocr_det.onnx");
        if (!f.exists() || f.length() == 0) {
            try (InputStream in = ctx.getAssets().open("ppocr_det.onnx"); OutputStream out = new FileOutputStream(f)) {
                byte[] b = new byte[1 << 16];
                int n;
                while ((n = in.read(b)) > 0) out.write(b, 0, n);
            }
        }
        return f.getAbsolutePath();
    }

    /** @param box optional user-drawn subtitle area, normalised 0..1 in display orientation. */
    public Result run(Context ctx, Uri src, RectF box, Listener l) throws Exception {
        long T0 = System.currentTimeMillis();
        VideoInfo info = VideoInfo.probe(ctx, src);
        Log.w(TAG, "video " + info.width + "x" + info.height + " rot=" + info.rotation + " disp=" + info.dispW + "x"
                + info.dispH + " fps=" + info.fps + " dur=" + info.durationUs / 1000 + "ms audio=" + (info.audioTrack >= 0));
        SubtitleDetector det = new SubtitleDetector(modelPath(ctx));
        l.onStage(0);

        // ---- stage 0: sample frames to find the subtitle band ----
        long t0 = System.currentTimeMillis();
        int[] manual = null;
        if (box != null) {
            manual = new int[]{Math.round(box.left * info.dispW), Math.round(box.top * info.dispH),
                    Math.round(box.right * info.dispW), Math.round(box.bottom * info.dispH)};
        }
        List<List<Rect>> samples = new ArrayList<>();
        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        try {
            mmr.setDataSource(ctx, src);
            int n = 10;
            for (int i = 0; i < n && !cancelled; i++) {
                long tu = (long) (info.durationUs * (i + 0.5) / n);
                Bitmap bm = mmr.getFrameAtTime(tu, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                if (bm == null) continue;
                Mat f = toDisplayBgr(bm, info);
                bm.recycle();
                int top = manual != null ? Math.max(0, Math.min(manual[1], info.dispH - 32) - 16) : info.dispH * 2 / 5;
                Mat lower = f.submat(top, info.dispH, 0, info.dispW);
                List<Rect> rs = det.detect(lower, 960);
                for (Rect r : rs) r.y += top;
                samples.add(rs);
                f.release();
                l.onProgress(i + 1, n);
            }
        } finally {
            mmr.release();
        }
        SubtitleBand band = SubtitleBand.fromSamples(samples, info.dispW, info.dispH, manual);
        long tPrescan = System.currentTimeMillis() - t0;
        if (band == null) {
            Log.w(TAG, "no subtitle line found in " + samples.size() + " samples (" + tPrescan + "ms)");
            throw new NoSubtitleException();
        }
        Log.w(TAG, "band y=" + band.y0 + ".." + band.y1 + " lineH=" + band.lineH + " prescan=" + tPrescan + "ms");

        // ---- pass 1: decode once, keep band crops (JPEG) + raw stroke candidates, detect every K ----
        l.onStage(1);
        long t1 = System.currentTimeMillis();
        int est = info.estimatedFrames();
        int bw = band.frameW, bh = band.y1 - band.y0;
        List<byte[]> bandJpg = new ArrayList<>(est + 16), raws = new ArrayList<>(est + 16);
        Map<Integer, List<Rect>> dets = new HashMap<>();
        List<Boolean> cut = new ArrayList<>(est + 16);
        FlowField ff = new FlowField(bw, bh, band.lineH);
        long decodeNs = 0;
        FrameReader reader = new FrameReader(ctx, src, info);
        MatOfInt jpgQ = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 95);
        Mat prevTiny = null;
        try {
            while (!cancelled) {
                long d0 = System.nanoTime();
                Mat f = reader.next();
                decodeNs += System.nanoTime() - d0;
                if (f == null) break;
                int i = bandJpg.size();
                Mat b = f.submat(band.y0, band.y1, 0, bw);
                MatOfByte enc = new MatOfByte();
                Imgcodecs.imencode(".jpg", b, enc, jpgQ);
                bandJpg.add(enc.toArray());
                enc.release();
                raws.add(StrokeMasker.rawPng(b, band.lineH));
                if (i % StrokeMasker.K == 0) dets.put(i, band.keep(det.detect(b, 640)));
                Mat g = ff.smallGray(b), tiny = new Mat();
                Imgproc.resize(g, tiny, new Size(64, Math.max(4, 64 * bh / bw)), 0, 0, Imgproc.INTER_AREA);
                tiny.convertTo(tiny, CvType.CV_32F);
                g.release();
                boolean c = false;
                if (prevTiny != null) {
                    Mat d = new Mat();
                    Core.absdiff(tiny, prevTiny, d);
                    c = Core.mean(d).val[0] > 30;
                    d.release();
                    prevTiny.release();
                }
                prevTiny = tiny;
                cut.add(c);
                b.release();
                if (i % 10 == 0) l.onProgress(Math.min(300, 300 * (i + 1) / Math.max(est, i + 1)), 1000);
            }
        } finally {
            reader.release();
            if (prevTiny != null) prevTiny.release();
        }
        if (cancelled) throw new InterruptedException();
        final int n = bandJpg.size();
        long tPass1 = System.currentTimeMillis() - t1;
        StrokeMasker sm = new StrokeMasker(bw, bh, band.lineH, raws, dets, n);

        // ---- pass 2: backward plate (latest-in-future clean background for each hole pixel) ----
        long t2a = System.currentTimeMillis();
        List<byte[]> bwdP = new ArrayList<>(n), bwdA = new ArrayList<>(n);
        int[] maskPx = new int[n];
        for (int i = 0; i < n; i++) { bwdP.add(null); bwdA.add(null); }
        Plate plate = new Plate(bw, bh);
        Mat nxtGray = null, nxtSm = null;
        MatOfInt png = new MatOfInt(Imgcodecs.IMWRITE_PNG_COMPRESSION, 1);
        for (int i = n - 1; i >= 0 && !cancelled; i--) {
            Mat b = Imgcodecs.imdecode(new MatOfByte(bandJpg.get(i)), Imgcodecs.IMREAD_COLOR);
            Mat m = sm.mask(i);
            maskPx[i] = Core.countNonZero(m);
            Mat g = ff.smallGray(b), sms = ff.smallMask(m);
            if (nxtGray != null) {
                if (cut.get(i + 1)) plate.reset();
                else {
                    Mat fab = ff.flow(g, nxtGray, sms, nxtSm), fba = ff.flow(nxtGray, g, nxtSm, sms);
                    Mat ok = ff.consistent(fab, fba);
                    Mat map = fab == null ? null : ff.fullMap(fab);
                    plate.warp(map, ok);
                    if (map != null) map.release();
                    if (fab != null) fab.release();
                    if (fba != null) fba.release();
                    ok.release();
                }
                nxtGray.release(); nxtSm.release();
            }
            plate.update(b, m);
            if (maskPx[i] > 0) {
                Mat pm = Mat.zeros(bh, bw, CvType.CV_8UC3), am = new Mat(bh, bw, CvType.CV_8U, new Scalar(255));
                Mat sel = new Mat();
                Core.bitwise_and(m, plate.V, sel);
                plate.P.copyTo(pm, sel);
                plate.A.copyTo(am, sel);
                MatOfByte e1 = new MatOfByte(), e2 = new MatOfByte();
                Imgcodecs.imencode(".png", pm, e1, png);
                Imgcodecs.imencode(".png", am, e2, png);
                bwdP.set(i, e1.toArray());
                bwdA.set(i, e2.toArray());
                pm.release(); am.release(); sel.release(); e1.release(); e2.release();
            }
            nxtGray = g; nxtSm = sms;
            b.release(); m.release();
            if (i % 10 == 0) l.onProgress(300 + 250 * (n - i) / n, 1000);
        }
        if (nxtGray != null) { nxtGray.release(); nxtSm.release(); }
        plate.release();
        if (cancelled) throw new InterruptedException();
        long tPass2 = System.currentTimeMillis() - t2a;

        // ---- pass 3: forward plate + per-line LaMa anchor + Telea, composite, encode ----
        long t3 = System.currentTimeMillis();
        File out = new File(ctx.getCacheDir(), "out_" + System.currentTimeMillis() + ".mp4");
        FrameReader behind = new FrameReader(ctx, src, info);
        Mp4Writer writer = new Mp4Writer(ctx, src, info, out);
        Plate fwd = new Plate(bw, bh), hal = new Plate(bw, bh);
        LamaInpainter lama = null;
        boolean lamaFailed = false;
        long pxF = 0, pxB = 0, pxH = 0, pxT = 0;
        int lastLama = -1000, t = 0, segments = 1;
        Mat prevGray = null, prevSm = null, prevStroke = null;
        boolean ok3 = false;
        try {
            for (t = 0; t < n && !cancelled; t++) {
                long d0 = System.nanoTime();
                Mat full = behind.next();
                decodeNs += System.nanoTime() - d0;
                if (full == null) break;
                Mat T = full.submat(band.y0, band.y1, 0, bw);
                Mat m = sm.mask(t);
                Mat g = ff.smallGray(T), sms = ff.smallMask(m);
                boolean segChange = false;
                if (prevGray != null) {
                    segChange = lineChanged(prevStroke, m);
                    if (segChange) segments++;
                    if (cut.get(t)) { fwd.reset(); hal.reset(); }
                    else {
                        Mat fab = ff.flow(g, prevGray, sms, prevSm), fba = ff.flow(prevGray, g, prevSm, sms);
                        Mat ok = ff.consistent(fab, fba);
                        Mat map = fab == null ? null : ff.fullMap(fab);
                        fwd.warp(map, ok);
                        if (segChange) hal.reset(); else hal.warp(map, ok);
                        if (map != null) map.release();
                        if (fab != null) fab.release();
                        if (fba != null) fba.release();
                        ok.release();
                    }
                    prevGray.release(); prevSm.release(); prevStroke.release();
                }
                fwd.update(T, m);
                int total = maskPx[t];
                if (total > 0) {
                    long f0 = System.nanoTime();
                    Mat R = T.clone(), rem = m.clone(), use = new Mat(), tmp = new Mat(), tmp2 = new Mat();
                    Mat bP = Imgcodecs.imdecode(new MatOfByte(bwdP.get(t)), Imgcodecs.IMREAD_COLOR);
                    Mat bA = Imgcodecs.imdecode(new MatOfByte(bwdA.get(t)), Imgcodecs.IMREAD_GRAYSCALE);
                    Mat bV = new Mat();
                    Core.compare(bA, new Scalar(255), bV, Core.CMP_LT);
                    // nearest-in-time real background: forward plate unless backward one is fresher
                    Core.compare(fwd.A, bA, tmp, Core.CMP_LE);
                    Core.bitwise_not(bV, tmp2);
                    Core.bitwise_or(tmp, tmp2, tmp);
                    Core.bitwise_and(tmp, fwd.V, tmp);
                    Core.bitwise_and(tmp, rem, use);
                    pxF += take(fwd.P, R, rem, use);
                    Core.bitwise_and(rem, bV, use);
                    pxB += take(bP, R, rem, use);
                    Core.bitwise_and(rem, hal.V, use);
                    pxH += take(hal.P, R, rem, use);
                    int left = Core.countNonZero(rem);
                    if (left > Math.max(60, total * 0.03) && !lamaFailed
                            && (t - lastLama >= 15 || segChange || Core.countNonZero(hal.V) == 0)) {
                        try {
                            if (lama == null) lama = new LamaInpainter(ctx);
                            Mat comp = full.clone();
                            R.copyTo(comp.submat(band.y0, band.y1, 0, bw));
                            Mat hole = Mat.zeros(full.size(), CvType.CV_8U);
                            rem.copyTo(hole.submat(band.y0, band.y1, 0, bw));
                            lama.inpaint(comp, hole);
                            Mat fb = comp.submat(band.y0, band.y1, 0, bw);
                            fb.copyTo(R, rem);
                            fb.copyTo(hal.P, rem);
                            hal.V.setTo(new Scalar(255), rem);
                            hal.A.setTo(new Scalar(0), rem);
                            pxH += left;
                            rem.setTo(new Scalar(0));
                            left = 0;
                            lastLama = t;
                            comp.release(); hole.release();
                        } catch (Throwable e) {
                            Log.w(TAG, "lama unavailable, Telea only", e);
                            lamaFailed = true;
                        }
                    }
                    if (left > 0) {
                        Mat R2 = new Mat();
                        Photo.inpaint(R, rem, R2, 3, Photo.INPAINT_TELEA);
                        R.release();
                        R = R2;
                        pxT += left;
                    }
                    Mat soft = new Mat(), inv = new Mat(), o = new Mat();
                    m.convertTo(soft, CvType.CV_32F, 1.0 / 255);
                    Imgproc.GaussianBlur(soft, soft, new Size(5, 5), 0);
                    Core.subtract(Mat.ones(soft.size(), CvType.CV_32F), soft, inv);
                    Imgproc.blendLinear(R, T, soft, inv, o);
                    o.copyTo(T);
                    R.release(); rem.release(); use.release(); tmp.release(); tmp2.release();
                    bP.release(); bA.release(); bV.release(); soft.release(); inv.release(); o.release();
                    fillNs += System.nanoTime() - f0;
                }
                writer.write(full, behind.ptsUs);
                prevGray = g; prevSm = sms; prevStroke = m;
                bwdP.set(t, null); bwdA.set(t, null);
                if (t % 5 == 0) l.onProgress(550 + 450 * (t + 1) / n, 1000);
            }
            if (prevGray != null) { prevGray.release(); prevSm.release(); prevStroke.release(); }
            if (cancelled) throw new InterruptedException();
            l.onStage(2);
            writer.finish();
            ok3 = true;
        } finally {
            behind.release();
            fwd.release(); hal.release();
            if (lama != null) lama.close();
            if (!ok3) { writer.abort(); out.delete(); }
        }
        long tPass3 = System.currentTimeMillis() - t3;
        Log.w(TAG, String.format(Locale.US,
                "frames=%d pass1(decode+detect+masks)=%dms pass2(backward)=%dms pass3(forward+encode)=%dms decode=%dms detect=%dms(%d) flow=%dms fill=%dms lama=%dms(%d calls) encode=%dms segments=%d",
                t, tPass1, tPass2, tPass3, decodeNs / 1000000, det.ns / 1000000, det.calls, ff.ns / 1000000,
                fillNs / 1000000, lama == null ? 0 : lama.ns / 1000000, lama == null ? 0 : lama.calls,
                writer.encodeNs / 1000000, segments));
        Log.w(TAG, "pixels fwdPlate=" + pxF + " bwdPlate=" + pxB + " lama=" + pxH + " telea=" + pxT);

        long t2 = System.currentTimeMillis();
        String name = "去字幕_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date()) + ".mp4";
        Result r = new Result();
        r.uri = GallerySaver.save(ctx, out, name);
        out.delete();
        r.ms = System.currentTimeMillis() - T0;
        r.frames = n;
        r.width = info.dispW;
        r.height = info.dispH;
        Log.w(TAG, "saved " + name + " save=" + (System.currentTimeMillis() - t2) + "ms total=" + r.ms + "ms");
        return r;
    }

    /** Copies src→dst where use, clears those pixels in rem; returns count. */
    private static long take(Mat src, Mat dst, Mat rem, Mat use) {
        int k = Core.countNonZero(use);
        if (k == 0) return 0;
        src.copyTo(dst, use);
        rem.setTo(new Scalar(0), use);
        return k;
    }

    /** Subtitle line changed if the stroke masks of consecutive frames overlap less than half. */
    private static boolean lineChanged(Mat a, Mat b) {
        Mat x = new Mat();
        Core.bitwise_or(a, b, x);
        int u = Core.countNonZero(x);
        Core.bitwise_and(a, b, x);
        int in = Core.countNonZero(x);
        x.release();
        return u > 0 && in < 0.5 * u;
    }

    private static Mat toDisplayBgr(Bitmap bm, VideoInfo info) {
        Mat rgba = new Mat(), bgr = new Mat();
        Utils.bitmapToMat(bm, rgba);
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR);
        rgba.release();
        boolean portraitDisp = info.dispH > info.dispW, portraitBm = bgr.rows() > bgr.cols();
        if (info.rotation != 0 && info.dispW != info.dispH && portraitDisp != portraitBm) {
            Mat r = new Mat();
            Core.rotate(bgr, r, info.rotation == 90 ? Core.ROTATE_90_CLOCKWISE
                    : info.rotation == 180 ? Core.ROTATE_180 : Core.ROTATE_90_COUNTERCLOCKWISE);
            bgr.release();
            bgr = r;
        }
        if (bgr.cols() != info.dispW || bgr.rows() != info.dispH) {
            Mat r = new Mat();
            Imgproc.resize(bgr, r, new Size(info.dispW, info.dispH));
            bgr.release();
            bgr = r;
        }
        return bgr;
    }
}
