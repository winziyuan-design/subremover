package com.chealert.subremover.media;

import android.content.Context;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

/** Sequential hardware decode to display-oriented BGR Mats. */
public final class FrameReader {
    private final MediaExtractor ex = new MediaExtractor();
    private final MediaCodec dec;
    private final MediaCodec.BufferInfo bi = new MediaCodec.BufferInfo();
    private final VideoInfo info;
    private boolean inEos, outEos;
    private byte[] i420;
    private Mat yuv, bgr;
    public long ptsUs;

    public FrameReader(Context ctx, Uri uri, VideoInfo info) throws Exception {
        this.info = info;
        ex.setDataSource(ctx, uri, null);
        ex.selectTrack(info.videoTrack);
        MediaFormat f = ex.getTrackFormat(info.videoTrack);
        f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
        dec = MediaCodec.createDecoderByType(f.getString(MediaFormat.KEY_MIME));
        dec.configure(f, null, null, 0);
        dec.start();
    }

    /** Next frame as BGR in display orientation (shared Mat, valid until next call), or null at end. */
    public Mat next() {
        while (!outEos) {
            if (!inEos) {
                int ii = dec.dequeueInputBuffer(5000);
                if (ii >= 0) {
                    ByteBuffer b = dec.getInputBuffer(ii);
                    int n = ex.readSampleData(b, 0);
                    if (n < 0) {
                        dec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        inEos = true;
                    } else {
                        dec.queueInputBuffer(ii, 0, n, ex.getSampleTime(), 0);
                        ex.advance();
                    }
                }
            }
            int oi = dec.dequeueOutputBuffer(bi, 5000);
            if (oi < 0) continue;
            if ((bi.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outEos = true;
            if (bi.size <= 0) { dec.releaseOutputBuffer(oi, false); continue; }
            Image img = dec.getOutputImage(oi);
            if (img == null) { dec.releaseOutputBuffer(oi, false); continue; }
            int w = img.getCropRect().width() & ~1, h = img.getCropRect().height() & ~1;
            int need = w * h * 3 / 2;
            if (i420 == null || i420.length != need) {
                i420 = new byte[need];
                if (yuv != null) yuv.release();
                yuv = new Mat(h * 3 / 2, w, CvType.CV_8UC1);
            }
            YuvIO.read(img, i420, w, h);
            img.close();
            ptsUs = bi.presentationTimeUs;
            dec.releaseOutputBuffer(oi, false);
            yuv.put(0, 0, i420);
            if (bgr == null) bgr = new Mat();
            Imgproc.cvtColor(yuv, bgr, Imgproc.COLOR_YUV2BGR_I420);
            Mat out = bgr;
            if (info.rotation != 0) {
                Mat r = new Mat();
                int code = info.rotation == 90 ? Core.ROTATE_90_CLOCKWISE
                        : info.rotation == 180 ? Core.ROTATE_180 : Core.ROTATE_90_COUNTERCLOCKWISE;
                Core.rotate(bgr, r, code);
                bgr.release();
                bgr = r;
                out = r;
            }
            if (out.cols() != info.dispW || out.rows() != info.dispH) {
                Mat c = out.submat(0, Math.min(info.dispH, out.rows()), 0, Math.min(info.dispW, out.cols())).clone();
                if (c.cols() != info.dispW || c.rows() != info.dispH) {
                    Mat z = new Mat(info.dispH, info.dispW, CvType.CV_8UC3, new org.opencv.core.Scalar(0, 0, 0));
                    c.copyTo(z.submat(0, c.rows(), 0, c.cols()));
                    c.release();
                    c = z;
                }
                bgr.release();
                bgr = c;
                out = c;
            }
            return out;
        }
        return null;
    }

    public void release() {
        try { dec.stop(); } catch (Exception ignored) { }
        try { dec.release(); } catch (Exception ignored) { }
        try { ex.release(); } catch (Exception ignored) { }
        if (yuv != null) yuv.release();
        if (bgr != null) bgr.release();
    }
}
