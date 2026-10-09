package com.chealert.subremover.media;

import android.content.Context;
import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.net.Uri;

import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.ByteBuffer;

/** H.264 hardware encode of BGR frames + untouched copy of the source audio track. */
public final class Mp4Writer {
    private final MediaCodec enc;
    private final MediaMuxer mux;
    private final MediaCodec.BufferInfo bi = new MediaCodec.BufferInfo();
    private final Context ctx;
    private final Uri src;
    private final VideoInfo info;
    private final int w, h;
    private int vTrack = -1, aTrack = -1;
    private boolean started;
    private final Mat yuv = new Mat();
    private byte[] i420;
    public long encodeNs;

    public Mp4Writer(Context ctx, Uri src, VideoInfo info, File out) throws Exception {
        this.ctx = ctx;
        this.src = src;
        this.info = info;
        this.w = info.dispW;
        this.h = info.dispH;
        int br = info.bitrate > 0 ? info.bitrate : (int) (w * h * info.fps * 0.2f);
        br = Math.max(2_000_000, Math.min(br, 20_000_000));
        MediaFormat f = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, w, h);
        f.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
        f.setInteger(MediaFormat.KEY_BIT_RATE, br);
        f.setInteger(MediaFormat.KEY_FRAME_RATE, Math.round(info.fps));
        f.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1);
        enc = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
        enc.configure(f, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
        enc.start();
        mux = new MediaMuxer(out.getAbsolutePath(), MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
    }

    public void write(Mat bgr, long ptsUs) {
        long t0 = System.nanoTime();
        Imgproc.cvtColor(bgr, yuv, Imgproc.COLOR_BGR2YUV_I420);
        int need = w * h * 3 / 2;
        if (i420 == null || i420.length != need) i420 = new byte[need];
        yuv.get(0, 0, i420);
        while (true) {
            int ii = enc.dequeueInputBuffer(5000);
            if (ii >= 0) {
                Image img = enc.getInputImage(ii);
                YuvIO.write(img, i420, w, h);
                enc.queueInputBuffer(ii, 0, need, ptsUs, 0);
                break;
            }
            drain(false);
        }
        drain(false);
        encodeNs += System.nanoTime() - t0;
    }

    private void drain(boolean eos) {
        while (true) {
            int oi = enc.dequeueOutputBuffer(bi, eos ? 10000 : 0);
            if (oi == MediaCodec.INFO_TRY_AGAIN_LATER) { if (!eos) return; else continue; }
            if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                vTrack = mux.addTrack(enc.getOutputFormat());
                if (info.audioTrack >= 0) {
                    try { aTrack = mux.addTrack(info.audioFormat); } catch (Exception e) { aTrack = -1; }
                }
                mux.start();
                started = true;
                continue;
            }
            if (oi < 0) continue;
            ByteBuffer b = enc.getOutputBuffer(oi);
            if ((bi.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) bi.size = 0;
            if (bi.size > 0 && started) {
                b.position(bi.offset);
                b.limit(bi.offset + bi.size);
                mux.writeSampleData(vTrack, b, bi);
            }
            enc.releaseOutputBuffer(oi, false);
            if ((bi.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) return;
        }
    }

    public void finish() throws Exception {
        long t0 = System.nanoTime();
        while (true) {
            int ii = enc.dequeueInputBuffer(10000);
            if (ii >= 0) { enc.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); break; }
            drain(false);
        }
        drain(true);
        encodeNs += System.nanoTime() - t0;
        if (aTrack >= 0 && started) copyAudio();
        enc.stop();
        enc.release();
        mux.stop();
        mux.release();
        yuv.release();
    }

    private void copyAudio() throws Exception {
        MediaExtractor ex = new MediaExtractor();
        try {
            ex.setDataSource(ctx, src, null);
            ex.selectTrack(info.audioTrack);
            ByteBuffer buf = ByteBuffer.allocateDirect(1 << 20);
            MediaCodec.BufferInfo ai = new MediaCodec.BufferInfo();
            while (true) {
                int n = ex.readSampleData(buf, 0);
                if (n < 0) break;
                ai.set(0, n, ex.getSampleTime(),
                        (ex.getSampleFlags() & MediaExtractor.SAMPLE_FLAG_SYNC) != 0 ? MediaCodec.BUFFER_FLAG_KEY_FRAME : 0);
                mux.writeSampleData(aTrack, buf, ai);
                ex.advance();
            }
        } finally {
            ex.release();
        }
    }

    public void abort() {
        try { enc.stop(); } catch (Exception ignored) { }
        try { enc.release(); } catch (Exception ignored) { }
        try { if (started) mux.stop(); } catch (Exception ignored) { }
        try { mux.release(); } catch (Exception ignored) { }
    }
}
