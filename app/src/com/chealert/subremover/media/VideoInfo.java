package com.chealert.subremover.media;

import android.content.Context;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

/** Track layout and geometry of a source video. */
public final class VideoInfo {
    public int videoTrack = -1, audioTrack = -1;
    public MediaFormat videoFormat, audioFormat;
    public int width, height;        // coded (decoder) size
    public int rotation;             // 0/90/180/270
    public int dispW, dispH;         // after rotation, even
    public float fps = 30f;
    public long durationUs;
    public int bitrate;

    public static VideoInfo probe(Context ctx, Uri uri) throws Exception {
        VideoInfo v = new VideoInfo();
        MediaExtractor ex = new MediaExtractor();
        try {
            ex.setDataSource(ctx, uri, null);
            for (int i = 0; i < ex.getTrackCount(); i++) {
                MediaFormat f = ex.getTrackFormat(i);
                String mime = f.getString(MediaFormat.KEY_MIME);
                if (mime == null) continue;
                if (mime.startsWith("video/") && v.videoTrack < 0) { v.videoTrack = i; v.videoFormat = f; }
                else if (mime.startsWith("audio/") && v.audioTrack < 0) { v.audioTrack = i; v.audioFormat = f; }
            }
        } finally {
            ex.release();
        }
        if (v.videoTrack < 0) throw new IllegalStateException("no video track");
        MediaFormat f = v.videoFormat;
        v.width = f.getInteger(MediaFormat.KEY_WIDTH);
        v.height = f.getInteger(MediaFormat.KEY_HEIGHT);
        if (f.containsKey(MediaFormat.KEY_FRAME_RATE)) {
            try { v.fps = f.getInteger(MediaFormat.KEY_FRAME_RATE); }
            catch (ClassCastException e) { v.fps = f.getFloat(MediaFormat.KEY_FRAME_RATE); }
        }
        if (v.fps <= 1 || v.fps > 240) v.fps = 30f;
        if (f.containsKey(MediaFormat.KEY_DURATION)) v.durationUs = f.getLong(MediaFormat.KEY_DURATION);
        if (f.containsKey("rotation-degrees")) v.rotation = f.getInteger("rotation-degrees");
        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        try {
            mmr.setDataSource(ctx, uri);
            String r = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
            if (r != null && v.rotation == 0) v.rotation = Integer.parseInt(r);
            String b = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE);
            if (b != null) v.bitrate = Integer.parseInt(b);
            if (v.durationUs <= 0) {
                String d = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                if (d != null) v.durationUs = Long.parseLong(d) * 1000L;
            }
        } catch (Exception ignored) {
        } finally {
            try { mmr.release(); } catch (Exception ignored) { }
        }
        v.rotation = ((v.rotation % 360) + 360) % 360;
        boolean swap = v.rotation == 90 || v.rotation == 270;
        v.dispW = (swap ? v.height : v.width) & ~1;
        v.dispH = (swap ? v.width : v.height) & ~1;
        return v;
    }

    public int estimatedFrames() {
        return Math.max(1, Math.round(durationUs / 1e6f * fps));
    }
}
