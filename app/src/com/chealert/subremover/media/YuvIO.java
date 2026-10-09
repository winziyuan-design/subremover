package com.chealert.subremover.media;

import android.graphics.Rect;
import android.media.Image;

import java.nio.ByteBuffer;

/** Copies between YUV_420_888 Images (any stride layout) and packed I420 byte arrays. */
final class YuvIO {
    private YuvIO() { }

    static void read(Image img, byte[] out, int w, int h) {
        Rect cr = img.getCropRect();
        Image.Plane[] p = img.getPlanes();
        int pos = 0;
        byte[] row = null;
        for (int pi = 0; pi < 3; pi++) {
            int sh = pi == 0 ? 0 : 1;
            int pw = w >> sh, ph = h >> sh;
            ByteBuffer buf = p[pi].getBuffer();
            int rs = p[pi].getRowStride(), ps = p[pi].getPixelStride();
            int x0 = cr.left >> sh, y0 = cr.top >> sh;
            if (ps == 1) {
                for (int r = 0; r < ph; r++) {
                    buf.position((y0 + r) * rs + x0);
                    buf.get(out, pos, pw);
                    pos += pw;
                }
            } else {
                int need = (pw - 1) * ps + 1;
                if (row == null || row.length < need) row = new byte[need];
                for (int r = 0; r < ph; r++) {
                    buf.position((y0 + r) * rs + x0 * ps);
                    buf.get(row, 0, need);
                    for (int c = 0; c < pw; c++) out[pos++] = row[c * ps];
                }
            }
        }
    }

    static void write(Image img, byte[] in, int w, int h) {
        Image.Plane[] p = img.getPlanes();
        int pos = 0;
        byte[] row = null;
        for (int pi = 0; pi < 3; pi++) {
            int sh = pi == 0 ? 0 : 1;
            int pw = w >> sh, ph = h >> sh;
            ByteBuffer buf = p[pi].getBuffer();
            int rs = p[pi].getRowStride(), ps = p[pi].getPixelStride();
            if (ps == 1) {
                for (int r = 0; r < ph; r++) {
                    buf.position(r * rs);
                    buf.put(in, pos, pw);
                    pos += pw;
                }
            } else {
                for (int r = 0; r < ph; r++) {
                    int base = r * rs;
                    for (int c = 0; c < pw; c++) buf.put(base + c * ps, in[pos++]);
                }
            }
        }
    }
}
