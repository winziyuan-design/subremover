package com.chealert.subremover.engine;

import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Where subtitles live: a horizontal band (y0..y1) plus the rule for which boxes inside it count as
 * subtitle lines. Auto mode: lines in the lower half, horizontally centred, single-line height,
 * recurring at the same height across sampled frames. Manual mode: anything centred inside the user box.
 */
public final class SubtitleBand {
    public int y0, y1;            // band rows in display coordinates
    public float lineH;           // typical subtitle line height (px)
    public final int frameW, frameH;
    private final int[] manual;   // x0,y0,x1,y1 in display coords, or null

    private SubtitleBand(int w, int h, int[] manual) { frameW = w; frameH = h; this.manual = manual; }

    public static boolean isLineCandidate(Rect r, int w, int h) {
        double cx = r.x + r.width / 2.0, cy = r.y + r.height / 2.0;
        return cy > h * 0.5 && Math.abs(cx - w / 2.0) < w * 0.12
                && r.height > h * 0.02 && r.height < h * 0.14 && r.width > r.height * 1.5;
    }

    private static boolean inManual(Rect r, int[] m) {
        double cx = r.x + r.width / 2.0, cy = r.y + r.height / 2.0;
        return cx >= m[0] && cx <= m[2] && cy >= m[1] && cy <= m[3];
    }

    /** @param samples full-frame detections from evenly spaced frames. Returns null if nothing found (auto). */
    public static SubtitleBand fromSamples(List<List<Rect>> samples, int w, int h, int[] manual) {
        List<Rect> cand = new ArrayList<>();
        for (List<Rect> l : samples)
            for (Rect r : l)
                if (manual != null ? inManual(r, manual) && r.width > r.height : isLineCandidate(r, w, h)) cand.add(r);
        SubtitleBand b = new SubtitleBand(w, h, manual);
        if (cand.isEmpty()) {
            if (manual == null) return null;
            b.lineH = Math.max(12, (manual[3] - manual[1]) * 0.6f);
            b.y0 = Math.max(0, manual[1]);
            b.y1 = Math.min(h, manual[3]);
            b.even();
            return b;
        }
        double[] cys = new double[cand.size()], lhs = new double[cand.size()];
        for (int i = 0; i < cand.size(); i++) {
            cys[i] = cand.get(i).y + cand.get(i).height / 2.0;
            lhs[i] = cand.get(i).height;
        }
        double cy = median(cys), lh = median(lhs);
        int lo = Integer.MAX_VALUE, hi = 0;
        for (Rect r : cand) {
            double c = r.y + r.height / 2.0;
            if (manual != null || Math.abs(c - cy) < lh * 1.6) {
                lo = Math.min(lo, r.y);
                hi = Math.max(hi, r.y + r.height);
            }
        }
        b.lineH = (float) lh;
        b.y0 = Math.max(0, (int) (lo - lh * 0.6));
        b.y1 = Math.min(h, (int) (hi + lh * 0.6));
        if (manual != null) { b.y0 = Math.min(b.y0, Math.max(0, manual[1])); b.y1 = Math.max(b.y1, Math.min(h, manual[3])); }
        b.even();
        return b;
    }

    private void even() {
        y0 &= ~1;
        y1 = Math.min(frameH, (y1 + 1) & ~1);
        if (y1 - y0 < 16) y1 = Math.min(frameH, y0 + 16);
    }

    /** Filter detections made on a band crop (band-local coordinates). */
    public List<Rect> keep(List<Rect> local) {
        List<Rect> out = new ArrayList<>();
        for (Rect r : local) {
            if (r.width <= r.height) continue;
            if (manual != null) {
                Rect g = new Rect(r.x, r.y + y0, r.width, r.height);
                if (inManual(g, manual)) out.add(r);
            } else {
                double cx = r.x + r.width / 2.0;
                if (Math.abs(cx - frameW / 2.0) < frameW * 0.2 && r.height < lineH * 1.8) out.add(r);
            }
        }
        return out;
    }

    private static double median(double[] a) {
        double[] c = a.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }
}
