package com.chealert.subremover.ui;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

/** Shows the first frame; a drag draws an optional subtitle box (normalised to the frame). */
public final class BoxView extends View {
    public interface OnBoxChanged { void onBox(RectF box); }

    private Bitmap bmp;
    private final RectF dst = new RectF();
    private RectF box;              // normalised
    private float sx, sy;
    private boolean enabledDraw = true;
    private OnBoxChanged cb;
    private final Paint img = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint stroke = new Paint(Paint.ANTI_ALIAS_FLAG), fill = new Paint(Paint.ANTI_ALIAS_FLAG);

    public BoxView(Context c) {
        super(c);
        float d = c.getResources().getDisplayMetrics().density;
        stroke.setStyle(Paint.Style.STROKE);
        stroke.setStrokeWidth(2 * d);
        stroke.setColor(0xFF3DDC97);
        fill.setColor(0x333DDC97);
    }

    public void setOnBoxChanged(OnBoxChanged c) { cb = c; }
    public void setBitmap(Bitmap b) { bmp = b; box = null; invalidate(); }
    public void setDrawEnabled(boolean e) { enabledDraw = e; }
    public RectF getBox() { return box; }
    public void clearBox() { box = null; invalidate(); if (cb != null) cb.onBox(null); }

    @Override
    protected void onDraw(Canvas c) {
        if (bmp == null) return;
        float vw = getWidth(), vh = getHeight();
        float s = Math.min(vw / bmp.getWidth(), vh / bmp.getHeight());
        float w = bmp.getWidth() * s, h = bmp.getHeight() * s;
        dst.set((vw - w) / 2, (vh - h) / 2, (vw + w) / 2, (vh + h) / 2);
        c.drawBitmap(bmp, null, dst, img);
        if (box != null) {
            RectF r = new RectF(dst.left + box.left * w, dst.top + box.top * h, dst.left + box.right * w, dst.top + box.bottom * h);
            c.drawRect(r, fill);
            c.drawRect(r, stroke);
        }
    }

    private float nx(float x) { return Math.max(0, Math.min(1, (x - dst.left) / dst.width())); }
    private float ny(float y) { return Math.max(0, Math.min(1, (y - dst.top) / dst.height())); }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (bmp == null || !enabledDraw || dst.width() <= 0) return false;
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                sx = nx(e.getX()); sy = ny(e.getY());
                getParent().requestDisallowInterceptTouchEvent(true);
                return true;
            case MotionEvent.ACTION_MOVE:
            case MotionEvent.ACTION_UP:
                float x = nx(e.getX()), y = ny(e.getY());
                RectF r = new RectF(Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y));
                if (r.width() > 0.03f && r.height() > 0.01f) box = r;
                invalidate();
                if (e.getActionMasked() == MotionEvent.ACTION_UP && cb != null) cb.onBox(box);
                return true;
        }
        return super.onTouchEvent(e);
    }
}
