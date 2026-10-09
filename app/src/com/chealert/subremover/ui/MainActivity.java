package com.chealert.subremover.ui;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.content.res.ColorStateList;

import com.chealert.subremover.engine.SubtitleRemover;

import java.util.Locale;

public final class MainActivity extends Activity {
    private static final int REQ_PICK = 1;
    private static final int BG = 0xFF0A0C10, SURFACE = 0xFF141820, MINT = 0xFF3DDC97, WHITE = 0xFFFFFFFF;

    private float dp;
    private BoxView preview;
    private Button pick, start, clearBox, play, share, cancel;
    private ProgressBar bar;
    private TextView status;
    private LinearLayout resultRow;
    private Uri source, result;
    private SubtitleRemover running;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        dp = getResources().getDisplayMetrics().density;
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        int p = (int) (20 * dp);
        root.setPadding(p, (int) (28 * dp), p, p);

        TextView title = new TextView(this);
        title.setText("去字幕");
        title.setTextColor(WHITE);
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title, new LinearLayout.LayoutParams(-1, -2));

        pick = button("选择视频", true);
        root.addView(pick, lp(16));
        pick.setOnClickListener(v -> pickVideo());

        preview = new BoxView(this);
        GradientDrawable card = new GradientDrawable();
        card.setColor(SURFACE);
        card.setCornerRadius(16 * dp);
        preview.setBackground(card);
        LinearLayout.LayoutParams plp = new LinearLayout.LayoutParams(-1, 0, 1f);
        plp.topMargin = (int) (16 * dp);
        root.addView(preview, plp);
        preview.setVisibility(View.INVISIBLE);
        preview.setOnBoxChanged(box -> clearBox.setVisibility(box != null && running == null ? View.VISIBLE : View.GONE));

        clearBox = button("清除框选", false);
        root.addView(clearBox, lp(12));
        clearBox.setVisibility(View.GONE);
        clearBox.setOnClickListener(v -> preview.clearBox());

        bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        bar.setProgressTintList(ColorStateList.valueOf(MINT));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(SURFACE));
        root.addView(bar, lp(16));
        bar.setVisibility(View.GONE);

        status = new TextView(this);
        status.setTextColor(WHITE);
        status.setTextSize(15);
        status.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(status, lp(8));

        start = button("开始去字幕", true);
        root.addView(start, lp(16));
        start.setVisibility(View.GONE);
        start.setOnClickListener(v -> startProcessing());

        cancel = button("取消", false);
        root.addView(cancel, lp(16));
        cancel.setVisibility(View.GONE);
        cancel.setOnClickListener(v -> { if (running != null) running.cancel(); });

        resultRow = new LinearLayout(this);
        resultRow.setOrientation(LinearLayout.HORIZONTAL);
        play = button("播放", true);
        share = button("分享", false);
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, (int) (52 * dp), 1f);
        resultRow.addView(play, half);
        LinearLayout.LayoutParams half2 = new LinearLayout.LayoutParams(0, (int) (52 * dp), 1f);
        half2.leftMargin = (int) (12 * dp);
        resultRow.addView(share, half2);
        root.addView(resultRow, lp(16));
        resultRow.setVisibility(View.GONE);
        play.setOnClickListener(v -> {
            if (result == null) return;
            Intent i = new Intent(Intent.ACTION_VIEW).setDataAndType(result, "video/mp4")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(i);
        });
        share.setOnClickListener(v -> {
            if (result == null) return;
            Intent i = new Intent(Intent.ACTION_SEND).setType("video/mp4").putExtra(Intent.EXTRA_STREAM, result)
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            startActivity(Intent.createChooser(i, "分享"));
        });
        setContentView(root);
    }

    private LinearLayout.LayoutParams lp(int topDp) {
        LinearLayout.LayoutParams l = new LinearLayout.LayoutParams(-1, (int) (52 * dp));
        l.topMargin = (int) (topDp * dp);
        return l;
    }

    private Button button(String text, boolean primary) {
        Button bt = new Button(this);
        bt.setText(text);
        bt.setAllCaps(false);
        bt.setTextSize(16);
        bt.setTypeface(Typeface.DEFAULT_BOLD);
        bt.setTextColor(primary ? BG : WHITE);
        bt.setStateListAnimator(null);
        GradientDrawable d = new GradientDrawable();
        d.setColor(primary ? MINT : SURFACE);
        d.setCornerRadius(14 * dp);
        bt.setBackground(d);
        return bt;
    }

    private void pickVideo() {
        Intent i = new Intent(Intent.ACTION_PICK);
        i.setDataAndType(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, "video/*");
        try {
            startActivityForResult(i, REQ_PICK);
        } catch (Exception e) {
            Intent g = new Intent(Intent.ACTION_GET_CONTENT).setType("video/*").addCategory(Intent.CATEGORY_OPENABLE);
            startActivityForResult(g, REQ_PICK);
        }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req != REQ_PICK || res != RESULT_OK || data == null || data.getData() == null) return;
        source = data.getData();
        result = null;
        resultRow.setVisibility(View.GONE);
        bar.setVisibility(View.GONE);
        status.setText("");
        clearBox.setVisibility(View.GONE);
        start.setVisibility(View.VISIBLE);
        preview.setVisibility(View.VISIBLE);
        preview.setBitmap(null);
        final Uri u = source;
        new Thread(() -> {
            Bitmap bm = null;
            MediaMetadataRetriever m = new MediaMetadataRetriever();
            try {
                m.setDataSource(this, u);
                bm = m.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
            } catch (Exception e) {
                Log.w(SubtitleRemover.TAG, "preview failed", e);
            } finally {
                try { m.release(); } catch (Exception ignored) { }
            }
            final Bitmap f = bm;
            runOnUiThread(() -> { if (u.equals(source)) preview.setBitmap(f); });
        }).start();
    }

    private void startProcessing() {
        if (source == null || running != null) return;
        final Uri u = source;
        final RectF box = preview.getBox();
        final SubtitleRemover sr = new SubtitleRemover();
        running = sr;
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        pick.setEnabled(false);
        pick.setAlpha(0.4f);
        start.setVisibility(View.GONE);
        clearBox.setVisibility(View.GONE);
        resultRow.setVisibility(View.GONE);
        cancel.setVisibility(View.VISIBLE);
        preview.setDrawEnabled(false);
        bar.setVisibility(View.VISIBLE);
        bar.setProgress(0);
        status.setText("找字幕位置");
        final long t0 = System.currentTimeMillis();
        final int[] stage = {0};
        new Thread(() -> {
            String msg;
            Uri out = null;
            try {
                SubtitleRemover.Result r = sr.run(this, u, box, new SubtitleRemover.Listener() {
                    @Override public void onStage(int s) {
                        stage[0] = s;
                        runOnUiThread(() -> status.setText(s == 0 ? "找字幕位置" : s == 1 ? "去字幕中" : "保存中"));
                    }
                    @Override public void onProgress(int done, int total) {
                        int pr = stage[0] == 0 ? done * 50 / Math.max(1, total)
                                : stage[0] == 1 ? 50 + (int) (950L * done / Math.max(1, total)) : 1000;
                        int pct = pr / 10;
                        runOnUiThread(() -> {
                            bar.setProgress(pr);
                            if (stage[0] == 1) status.setText(String.format(Locale.CHINA, "去字幕中 %d%%", pct));
                        });
                    }
                });
                out = r.uri;
                long s = r.ms / 1000;
                msg = String.format(Locale.CHINA, "完成 · 用时 %d分%02d秒", s / 60, s % 60);
            } catch (SubtitleRemover.NoSubtitleException e) {
                msg = "没找到字幕，在画面上框出字幕位置再试";
            } catch (InterruptedException e) {
                msg = "已取消";
            } catch (Throwable e) {
                Log.w(SubtitleRemover.TAG, "failed", e);
                msg = "处理失败";
            }
            final String m = msg;
            final Uri o = out;
            runOnUiThread(() -> finishUi(m, o));
        }).start();
    }

    private void finishUi(String msg, Uri out) {
        running = null;
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        pick.setEnabled(true);
        pick.setAlpha(1f);
        cancel.setVisibility(View.GONE);
        preview.setDrawEnabled(true);
        status.setText(msg);
        result = out;
        if (out != null) {
            bar.setProgress(1000);
            resultRow.setVisibility(View.VISIBLE);
        } else {
            bar.setVisibility(View.GONE);
            start.setVisibility(View.VISIBLE);
            if (preview.getBox() != null) clearBox.setVisibility(View.VISIBLE);
        }
    }

    @Override
    public void onBackPressed() {
        if (running != null) { moveTaskToBack(true); return; }
        super.onBackPressed();
    }
}
