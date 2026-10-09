package com.chealert.subremover.debug;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.Uri;
import android.os.IBinder;
import android.util.Log;

import com.chealert.subremover.engine.SubtitleRemover;

import java.io.File;

/**
 * Shell-only benchmark entry (guarded by the DUMP permission): runs the full pipeline on a file that was
 * pushed into this app's own external files dir, without touching the UI.
 */
public final class BenchService extends Service {
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel("bench", "去字幕", NotificationManager.IMPORTANCE_LOW));
        Notification n = new Notification.Builder(this, "bench").setContentTitle("去字幕中")
                .setSmallIcon(android.R.drawable.stat_sys_download).build();
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        String name = intent == null ? null : intent.getStringExtra("file");
        File dir = getExternalFilesDir(null);
        File f = name == null ? null : new File(dir, new File(name).getName());
        new Thread(() -> {
            try {
                if (f == null || !f.isFile()) throw new IllegalArgumentException("missing " + f);
                Log.w(SubtitleRemover.TAG, "bench start " + f.getName());
                SubtitleRemover.Result r = new SubtitleRemover().run(this, Uri.fromFile(f), null, new SubtitleRemover.Listener() {
                    int last = -1;
                    @Override public void onStage(int s) { Log.w(SubtitleRemover.TAG, "stage " + s); }
                    @Override public void onProgress(int d, int t) {
                        int p = d * 10 / Math.max(1, t);
                        if (p != last) { last = p; Log.w(SubtitleRemover.TAG, "progress " + d + "/" + t); }
                    }
                });
                Log.w(SubtitleRemover.TAG, "bench done " + r.uri + " " + r.width + "x" + r.height + " frames=" + r.frames + " ms=" + r.ms);
            } catch (Throwable e) {
                Log.w(SubtitleRemover.TAG, "bench failed", e);
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE);
                stopSelf(startId);
            }
        }).start();
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent i) { return null; }
}
