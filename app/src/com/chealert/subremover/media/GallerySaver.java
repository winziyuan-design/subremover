package com.chealert.subremover.media;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;

/** Publishes a finished mp4 into Movies/去字幕 via MediaStore. */
public final class GallerySaver {
    public static final String DIR = Environment.DIRECTORY_MOVIES + "/去字幕";

    private GallerySaver() { }

    public static Uri save(Context ctx, File file, String name) throws Exception {
        ContentResolver cr = ctx.getContentResolver();
        ContentValues v = new ContentValues();
        v.put(MediaStore.Video.Media.DISPLAY_NAME, name);
        v.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
        v.put(MediaStore.Video.Media.RELATIVE_PATH, DIR);
        v.put(MediaStore.Video.Media.IS_PENDING, 1);
        Uri uri = cr.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, v);
        if (uri == null) throw new IllegalStateException("insert failed");
        try (InputStream in = new FileInputStream(file); OutputStream out = cr.openOutputStream(uri)) {
            byte[] b = new byte[1 << 16];
            int n;
            while ((n = in.read(b)) > 0) out.write(b, 0, n);
        }
        v.clear();
        v.put(MediaStore.Video.Media.IS_PENDING, 0);
        cr.update(uri, v, null, null);
        return uri;
    }
}
