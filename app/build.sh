#!/bin/sh
set -eu
ROOT="$(cd "$(dirname "$0")" && pwd)"
export JAVA_HOME=/workspace/jdk17; export PATH="$JAVA_HOME/bin:$PATH"
SDK=/workspace/android-sdk; BT=$SDK/build-tools/35.0.0; JAR=$SDK/platforms/android-34/android.jar
VERSION_CODE=5; VERSION_NAME=0.3.0
CV_JAR="$ROOT/libs/opencv-4.10.0.jar"; ORT_JAR="$ROOT/libs/onnxruntime-1.30.0.jar"
cd "$ROOT"; rm -rf build gen classes; mkdir -p build/compiled gen classes dist
"$BT/aapt2" compile --dir res -o build/compiled/res.zip
"$BT/aapt2" link -o build/res.apk --manifest AndroidManifest.xml -I "$JAR" --java gen \
  --custom-package com.chealert.subremover --min-sdk-version 29 --target-sdk-version 34 \
  --version-code $VERSION_CODE --version-name $VERSION_NAME -A assets --no-compress-regex '\.onnx$' build/compiled/res.zip
find src gen -name '*.java' > build/sources.txt
javac --release 11 -encoding UTF-8 -Xlint:-options -classpath "$JAR:$CV_JAR:$ORT_JAR" -d classes @build/sources.txt
find classes -name '*.class' > build/classes.txt
"$BT/d8" --min-api 29 --lib "$JAR" --output build @build/classes.txt "$CV_JAR" "$ORT_JAR" 2>&1 | grep -v -i "warning\|missing\|^ *at \|R\$\|org.opencv.R" | head -5 || true
cp build/res.apk build/unsigned.apk
python3 - <<'PY'
import os, zipfile
with zipfile.ZipFile("build/unsigned.apk", "a") as z:
    for n in sorted(os.listdir("build")):
        if n.startswith("classes") and n.endswith(".dex"): z.write(os.path.join("build", n), n)
    for abi in os.listdir("libs/native"):
        for so in os.listdir(os.path.join("libs/native", abi)):
            zi = zipfile.ZipInfo("lib/%s/%s" % (abi, so)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            z.writestr(zi, open(os.path.join("libs/native", abi, so), "rb").read())
PY
"$BT/zipalign" -f -p 4 build/unsigned.apk build/aligned.apk
[ -f release.keystore ] || keytool -genkeypair -keystore release.keystore -storepass subremover -keypass subremover -alias subremover -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=SubRemover" >/dev/null
"$BT/apksigner" sign --ks release.keystore --ks-pass pass:subremover --key-pass pass:subremover --ks-key-alias subremover --out "dist/subremover-$VERSION_NAME.apk" build/aligned.apk
ls -lh dist/
