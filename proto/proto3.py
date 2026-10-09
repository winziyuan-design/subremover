# v0.2 pipeline mirror: stroke masks, bidirectional background plates with consistency check,
# per-line LaMa anchor propagated forward, Telea last resort.
import cv2, numpy as np, sys, time, onnxruntime as ort
DET = "/workspace/subremover/third_party/det.onnx"
LAMA = "/workspace/subremover/third_party/lama_w8e.onnx"
src, dst = sys.argv[1], sys.argv[2]
K = 6; S = 0.5
det = cv2.dnn_TextDetectionModel_DB(DET)
det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(2.0)
MEAN = (122.67891434, 116.66876762, 104.00698793)
lama = ort.InferenceSession(LAMA)
def detect(img, maxw):
    h, w = img.shape[:2]; s = min(1.0, maxw / w)
    iw = max(32, int(round(w*s/32))*32); ih = max(32, int(round(h*s/32))*32)
    det.setInputParams(1.0/255, (iw, ih), MEAN)
    polys, _ = det.detect(img)
    return [cv2.boundingRect(np.array(p, np.int32)) for p in polys]
T0 = time.time(); tm = {}
cap = cv2.VideoCapture(src); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    if not frames: H, Wd = f.shape[:2]
    frames.append(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 97])[1])
N = len(frames)
class _F:
    def __getitem__(self, i): return cv2.imdecode(frames[i], cv2.IMREAD_COLOR)
FR = _F()
# ---- band from samples (lower 60% only) ----
cand = []
for i in np.linspace(0, N-1, 10).astype(int):
    top = H*2//5
    for (x,y,bw,bh) in detect(FR[int(i)][top:], 960):
        y += top; cx = x + bw/2
        if y+bh/2 > H*0.5 and abs(cx-Wd/2) < Wd*0.12 and H*0.02 < bh < H*0.14 and bw > bh*1.5: cand.append((y, y+bh))
cy = np.median([(a+b)/2 for a,b in cand]); LH = np.median([b-a for a,b in cand])
keep = [(a,b) for a,b in cand if abs((a+b)/2-cy) < LH*1.6]
y0 = max(0, int(min(a for a,b in keep)-LH*0.6)) & ~1; y1 = min(H, int(max(b for a,b in keep)+LH*0.6))
bands = [FR[i][y0:y1].copy() for i in range(N)]; bh_, bw_ = bands[0].shape[:2]
print("band", y0, y1, "LH", LH)
# ---- detection every K, box regions ----
t = time.time(); dets = {}
for i in range(0, N, K):
    dets[i] = [r for r in detect(bands[i], 640) if abs(r[0]+r[2]/2-Wd/2) < Wd*0.2 and r[3] < LH*1.8 and r[2] > r[3]]
tm['detect'] = time.time()-t
pad = int(LH*0.15)+2
hatK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(LH*0.5)|1, int(LH*0.5)|1))
dk = max(3, int(round(LH*0.12))); dilK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dk+1, 2*dk+1))
def region(i):
    a = i//K*K; m = np.zeros((bh_, bw_), np.uint8)
    for j in (a-K, a, a+K):
        for (x,y,w,h) in dets.get(j, []): cv2.rectangle(m, (x-pad, y-pad), (x+w+pad, y+h+pad), 255, -1)
    return m
def stroke_raw(i):
    reg = region(i)
    if not reg.any(): return reg
    hsv = cv2.cvtColor(bands[i], cv2.COLOR_BGR2HSV); g = hsv[...,2]
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, hatK)
    s = ((th > 35) & (g > 170) & (hsv[...,1] < 80)).astype(np.uint8)*255
    return s & reg
_raw = {}
def raw(i):
    if i not in _raw: _raw[i] = stroke_raw(min(max(i,0),N-1))
    return _raw[i]
def stroke(i):
    # subtitle strokes stay put while the background moves: keep pixels stable over 2 frames back or forward
    s = (raw(i) & raw(i-2)) | (raw(i) & raw(i+2))
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    return cv2.dilate(s, dilK) & cv2.dilate(region(i), dilK)
t = time.time(); masks = [stroke(i) for i in range(N)]; tm['mask'] = time.time()-t
area_box = sum(int((region(i)>0).sum()) for i in range(N)); area_st = sum(int((m>0).sum()) for m in masks)
print("mask px box=%d stroke=%d (%.0f%%)" % (area_box, area_st, 100*area_st/max(1,area_box)))
# line segments (for LaMa anchor reset)
seg = [0]*N
for i in range(1, N):
    a, b = masks[i-1] > 0, masks[i] > 0
    u = (a|b).sum(); iou = (a&b).sum()/u if u else 1
    seg[i] = seg[i-1] + (1 if iou < 0.5 else 0)
# ---- consecutive flows (completed inside holes) + consistency ----
sw, sh = int(bw_*S), int(bh_*S)
grays = [cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA) for b in bands]
smask = [cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST) for m in masks]
tiny = [cv2.resize(g, (64, max(4, 64*sh//sw))).astype(np.float32) for g in grays]
cut = [False] + [np.abs(tiny[i]-tiny[i-1]).mean() > 30 for i in range(1, N)]
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
sig = max(4.0, LH*S*0.8)
gxs, gys = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
gxf, gyf = np.meshgrid(np.arange(bw_, dtype=np.float32), np.arange(bh_, dtype=np.float32))
def flow(a, b):   # for pixel in a -> offset to b, completed
    f = dis.calc(grays[a], grays[b], None)
    hole = cv2.dilate(((smask[a] > 0) | (smask[b] > 0)).astype(np.uint8), np.ones((5,5),np.uint8)) > 0
    v = (~hole).astype(np.float32)
    num = cv2.GaussianBlur(f*v[...,None], (0,0), sig); den = np.maximum(cv2.GaussianBlur(v, (0,0), sig), 1e-4)[...,None]
    f = np.where(hole[...,None], num/den, f)
    if np.median(np.abs(f)) < 0.15 and np.abs(f).max() < 0.6: f[:] = 0   # static: avoid resampling blur
    return f
def consistent(fab, fba):
    mx = gxs + fab[...,0]; my = gys + fab[...,1]
    back = cv2.remap(fba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(99,99))
    err = np.linalg.norm(fab + back, axis=2)
    return err < 0.5 + 0.05*np.linalg.norm(fab, axis=2)
def fullmap(f):
    fu = cv2.resize(f, (bw_, bh_))/S
    return gxf + fu[...,0], gyf + fu[...,1]
def warp_plate(P, A, V, a, b):
    """plate defined at frame b -> frame a coordinates"""
    if cut[max(a,b)]: return np.zeros_like(P), np.full_like(A, 255), np.zeros_like(V)
    fab = flow(a, b); fba = flow(b, a)
    ok = cv2.resize(consistent(fab, fba).astype(np.uint8), (bw_, bh_), interpolation=cv2.INTER_NEAREST)
    if not fab.any():
        return P.copy(), np.minimum(A.astype(np.int16)+1, 255).astype(np.uint8), V & ok
    mx, my = fullmap(fab)
    P2 = cv2.remap(P, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    A2 = cv2.remap(A, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    V2 = cv2.remap(V, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return P2, np.minimum(A2.astype(np.int16)+1, 255).astype(np.uint8), V2 & ok
def update(P, A, V, i):
    free = masks[i] == 0
    P[free] = bands[i][free]; A[free] = 0; V[free] = 1
# ---- backward pass ----
t = time.time()
bwd = [None]*N
P = bands[N-1].copy(); A = np.full((bh_, bw_), 255, np.uint8); V = np.zeros((bh_, bw_), np.uint8); update(P, A, V, N-1)
bwd[N-1] = (P.copy(), A.copy(), V.copy())
for i in range(N-2, -1, -1):
    P, A, V = warp_plate(P, A, V, i, i+1); update(P, A, V, i); bwd[i] = (P.copy(), A.copy(), V.copy())
tm['bwd'] = time.time()-t
# ---- forward pass + LaMa anchors ----
t = time.time(); tl = 0; nl = 0; st = dict(fwd=0, bwd=0, lama=0, telea=0)
def run_lama(full, hole):  # full BGR frame, hole uint8 full-frame
    ys, xs = np.nonzero(hole); x0_, x1_, y0_, y1_ = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(512, (x1_-x0_)*1.3, (y1_-y0_)*3)); side = min(side, min(full.shape[:2]) if False else max(full.shape[:2]))
    cxx, cyy = (x0_+x1_)//2, (y0_+y1_)//2
    cw, ch_ = min(side, full.shape[1]), min(side, full.shape[0])
    lx = int(np.clip(cxx-cw//2, 0, full.shape[1]-cw)); ly = int(np.clip(cyy-ch_//2, 0, full.shape[0]-ch_))
    crop = full[ly:ly+ch_, lx:lx+cw]; hm = hole[ly:ly+ch_, lx:lx+cw]
    img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512,512), interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2,0,1)[None]/255
    mk = (cv2.resize(cv2.dilate(hm, np.ones((5,5),np.uint8)), (512,512)) > 0).astype(np.float32)[None,None]
    o = lama.run(None, {"image": img, "mask": mk})[0][0].transpose(1,2,0)
    o = cv2.cvtColor(np.clip(o,0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    o = cv2.resize(o, (cw, ch_), interpolation=cv2.INTER_CUBIC)
    out = full.copy(); region = out[ly:ly+ch_, lx:lx+cw]; region[hm>0] = o[hm>0]
    return out
outs = []
P = bands[0].copy(); A = np.full((bh_, bw_), 255, np.uint8); V = np.zeros((bh_, bw_), np.uint8)
Hh = np.zeros_like(bands[0]); HV = np.zeros((bh_, bw_), np.uint8)
lastLama = -99
for i in range(N):
    if i > 0:
        P, A, V = warp_plate(P, A, V, i, i-1)
        if seg[i] != seg[i-1] or cut[i]: HV[:] = 0
        else: Hh, _, HV = warp_plate(Hh, np.zeros_like(A), HV, i, i-1)
    update(P, A, V, i)
    T = bands[i]; m = masks[i] > 0
    if not m.any(): outs.append(T); continue
    R = T.copy(); rem = m.copy()
    bP, bA, bV = bwd[i]
    useF = rem & (V > 0) & (~(bV > 0) | (A <= bA)); R[useF] = P[useF]; rem &= ~useF; st['fwd'] += useF.sum()
    useB = rem & (bV > 0); R[useB] = bP[useB]; rem &= ~useB; st['bwd'] += useB.sum()
    useH = rem & (HV > 0); R[useH] = Hh[useH]; rem &= ~useH; st['lama'] += useH.sum()
    if rem.sum() > max(60, 0.03*m.sum()) and (i - lastLama >= 15 or seg[i] != seg[max(0,i-1)] or not HV.any()):
        full = FR[i]; full[y0:y1] = R
        hole = np.zeros((H, Wd), np.uint8); hole[y0:y1][rem] = 255
        t1 = time.time(); filled = run_lama(full, hole); tl += time.time()-t1; nl += 1
        fb = filled[y0:y1]; R[rem] = fb[rem]; Hh[rem] = fb[rem]; HV[rem] = 1; st['lama'] += rem.sum(); rem[:] = False; lastLama = i
    if rem.any():
        st['telea'] += rem.sum(); R = cv2.inpaint(R, rem.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
    soft = cv2.GaussianBlur(m.astype(np.float32), (5,5), 0)[...,None]
    outs.append((R*soft + T*(1-soft)).astype(np.uint8))
tm['fwd'] = time.time()-t; tm['lama'] = tl
print("lama calls", nl, "segments", seg[-1]+1, {k:int(v) for k,v in st.items()}, {k:round(v,1) for k,v in tm.items()}, "total %.1fs" % (time.time()-T0))
vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(5), (Wd, H))
for i in range(N):
    f = FR[i]; f[y0:y1] = outs[i]; vw.write(f)
vw.release()
