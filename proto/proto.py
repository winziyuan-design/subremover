# Python mirror of the on-device pipeline, for correctness testing on the box.
import cv2, numpy as np, sys, time
MODEL = "/workspace/subremover/third_party/det.onnx"
src, dst = sys.argv[1], sys.argv[2]
K = 3            # detect every K frames
W = 45           # temporal window each side
DIL = 8          # mask dilation px (outline/shadow)
OFFS = []
for o in [1,2,3,4,6,8,11,15,20,26,33,40,45]:
    OFFS += [o, -o]

det = cv2.dnn_TextDetectionModel_DB(MODEL)
det.setBinaryThreshold(0.3); det.setPolygonThreshold(0.5); det.setMaxCandidates(200); det.setUnclipRatio(2.0)
MEAN = (122.67891434, 116.66876762, 104.00698793)

def detect(img, maxw=960):
    h, w = img.shape[:2]
    s = min(1.0, maxw / w)
    iw = max(32, int(round(w*s/32))*32); ih = max(32, int(round(h*s/32))*32)
    det.setInputParams(1.0/255, (iw, ih), MEAN)
    polys, conf = det.detect(img)
    out = []
    for p in polys:
        p = np.array(p, np.float32); p[:,0] *= w/iw * (iw/(w*s)) if False else 1
        x, y, bw, bh = cv2.boundingRect(p.astype(np.int32))
        out.append((x, y, bw, bh))
    return out

cap = cv2.VideoCapture(src)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
Wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hd = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
t0 = time.time()
# ---- stage 0: prescan to find the subtitle band (lower part, stable) ----
cand = []
for i in np.linspace(0, N-1, 12).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
    if not ok: continue
    for (x,y,bw,bh) in detect(f):
        cx = x + bw/2
        if y + bh/2 > Hd*0.5 and abs(cx - Wd/2) < Wd*0.12 and Hd*0.02 < bh < Hd*0.14 and bw > bh*1.5:
            cand.append((y, y+bh))
if not cand: print("no subtitles"); sys.exit(1)
cy = np.median([(a+b)/2 for a,b in cand]); lh = np.median([b-a for a,b in cand])
keep = [(a,b) for a,b in cand if abs((a+b)/2 - cy) < lh*1.6]
y0 = max(0, int(min(a for a,b in keep) - lh*0.6)); y1 = min(Hd, int(max(b for a,b in keep) + lh*0.6))
LH = lh
print("band", y0, y1, "line", lh, "prescan %.2fs" % (time.time()-t0))
# ---- stage 1: read band crops + detect every K ----
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
bands, dets = [], {}
t1 = time.time()
i = 0
while True:
    ok, f = cap.read()
    if not ok: break
    b = f[y0:y1].copy(); bands.append(b)
    if i % K == 0:
        dets[i] = [r for r in detect(b) if abs(r[0]+r[2]/2 - Wd/2) < Wd*0.2 and r[3] < LH*1.8 and r[2] > r[3]]
    i += 1
N = len(bands); bh_, bw_ = bands[0].shape[:2]
last = (N-1)//K*K
if last not in dets: dets[last] = []
print("decode+detect %.2fs, %d dets" % (time.time()-t1, len(dets)))
ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2*DIL+1, 2*DIL+1))
def detmask(j):
    m = np.zeros((bh_, bw_), np.uint8)
    for (x,y,w_,h_) in dets.get(j, []):
        cv2.rectangle(m, (x,y), (x+w_-1, y+h_-1), 255, -1)
    return m
dm = {j: detmask(j) for j in dets}
masks = []
for t in range(N):
    a = t//K*K; b = min(a+K, last)
    m = dm[a] | dm[b]
    # timeline extension (like VSR's forward/backward frame count)
    for j in (a-K, b+K):
        if j in dm and abs(j - t) <= K: m |= dm[j]
    masks.append(cv2.dilate(m, ker))
# ---- stage 2: consecutive motion (t-1 -> t) as partial affine on unmasked pixels ----
t2 = time.time()
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
S = 0.5
grays = [cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None, fx=S, fy=S) for b in bands]
small_masks = [cv2.resize(m, (grays[0].shape[1], grays[0].shape[0]), interpolation=cv2.INTER_NEAREST) for m in masks]
A = [None]*N   # A[t]: maps coords in frame t-1 -> frame t  (2x3), None = cut
gy, gx = np.mgrid[4:grays[0].shape[0]:8, 4:grays[0].shape[1]:8]
tiny = [cv2.resize(g, (64, max(8, 64*g.shape[0]//g.shape[1]))).astype(np.float32) for g in grays]
for t in range(1, N):
    if np.abs(tiny[t]-tiny[t-1]).mean() > 30: A[t] = None; continue
    flow = dis.calc(grays[t], grays[t-1], None)  # for pixel in t, offset to t-1
    ok = (small_masks[t][gy, gx] == 0) & (small_masks[t-1][gy, gx] == 0)
    if ok.sum() < 20: A[t] = np.float32([[1,0,0],[0,1,0]]); continue
    pt = np.stack([gx[ok], gy[ok]], 1).astype(np.float32)
    ps = pt + flow[gy[ok], gx[ok]]
    M, inl = cv2.estimateAffinePartial2D(ps, pt, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if M is None or inl.mean() < 0.3: A[t] = None; continue
    M[:,2] /= S
    A[t] = M.astype(np.float32)
print("motion %.2fs" % (time.time()-t2))
def compose(M2, M1):  # apply M1 then M2
    a = np.vstack([M1, [0,0,1]]); b = np.vstack([M2, [0,0,1]]); return (b@a)[:2]
def chain(s, t):
    M = np.float32([[1,0,0],[0,1,0]])
    if s < t:
        for k in range(s+1, t+1):
            if A[k] is None: return None
            M = compose(A[k], M)
    else:
        for k in range(s, t, -1):
            if A[k] is None: return None
            M = compose(cv2.invertAffineTransform(A[k]), M)
    return M
# ---- stage 3: temporal background exposure + fallback ----
t3 = time.time()
out_bands = []; prev = None; stats = [0,0,0]
ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
for t in range(N):
    T = bands[t]; m = masks[t]
    if not m.any(): out_bands.append(T); prev = (T, t); continue
    R = T.copy(); rem = m.copy(); total = int((m>0).sum())
    edge = cv2.dilate(m, ring) & ~m
    for o in OFFS:
        s = t + o
        if s < 0 or s >= N or not rem.any(): continue
        M = chain(s, t)
        if M is None: continue
        Ws = cv2.warpAffine(bands[s], M, (bw_, bh_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        Wm = cv2.warpAffine(masks[s], M, (bw_, bh_), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # validity: compare on visible ring around the hole
        e = (edge > 0) & (Wm == 0)
        if e.sum() > 50:
            err = np.abs(Ws[e].astype(np.int16) - T[e].astype(np.int16)).mean()
            if err > 18: continue
        take = (rem > 0) & (Wm == 0)
        R[take] = Ws[take]; rem[take] = 0
    stats[0] += total - int((rem>0).sum())
    if rem.any() and prev is not None:
        M = chain(prev[1], t)
        if M is not None:
            Wp = cv2.warpAffine(prev[0], M, (bw_, bh_))
            Wv = cv2.warpAffine(np.full_like(m, 255), M, (bw_, bh_))
            take = (rem > 0) & (Wv > 0)
            e = (edge > 0)
            okp = e.sum() < 50 or np.abs(Wp[e].astype(np.int16) - T[e].astype(np.int16)).mean() <= 18
            if okp:
              R[take] = Wp[take]; stats[1] += int(take.sum()); rem[take] = 0
    if rem.any():
        stats[2] += int((rem>0).sum())
        R = cv2.inpaint(R, rem, 3, cv2.INPAINT_TELEA)
    # soften seam: blend a 3px feather at mask border
    soft = cv2.GaussianBlur(m, (5,5), 0).astype(np.float32)[...,None]/255
    O = (R*soft + T*(1-soft)).astype(np.uint8)
    O[m>0] = R[m>0] if False else O[m>0]
    out_bands.append(O); prev = (O, t)
print("fill %.2fs  temporal=%d prevprop=%d telea=%d" % (time.time()-t3, *stats))
# ---- stage 4: write ----
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
vw = cv2.VideoWriter(dst + ".noaudio.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wd, Hd))
for t in range(N):
    ok, f = cap.read(); f[y0:y1] = out_bands[t]; vw.write(f)
vw.release()
print("total %.2fs" % (time.time()-t0))
