# Synthetic burned-in subtitles / watermarks: per-frame sparse RGBA layers over clean frames (clean frame = ground truth, dilated layer alpha = hole mask).
import os, glob, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
COMMON=('的一是不了人我在有他这中大来上个国到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关并内加化由却代军产入先山五太水万市眼体别处总才场师书比住员九笑性通目华报立马命张活难神数件安表原车白应路期叫死常提感金何更反合放做系计或司利受光王果亲界及今京务制解各任至清物台象记边共风战干接它许八特觉望直服毛林题建南度统色字请交爱让认算论百吃义科怎元社术结六功指思非流每青管夫连远资队跟带花快条院变联言权往展该领传近留红治决周保达办运武半候七必城父强步完革深区即求品士转量空甚众技轻程告江语英基派满式李息写呢识极令黄德收脸钱党倒未持音哥找片拿视场')
PUNCT='，。！？、…'
LATIN='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghkmnprstuvwxyz0123456789'
def find_fonts(paths=None):
    if paths:
        fs=[f for p in paths for f in (sorted(glob.glob(os.path.join(p,'**','*.[ot]t[fc]'),recursive=True)) if os.path.isdir(p) else [p])]
        if fs: return fs
    try: out=subprocess.run(['fc-list',':lang=zh','file'],capture_output=True,text=True).stdout
    except OSError: out=''
    fs=sorted({l.split(':')[0].strip() for l in out.splitlines() if l.strip()})
    if not fs: raise RuntimeError('no CJK font found: pass --fonts (e.g. NotoSansCJK / SourceHanSans)')
    return fs
_FC={}
def _font(path,size):
    k=(path,size)
    if k not in _FC:
        if len(_FC)>256: _FC.clear()
        _FC[k]=ImageFont.truetype(path,size)
    return _FC[k]
_GL={}
def _drawable(path, text):
    """drop characters the font renders as its .notdef box"""
    f=_font(path,32); tofu=_GL.setdefault((path,None),bytes(f.getmask('\U000F0000')))
    def ok(ch):
        k=(path,ch)
        if k not in _GL: _GL[k]=ch.isspace() or bytes(f.getmask(ch))!=tofu
        return _GL[k]
    return ''.join(ch for ch in text if ok(ch))
def rand_line(rng, corpus=None, lo=3, hi=16):
    if corpus and rng.random()<0.8:
        s=corpus[rng.integers(len(corpus))].strip()
        if s: return s[:hi]
    n=int(rng.integers(lo,hi+1)); s=''.join(COMMON[i] for i in rng.integers(0,len(COMMON),n))
    if n>6 and rng.random()<0.3: k=int(rng.integers(2,n-2)); s=s[:k]+PUNCT[rng.integers(len(PUNCT))]+s[k:]
    if rng.random()<0.1: s+=''.join(LATIN[i] for i in rng.integers(0,len(LATIN),int(rng.integers(2,5))))
    return s
def render_text(text, font, size, rng, fill=None, stroke=None, shadow=None):
    """-> premultiplied (color HxWx3 uint8, alpha HxW float32) of white text + dark outline (+ blurred drop shadow)"""
    text=_drawable(font,text) or '的'; f=_font(font,size); sw=max(1,int(round(size*(stroke if stroke is not None else rng.uniform(0.04,0.12)))))
    l,t,r,b=f.getbbox(text,stroke_width=sw); pad=sw+size//4
    w,h=r-l+2*pad,b-t+2*pad; img=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(img)
    if fill is None:
        fill=(255,255,255) if rng.random()<0.75 else tuple(int(v) for v in rng.choice([(255,235,90),(250,250,210),(120,230,255),(255,190,60)]))
    sc=tuple(int(v) for v in (rng.integers(0,40,3) if rng.random()<0.9 else rng.integers(0,255,3)))
    shd=shadow if shadow is not None else rng.random()<0.5
    if shd:
        sh=Image.new('RGBA',(w,h),(0,0,0,0)); o=max(1,size//14)
        ImageDraw.Draw(sh).text((pad-l+o,pad-t+o),text,font=f,fill=(0,0,0,int(rng.integers(120,230))),stroke_width=sw,stroke_fill=(0,0,0,200))
        img=Image.alpha_composite(sh.filter(ImageFilter.GaussianBlur(max(0.5,size/30))),img)
    d=ImageDraw.Draw(img)
    if sw>0 and rng.random()<0.92: d.text((pad-l,pad-t),text,font=f,fill=fill+(255,),stroke_width=sw,stroke_fill=sc+(255,))
    else: d.text((pad-l,pad-t),text,font=f,fill=fill+(255,))
    a=np.asarray(img,np.float32)/255
    return a[...,:3],a[...,3]
def _paste(acc_c, acc_a, x, y, c, a, op=1.0):
    H,W=acc_a.shape; h,w=a.shape; xa,ya,xb,yb=max(0,x),max(0,y),min(W,x+w),min(H,y+h)
    if xb<=xa or yb<=ya: return
    sa=a[ya-y:yb-y,xa-x:xb-x]*op; sc=c[ya-y:yb-y,xa-x:xb-x]
    acc_c[ya:yb,xa:xb]=sc*sa[...,None]+acc_c[ya:yb,xa:xb]*(1-sa[...,None]); acc_a[ya:yb,xa:xb]=sa+acc_a[ya:yb,xa:xb]*(1-sa)
class Overlay:
    """T-frame overlay track for a W x H canvas. mode='frame': subtitles near bottom-centre; mode='crop': anywhere in a training crop"""
    def __init__(s, W, H, T, rng, fonts, mode='frame', corpus=None, size=None, p_wm=0.4, p_two=0.25, p_gap=0.1, p_move=0.5, seg=(12,60)):
        s.W,s.H,s.T,s.items=W,H,T,[]
        base=min(W,H); sz=size or int(rng.uniform(0.035,0.075)*base if mode=='frame' else rng.uniform(18,72))
        t=int(-rng.integers(0,seg[0]))
        font=fonts[rng.integers(len(fonts))]; cy=rng.uniform(0.78,0.93) if mode=='frame' else rng.uniform(0.15,0.85)
        cx=rng.normal(0.5,0.02 if mode=='frame' else 0.2)
        while t<T:
            n=int(rng.integers(*seg))
            if rng.random()>p_gap:
                lines=[rand_line(rng,corpus) for _ in range(2 if rng.random()<p_two else 1)]
                layers=[render_text(l,font,sz,rng,fill=(255,255,255) if rng.random()<0.8 else None) for l in lines]
                hh=sum(a.shape[0] for _,a in layers)-int(sz*0.3)*(len(layers)-1); y=int(cy*H-hh/2); fade=int(rng.integers(0,3))
                for c,a in layers:
                    x=int(cx*W-a.shape[1]/2); s.items.append(dict(t0=t,t1=t+n,c=c,a=a,x0=x,y0=y,vx=0.,vy=0.,op=1.0,fade=fade)); y+=a.shape[0]-int(sz*0.3)
            t+=n
        if rng.random()<p_wm:
            txt=('@' if rng.random()<0.6 else '')+rand_line(rng,corpus,2,7)
            c,a=render_text(txt,fonts[rng.integers(len(fonts))],int(rng.uniform(0.025,0.06)*base) if mode=='frame' else int(rng.uniform(14,48)),rng,fill=(255,255,255),shadow=False)
            op=rng.uniform(0.35,0.9); mv=rng.random()<p_move
            x0,y0=int(rng.uniform(0,max(1,W-a.shape[1]))),int(rng.uniform(0,max(1,H-a.shape[0])))
            v=rng.uniform(0.5,4.0)*rng.choice([-1,1],2) if mv else np.zeros(2)
            s.items.append(dict(t0=0,t1=T,c=c,a=a,x0=x0,y0=y0,vx=v[0],vy=v[1],op=op,fade=0,bounce=True))
    def layer(s, t):
        """-> (color HxWx3 float 0..1 premultiplied-composited, alpha HxW float)"""
        c=np.zeros((s.H,s.W,3),np.float32); a=np.zeros((s.H,s.W),np.float32)
        for it in s.items:
            if not it['t0']<=t<it['t1']: continue
            op=it['op']; f=it['fade']
            if f: op*=min(1.0,(t-it['t0']+1)/(f+1),(it['t1']-t)/(f+1))
            x,y=it['x0']+it['vx']*t,it['y0']+it['vy']*t
            if it.get('bounce'):
                h,w=it['a'].shape; x=_tri(x,max(1,s.W-w)); y=_tri(y,max(1,s.H-h))
            _paste(c,a,int(round(x)),int(round(y)),it['c'],it['a'],op)
        return c,a
    def apply(s, frame, t, dil=None, rng=None):
        """frame BGR uint8 -> (composited BGR uint8, hole mask uint8 255)"""
        c,a=s.layer(t); o=frame.astype(np.float32)/255; o=o*(1-a[...,None])+c[...,::-1]
        m=(a>0.03).astype(np.uint8)*255
        k=dil if dil is not None else 3
        if m.any() and k>0: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1)))
        return np.clip(o*255+0.5,0,255).astype(np.uint8),m
def _tri(v, n):
    v=abs(v)%(2*n); return v if v<n else 2*n-v
def jpeg(img, q):
    return cv2.imdecode(cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,int(q)])[1],cv2.IMREAD_COLOR)
