# Fine-tune STTN (researchmm/STTN, MIT) on synthetic burned-in subtitles. Single GPU: python finetune.py ...; multi GPU: torchrun --nproc_per_node=N finetune.py ...
import os, sys, json, time, math, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.dirname(HERE))
import sttn_net, dataset, evaluate
ap=argparse.ArgumentParser()
ap.add_argument('--config'); ap.add_argument('--data',nargs='+'); ap.add_argument('--val'); ap.add_argument('--out',default='runs/ft')
ap.add_argument('--sttn-repo',default=sttn_net.STTN_REPO); ap.add_argument('--ckpt',default=sttn_net.STTN_CKPT); ap.add_argument('--resume',default='auto')
ap.add_argument('--iters',type=int,default=100000); ap.add_argument('--bs',type=int,default=4); ap.add_argument('--T',type=int,default=5); ap.add_argument('--nb',type=int,default=3)
ap.add_argument('--lr',type=float,default=5e-5); ap.add_argument('--lr-d',type=float,default=1e-4); ap.add_argument('--warmup',type=int,default=1000); ap.add_argument('--decay',default='cosine')
ap.add_argument('--w-hole',type=float,default=1.0); ap.add_argument('--w-valid',type=float,default=1.0); ap.add_argument('--w-adv',type=float,default=0.01); ap.add_argument('--w-perc',type=float,default=0.05)
ap.add_argument('--d-start',type=int,default=2000); ap.add_argument('--amp',default='bf16',choices=['none','bf16','fp16']); ap.add_argument('--workers',type=int,default=8)
ap.add_argument('--scale',type=float,nargs=2,default=[0.7,1.0]); ap.add_argument('--p-rot',type=float,default=0.15); ap.add_argument('--p-free',type=float,default=0.15); ap.add_argument('--p-jpeg',type=float,default=0.7)
ap.add_argument('--fonts',nargs='*'); ap.add_argument('--corpus'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
ap.add_argument('--log-every',type=int,default=50); ap.add_argument('--save-every',type=int,default=2000); ap.add_argument('--val-every',type=int,default=2000); ap.add_argument('--val-limit',type=int,default=0)
ap.add_argument('--threads',type=int,default=0)
a=ap.parse_args()
if a.config:
    cfg=json.load(open(a.config)); ap.set_defaults(**{k.replace('-','_'):v for k,v in cfg.items()}); a=ap.parse_args()
if not a.data: ap.error('--data required (dirs from prep_frames.py)')
ddp=int(os.environ.get('WORLD_SIZE','1'))>1; rank=int(os.environ.get('RANK','0'))
if ddp:
    torch.distributed.init_process_group('nccl'); lr_=int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr_); dev=torch.device('cuda',lr_)
else: dev=torch.device(a.device)
if a.threads: torch.set_num_threads(a.threads)
torch.manual_seed(a.seed+rank); main=rank==0
os.makedirs(a.out,exist_ok=True)
if main: json.dump(vars(a),open(os.path.join(a.out,'args.json'),'w'),indent=1)
netG=sttn_net.generator(a.sttn_repo,a.ckpt or None,init=not a.ckpt).to(dev)
netD=sttn_net.discriminator(a.sttn_repo).to(dev) if a.w_adv>0 else None
optG=torch.optim.Adam(netG.parameters(),lr=a.lr,betas=(0.0,0.99))
optD=torch.optim.Adam(netD.parameters(),lr=a.lr_d,betas=(0.0,0.99)) if netD else None
use_amp=a.amp!='none' and dev.type=='cuda'; adt={'bf16':torch.bfloat16,'fp16':torch.float16}.get(a.amp)
scaler=torch.amp.GradScaler('cuda',enabled=use_amp and a.amp=='fp16')
step=0; ck=os.path.join(a.out,'last.pth') if a.resume=='auto' else a.resume
if ck and os.path.exists(ck):
    s=torch.load(ck,map_location='cpu'); netG.load_state_dict(s['netG']); optG.load_state_dict(s['optG']); step=s['step']
    if netD and 'netD' in s: netD.load_state_dict(s['netD']); optD.load_state_dict(s['optD'])
    if 'scaler' in s: scaler.load_state_dict(s['scaler'])
    if main: print('resumed',ck,'step',step)
G=nn.parallel.DistributedDataParallel(netG,device_ids=[dev.index]) if ddp else netG
class DClip(nn.Module):
    """T-PatchGAN on [B*T,C,H,W] -> per-clip features (the upstream forward folds the batch into time)"""
    def __init__(s, d): super().__init__(); s.d=d
    def forward(s, x, b): return s.d.conv(x.view(b,-1,*x.shape[1:]).transpose(1,2))
D=DClip(netD) if netD else None
if ddp and D: D=nn.parallel.DistributedDataParallel(D,device_ids=[dev.index])
class VGG(nn.Module):
    def __init__(s):
        super().__init__(); import torchvision
        f=torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in f.parameters(): p.requires_grad_(False)
        s.f=f; s.ids={3,8,15}
        s.register_buffer('mu',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)); s.register_buffer('sd',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(s, x, y):
        x=((x+1)/2-s.mu)/s.sd; y=((y+1)/2-s.mu)/s.sd; l=0
        for i,m in enumerate(s.f):
            x=m(x); y=m(y)
            if i in s.ids: l=l+F.l1_loss(x,y)
        return l
vgg=VGG().to(dev) if a.w_perc>0 else None
def hinge_d(real, fake): return (F.relu(1-real).mean()+F.relu(1+fake).mean())/2
def lr_at(s):
    w=min(1.0,(s+1)/max(1,a.warmup))
    if a.decay=='cosine': w*=0.1+0.9*0.5*(1+math.cos(math.pi*min(1.0,s/a.iters)))
    return w
ds=dataset.Clips(a.data,T=a.T,nb=a.nb,scale=tuple(a.scale),p_rot=a.p_rot,p_free=a.p_free,p_jpeg=a.p_jpeg,fonts=a.fonts,corpus=a.corpus)
dl=torch.utils.data.DataLoader(ds,batch_size=a.bs,num_workers=a.workers,pin_memory=dev.type=='cuda',drop_last=True,persistent_workers=a.workers>0,shuffle=False)
def save(tag):
    if not main: return
    s=dict(netG=netG.state_dict(),optG=optG.state_dict(),step=step,scaler=scaler.state_dict(),args=vars(a))
    if netD: s.update(netD=netD.state_dict(),optD=optD.state_dict())
    torch.save(s,os.path.join(a.out,'last.pth.tmp')); os.replace(os.path.join(a.out,'last.pth.tmp'),os.path.join(a.out,'last.pth'))
    torch.save({'netG':netG.state_dict()},os.path.join(a.out,f'gen_{tag}.pth'))
def validate():
    if not (main and a.val): return None
    netG.eval(); w=sttn_net.Wrap(netG)
    def run(fr,mk):
        with torch.inference_mode(), torch.autocast(dev.type,dtype=adt,enabled=use_amp):
            return w(torch.from_numpy(fr).to(dev),torch.from_numpy(mk).to(dev)).float().cpu().numpy()
    r=evaluate.evaluate(run,a.val,a.val_limit); netG.train(); return r
log=open(os.path.join(a.out,'log.jsonl'),'a') if main else None
G.train(); t0=time.time(); it=iter(dl); acc={}
if main and step==0 and a.val:
    r=validate(); print('val@0',json.dumps(r)); log.write(json.dumps(dict(step=0,val=r))+'\n'); log.flush()
while step<a.iters:
    try: gt,inp,m=next(it)
    except StopIteration: it=iter(dl); gt,inp,m=next(it)
    gt,inp,m=gt.to(dev,non_blocking=True),inp.to(dev,non_blocking=True),m.to(dev,non_blocking=True)
    b,t,c,h,w=gt.shape; k=lr_at(step)
    for g in optG.param_groups: g['lr']=a.lr*k
    if optD:
        for g in optD.param_groups: g['lr']=a.lr_d*k
    with torch.autocast(dev.type,dtype=adt,enabled=use_amp):
        pred=G(inp*(1-m),m).float()
    gtf=gt.view(b*t,c,h,w); mf=m.view(b*t,1,h,w); comp=gtf*(1-mf)+pred*mf; L={}
    useD=D is not None and step>=a.d_start
    if useD:
        with torch.autocast(dev.type,dtype=adt,enabled=use_amp):
            fr,ff=D(torch.cat([gtf,comp.detach()]),2*b).float().chunk(2); ld=hinge_d(fr,ff)
        optD.zero_grad(set_to_none=True); scaler.scale(ld).backward(); scaler.step(optD); L['d']=ld.item()
    L['hole']=F.l1_loss(pred*mf,gtf*mf)/mf.mean().clamp_min(1e-4)*a.w_hole
    L['valid']=F.l1_loss(pred*(1-mf),gtf*(1-mf))/(1-mf).mean().clamp_min(1e-4)*a.w_valid
    lg=L['hole']+L['valid']
    if useD:
        Dm=D.module if ddp else D; Dm.requires_grad_(False)
        with torch.autocast(dev.type,dtype=adt,enabled=use_amp): L['adv']=-Dm(comp,b).float().mean()*a.w_adv
        Dm.requires_grad_(True)
        lg=lg+L['adv']
    if vgg is not None:
        with torch.autocast(dev.type,dtype=adt,enabled=use_amp): L['perc']=vgg(comp,gtf).float()*a.w_perc
        lg=lg+L['perc']
    optG.zero_grad(set_to_none=True); scaler.scale(lg).backward(); scaler.unscale_(optG); nn.utils.clip_grad_norm_(netG.parameters(),1.0); scaler.step(optG); scaler.update()
    step+=1
    for kk,v in L.items(): acc[kk]=acc.get(kk,0)+float(v)
    if main and step%a.log_every==0:
        r={kk:round(v/a.log_every,4) for kk,v in acc.items()}; r.update(step=step,lr=a.lr*k,s_per_it=round((time.time()-t0)/a.log_every,3)); acc={}; t0=time.time()
        print(json.dumps(r),flush=True); log.write(json.dumps(r)+'\n'); log.flush()
    if step%a.save_every==0 or step==a.iters: save(f'{step:07d}')
    if a.val and (step%a.val_every==0 or step==a.iters):
        r=validate()
        if main: print('val',step,json.dumps(r),flush=True); log.write(json.dumps(dict(step=step,val=r))+'\n'); log.flush()
if ddp: torch.distributed.destroy_process_group()
