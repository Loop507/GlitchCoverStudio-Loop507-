import streamlit as st
import numpy as np
import io
import random
import math
import os
import tempfile
import time
import hashlib
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageEnhance
import librosa
import imageio.v2 as imageio
from moviepy import VideoFileClip, AudioFileClip

st.set_page_config(page_title="GlitchCover Studio by Loop507", layout="centered")
st.title("🎵 GlitchCover Studio by Loop507")
st.markdown("Copertine e video glitch generati dall'analisi del tuo brano.")

# ═══════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════

def get_dimensions(fmt):
    return {"Quadrato (1:1)":(800,800),"Verticale (9:16)":(720,1280),"Orizzontale (16:9)":(1280,720)}.get(fmt,(800,800))

def make_divisible(size, block=16):
    w, h = size
    return (math.ceil(w/block)*block, math.ceil(h/block)*block)

# ═══════════════════════════════════════════
# ANALISI AUDIO — due livelli
# ═══════════════════════════════════════════

@st.cache_data(show_spinner=False)
def analyze_audio_features(audio_bytes: bytes):
    """Features medie per la copertina (veloce, 30s)."""
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=30)
        try:
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(bpm) if bpm is not None and not np.isnan(bpm) else 120.0
        except: bpm = 120.0

        def safe(v, fb): r = float(np.mean(v)); return r if not np.isnan(r) else fb

        rms   = safe(librosa.feature.rms(y=y), 0.05)
        sc    = safe(librosa.feature.spectral_centroid(y=y, sr=sr), 2000.0)
        sb    = safe(librosa.feature.spectral_bandwidth(y=y, sr=sr), 1000.0)
        sr_   = safe(librosa.feature.spectral_rolloff(y=y, sr=sr), 3000.0)
        zcr   = safe(librosa.feature.zero_crossing_rate(y), 0.1)
        mfcc  = np.nan_to_num(np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1))
        try:
            yh, yp = librosa.effects.hpss(y)
            he = safe(librosa.feature.rms(y=yh), rms*0.6)
            pe = safe(librosa.feature.rms(y=yp), rms*0.4)
        except: he, pe = rms*0.6, rms*0.4
        try:
            of = librosa.onset.onset_detect(y=y, sr=sr)
            os_ = len(of)/len(y)*sr if len(y)>0 else 1.0
        except: os_ = 1.0
        try:
            ch = librosa.feature.chroma_stft(y=y, sr=sr)
            tc = int(np.argmax(np.mean(ch, axis=1)))
        except: tc = 0

        dr = abs(sb - sc)
        fs = f"{bpm:.2f}_{rms:.4f}_{sc:.0f}_{len(audio_bytes)}"
        ho = hashlib.sha256((fs + str(audio_bytes[:1000])).encode()).hexdigest()
        hi = int(ho[:8], 16)

        if   rms>0.08 and bpm>120 and pe>0.03: emotion="Aggressive"
        elif rms>0.05 and bpm>110:             emotion="Energetico"
        elif rms<0.02 and bpm<70:              emotion="Ambient"
        elif rms<0.03 and bpm<90:              emotion="Calmo"
        elif he>pe*2:                          emotion="Melodico"
        elif pe>he*1.5:                        emotion="Ritmico"
        elif dr>3000:                          emotion="Dinamico"
        else:                                  emotion="Equilibrato"

        return dict(bpm=bpm,rms=rms,spectral_centroid=sc,spectral_bandwidth=sb,
                    spectral_rolloff=sr_,zero_crossing_rate=zcr,dynamic_range=dr,
                    harmonic_energy=he,percussive_energy=pe,onset_strength=os_,
                    tonal_centroid=tc,mfcc_features=mfcc,emotion=emotion,
                    file_hash=hi,file_size=len(audio_bytes),audio_signature=ho[:16])
    except Exception as e:
        st.error(f"Errore analisi audio: {e}"); return None

@st.cache_data(show_spinner=False)
def extract_frame_curves(audio_bytes: bytes, fps: int, duration_sec: float):
    """Estrae curve RMS + onset frame per frame per il video (audio completo)."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True,
                         duration=duration_sec if duration_sec and duration_sec > 0 else None)
    total_frames = int(len(y)/sr * fps)
    hop = max(1, len(y)//total_frames)

    rms_frames   = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset_env    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    zcr_frames   = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn)/(mx - mn + 1e-9)

    rms_n   = norm(rms_frames)
    onset_n = norm(onset_env)
    zcr_n   = norm(zcr_frames)

    # Smoothing con hanning — rende tutto fluido
    win = np.hanning(fps//2 + 1)
    win /= win.sum()
    rms_smooth   = np.convolve(rms_n,   win, mode='same')
    onset_smooth = np.convolve(onset_n, win, mode='same')
    zcr_smooth   = np.convolve(zcr_n,   win, mode='same')

    # Resample a esattamente total_frames
    def resample(arr, n):
        idx = np.linspace(0, len(arr)-1, n)
        return np.interp(idx, np.arange(len(arr)), arr)

    return (resample(rms_smooth, total_frames),
            resample(onset_smooth, total_frames),
            resample(zcr_smooth, total_frames),
            float(len(y)/sr))

# ═══════════════════════════════════════════
# MOTORE VISIVO — PALETTE + 10 BACKGROUND
# ═══════════════════════════════════════════

def build_palette(features, seed):
    rng = random.Random(seed)
    base_hue = ((features['tonal_centroid']*30) + rng.randint(0,360)) % 360
    sat = 0.4 + min(0.6, features['rms']*8)
    val = 0.5 + rng.uniform(-0.2, 0.2)
    scheme = rng.randint(0,5)
    if   scheme==0: hues=[base_hue,(base_hue+180)%360,(base_hue+30)%360,(base_hue+210)%360,(base_hue+60)%360,(base_hue+240)%360]
    elif scheme==1: hues=[base_hue,(base_hue+120)%360,(base_hue+240)%360,(base_hue+60)%360,(base_hue+180)%360,(base_hue+300)%360]
    elif scheme==2: hues=[(base_hue+i*15)%360 for i in range(6)]
    elif scheme==3: hues=[base_hue,(base_hue+150)%360,(base_hue+210)%360,(base_hue+30)%360,(base_hue+330)%360,(base_hue+90)%360]
    elif scheme==4: hues=[base_hue,(base_hue+90)%360,(base_hue+270)%360,(base_hue+45)%360,(base_hue+135)%360,(base_hue+225)%360]; sat=min(1.0,sat+0.3)
    else:           hues=[(base_hue+rng.randint(-20,20))%360 for _ in range(6)]

    def hsv(h,s,v):
        h/=360; i=int(h*6); f=h*6-i; p,q,t_=v*(1-s),v*(1-f*s),v*(1-(1-f)*s)
        return tuple(int(x*255) for x in [(v,t_,p),(q,v,p),(p,v,t_),(p,q,v),(t_,p,v),(v,p,q)][i%6])

    return [hsv(h, max(0.3,min(1.0,sat+rng.uniform(-0.1,0.1))),
                max(0.15,min(0.95,val+rng.uniform(-0.15,0.15)))) for h in hues]

# Background generators
def bg_plasma(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    s=rng.uniform(0.004,0.02); p1=rng.uniform(0,math.tau); p2=rng.uniform(0,math.tau)
    xv,yv=np.meshgrid(np.linspace(0,w*s,w),np.linspace(0,h*s,h))
    pl=(np.sin(xv+p1)+np.sin(yv+p2)+np.sin((xv+yv)*0.7)+np.sin(np.sqrt(xv**2+yv**2+0.01)))
    pl=(pl-pl.min())/(pl.max()-pl.min()+1e-9)
    n=len(pal); ca=np.array(pal,dtype=np.float32)
    idx=np.clip((pl*(n-1)).astype(int),0,n-2); frac=(pl*(n-1))-idx
    return Image.fromarray((ca[idx]*(1-frac[...,None])+ca[idx+1]*frac[...,None]).astype(np.uint8),'RGB')

def bg_voronoi(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    pts=np.array([(rng.randint(0,w),rng.randint(0,h)) for _ in range(rng.randint(6,20))])
    cols=np.array([pal[i%len(pal)] for i in range(len(pts))],dtype=np.uint8)
    xv,yv=np.meshgrid(np.arange(w),np.arange(h))
    dx=xv[...,None]-pts[:,0]; dy=yv[...,None]-pts[:,1]
    return Image.fromarray(cols[np.argmin(dx**2+dy**2,axis=2)],'RGB')

def bg_gradient(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    a=rng.uniform(0,math.pi)
    xv,yv=np.meshgrid(np.arange(w),np.arange(h))
    proj=xv*math.cos(a)+yv*math.sin(a)
    pn=(proj-proj.min())/(proj.max()-proj.min()+1e-9)
    c1,c2=np.array(pal[0],dtype=np.float32),np.array(pal[rng.randint(1,len(pal)-1)],dtype=np.float32)
    return Image.fromarray((c1*(1-pn[...,None])+c2*pn[...,None]).astype(np.uint8),'RGB')

def bg_interference(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    f1=f['spectral_centroid']/5000*rng.uniform(0.5,2.0)
    f2=f['spectral_bandwidth']/3000*rng.uniform(0.5,2.0)
    ph=rng.uniform(0,math.tau)
    xv,yv=np.meshgrid(np.linspace(0,w*0.02,w),np.linspace(0,h*0.02,h))
    wave=np.sin(xv*f1+ph)*np.cos(yv*f2)+np.sin(np.sqrt((xv-w*0.01)**2+(yv-h*0.01)**2)*f1)
    wave=(wave-wave.min())/(wave.max()-wave.min()+1e-9)
    n=len(pal); ca=np.array(pal,dtype=np.float32)
    idx=np.clip((wave*(n-1)).astype(int),0,n-2); frac=(wave*(n-1))-idx
    return Image.fromarray((ca[idx]*(1-frac[...,None])+ca[idx+1]*frac[...,None]).astype(np.uint8),'RGB')

def bg_radial(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    xv,yv=np.meshgrid(np.arange(w)-w//2,np.arange(h)-h//2)
    angles=np.arctan2(yv,xv); ns=rng.randint(6,24)
    idx=((angles+math.pi)/(2*math.pi)*ns).astype(int)%ns
    ca=np.array(pal,dtype=np.uint8); arr=ca[idx%len(pal)]
    dist=np.sqrt(xv**2+yv**2); dn=dist/(dist.max()+1e-9)
    return Image.fromarray((arr*(0.3+0.7*dn[...,None])).astype(np.uint8),'RGB')

def bg_fractal(size,pal,seed,f):
    rng=random.Random(seed); np.random.seed(seed%2147483647); w,h=size
    noise=np.zeros((h,w),dtype=np.float32)
    for oct_ in range(rng.randint(3,6)):
        sc=2**oct_; sh,sw=max(2,h//sc),max(2,w//sc)
        layer=Image.fromarray((np.random.rand(sh,sw)*255).astype(np.uint8),'L').resize((w,h),Image.BILINEAR)
        noise+=np.array(layer,dtype=np.float32)/255.0/(2**oct_)
    noise=(noise-noise.min())/(noise.max()-noise.min()+1e-9)
    n=len(pal); ca=np.array(pal,dtype=np.float32)
    idx=np.clip((noise*(n-1)).astype(int),0,n-2); frac=(noise*(n-1))-idx
    return Image.fromarray((ca[idx]*(1-frac[...,None])+ca[idx+1]*frac[...,None]).astype(np.uint8),'RGB')

def bg_stripes(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    a=math.radians(rng.uniform(15,75)); sw=max(10,int(f['spectral_bandwidth']/50+rng.uniform(5,50)))
    xv,yv=np.meshgrid(np.arange(w),np.arange(h))
    idx=((xv*math.cos(a)+yv*math.sin(a)).astype(int)//sw)%len(pal)
    return Image.fromarray(np.array(pal,dtype=np.uint8)[idx],'RGB')

def bg_grid(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    bw=max(20,int(f['bpm']/3)); bh=bw
    arr=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(0,h,bh):
        for x in range(0,w,bw):
            c=pal[rng.randint(0,len(pal)-1)]; br=rng.uniform(0.3,1.0)
            arr[y:min(y+bh,h),x:min(x+bw,w)]=tuple(int(v*br) for v in c)
    return Image.fromarray(arr,'RGB')

def bg_scanlines(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    arr=np.zeros((h,w,3),dtype=np.uint8)
    lh=max(2,int(f['bpm']/20))
    for y in range(0,h,lh):
        c=pal[rng.randint(0,len(pal)-1)]; br=rng.uniform(0.4,1.0)
        arr[y:y+lh,:]=tuple(int(v*br) for v in c)
    return Image.fromarray(arr,'RGB')

def bg_concentric(size,pal,seed,f):
    rng=random.Random(seed); w,h=size
    img=Image.new('RGB',size,pal[0]); draw=ImageDraw.Draw(img)
    step=max(8,int(f['spectral_bandwidth']/80))
    cx,cy=w//2+rng.randint(-w//6,w//6),h//2+rng.randint(-h//6,h//6)
    shape=rng.randint(0,1)
    for i,r in enumerate(range(0,max(w,h),step)):
        c=pal[i%len(pal)]
        if shape==0: draw.ellipse([cx-r,cy-r,cx+r,cy+r],outline=c,width=max(1,step//3))
        else:        draw.rectangle([cx-r,cy-r,cx+r,cy+r],outline=c,width=max(1,step//3))
    return img

BG_POOL = [bg_plasma,bg_voronoi,bg_gradient,bg_interference,bg_radial,
           bg_fractal,bg_stripes,bg_grid,bg_scanlines,bg_concentric]

def pick_background(size, features, seed, palette):
    rng=random.Random(seed)
    # Scelta guidata dagli MFCC — non casuale pura
    mfcc=features['mfcc_features']
    bg_idx=int(abs(mfcc[0]+mfcc[1]))%len(BG_POOL)
    # Ma il seed la modifica per varietà tra rigenerazioni
    bg_idx=(bg_idx+rng.randint(0,3))%len(BG_POOL)
    return BG_POOL[bg_idx](size,palette,seed,features)

# ═══════════════════════════════════════════
# OVERLAY
# ═══════════════════════════════════════════

def ov_channel_bleed(img,features,seed,intensity=1.0):
    rng=random.Random(seed)
    base=max(3,min(40,int(features['spectral_bandwidth']/100*intensity)))
    rm=max(2,min(30,int(features['rms']*200*intensity)))
    r,g,b=img.split()
    r=ImageChops.offset(r,int((base+rng.randint(-rm,rm))*intensity),rng.randint(-5,5))
    g=ImageChops.offset(g,rng.randint(-base//2,base//2),rng.randint(-3,3))
    b=ImageChops.offset(b,int((-base+rng.randint(-rm//2,rm//2))*intensity),rng.randint(-5,5))
    return Image.merge('RGB',(r,g,b))

def ov_corruption(img,features,seed,intensity=1.0):
    rng=random.Random(seed); np.random.seed(seed%2147483647)
    w,h=img.size; pixels=np.array(img)
    ci=max(0.05,min(0.8,features['zero_crossing_rate']*5+features['rms']*3))*intensity
    for _ in range(int(ci*30)):
        ct=rng.randint(0,3)
        if ct==0:
            bw=rng.randint(10,min(80,w//4,h//4)); bh=rng.randint(10,min(80,w//4,h//4))
            x=rng.randint(0,max(1,w-bw)); y=rng.randint(0,max(1,h-bh))
            pixels[y:min(y+bh,h),x:min(x+bw,w)]=255-pixels[y:min(y+bh,h),x:min(x+bw,w)]
        elif ct==1:
            y=rng.randint(0,h-1); lh=rng.randint(1,min(5,h-y)); shift=int(rng.randint(-50,50)*intensity)
            if shift>0 and shift<w: pixels[y:y+lh,shift:]=pixels[y:y+lh,:-shift]
            elif shift<0 and abs(shift)<w: pixels[y:y+lh,:shift]=pixels[y:y+lh,-shift:]
        elif ct==2:
            bw=rng.randint(5,min(30,w//8)); bh=rng.randint(5,min(30,h//8))
            x=rng.randint(0,max(1,w-bw)); y=rng.randint(0,max(1,h-bh))
            pixels[y:min(y+bh,h),x:min(x+bw,w)]=np.random.randint(0,255,(min(bh,h-y),min(bw,w-x),3))
        elif ct==3:
            sh=rng.randint(1,min(8,h//10)); y=rng.randint(0,max(1,h-sh))
            ty=rng.randint(0,max(1,h-sh)); pixels[ty:ty+sh]=pixels[y:y+sh]
    return Image.fromarray(pixels)

def ov_pixel_sort(img,features,seed,intensity=1.0):
    rng=random.Random(seed); w,h=img.size; pixels=np.array(img)
    si=max(0.05,min(0.9,(features['onset_strength']/10+features['percussive_energy']*5)*intensity))
    step=max(1,int(8/intensity)) if intensity>0 else 8
    if rng.random()<0.5:
        for y in range(0,h,rng.randint(1,step)):
            if rng.random()<si:
                row=pixels[y].copy()
                pixels[y]=row[np.argsort(np.mean(row,axis=1) if rng.random()<0.5 else row[:,rng.randint(0,2)])]
    else:
        for x in range(0,w,rng.randint(1,step)):
            if rng.random()<si:
                col=pixels[:,x].copy()
                pixels[:,x]=col[np.argsort(np.mean(col,axis=1) if rng.random()<0.5 else col[:,rng.randint(0,2)])]
    return Image.fromarray(pixels)

def ov_scan_interference(img,seed,intensity=1.0):
    rng=random.Random(seed); pixels=np.array(img,dtype=np.float32); h=pixels.shape[0]; w=pixels.shape[1]
    for _ in range(int(rng.randint(2,10)*intensity)):
        y0=rng.randint(0,h-1); bh=rng.randint(1,max(2,int(h*0.03)))
        y1=min(y0+bh,h); shift=int(rng.randint(-30,30)*intensity)
        if shift>0 and shift<w: pixels[y0:y1,shift:]=pixels[y0:y1,:-shift]
        elif shift<0 and abs(shift)<w: pixels[y0:y1,:shift]=pixels[y0:y1,-shift:]
        pixels[y0:y1]=np.clip(pixels[y0:y1]*rng.uniform(1.1,1.8*intensity),0,255)
    return Image.fromarray(pixels.astype(np.uint8),'RGB')

def ov_geometric(img,features,seed,palette):
    rng=random.Random(seed); w,h=img.size
    ov=Image.new('RGBA',(w,h),(0,0,0,0)); draw=ImageDraw.Draw(ov)
    for _ in range(rng.randint(3,15)):
        c=palette[rng.randint(0,len(palette)-1)]; a=rng.randint(40,150)
        st=rng.randint(0,3)
        if st==0: draw.line([(rng.randint(0,w),rng.randint(0,h)),(rng.randint(0,w),rng.randint(0,h))],fill=(*c,a),width=rng.randint(2,10))
        elif st==1:
            x1,y1=rng.randint(0,w),rng.randint(0,h)
            draw.rectangle([x1,y1,x1+rng.randint(20,200),y1+rng.randint(20,200)],outline=(*c,a),width=rng.randint(1,5))
        elif st==2: draw.polygon([(rng.randint(0,w),rng.randint(0,h)) for _ in range(3)],fill=(*c,a//2),outline=(*c,a))
        else:
            cx,cy=rng.randint(0,w),rng.randint(0,h); rx,ry=rng.randint(10,100),rng.randint(10,100)
            draw.ellipse([cx-rx,cy-ry,cx+rx,cy+ry],outline=(*c,a),width=rng.randint(1,4))
    return Image.alpha_composite(img.convert('RGBA'),ov).convert('RGB')

def ov_databend(img,seed,intensity=1.0):
    rng=random.Random(seed); pixels=np.array(img); h,w=pixels.shape[:2]
    for _ in range(int(rng.randint(2,8)*intensity)):
        sy=rng.randint(0,h-1); sh=rng.randint(2,max(3,int(h*0.05)))
        dy=(sy+rng.randint(h//4,3*h//4))%(h-sh)
        chunk=pixels[sy:sy+sh].copy()
        if rng.random()<0.5: chunk[:,:,rng.randint(0,2)]=255-chunk[:,:,rng.randint(0,2)]
        pixels[dy:dy+sh]=chunk
    return Image.fromarray(pixels,'RGB')

# ═══════════════════════════════════════════
# GENERA IMMAGINE STATICA
# ═══════════════════════════════════════════

def generate_cover(features, seed, size=(800,800), glitch_controls=None, glitch_intensity=1.0):
    if glitch_controls is None:
        glitch_controls={'pixel_sorting':True,'digital_corruption':True,
                         'chromatic_aberration':True,'black_and_white':False,
                         'distorted_lines':False,'vhs_effect':False,'bw_intensity':0.5}
    rng=random.Random(seed); np.random.seed(seed%2147483647)

    # DNA visivo dagli MFCC
    mfcc=features['mfcc_features']
    palette_seed = seed ^ int(abs(mfcc[2])*1000) ^ int(abs(mfcc[5])*100)
    palette=build_palette(features, palette_seed)

    img=pick_background(size, features, seed, palette)

    # Blend secondo background (probabilità da MFCC)
    if abs(mfcc[3])%2 < 1:
        s2=seed+rng.randint(100000,999999)
        img2=pick_background(size,features,s2,build_palette(features,s2))
        bm=int(abs(mfcc[4]))%4
        alpha=rng.uniform(0.25,0.6)
        try:
            img=[Image.blend,ImageChops.multiply,ImageChops.screen,ImageChops.add_modulo][bm](
                img if bm==0 else img, img2 if bm==0 else img2, alpha) if bm==0 else \
                [ImageChops.multiply,ImageChops.screen,ImageChops.add_modulo][bm-1](img,img2)
        except: pass

    # Effetti base — ordine determinato dagli MFCC (non identico per ogni brano)
    effect_order = list(range(3))
    rng2=random.Random(int(abs(mfcc[6])*1000)+seed)
    rng2.shuffle(effect_order)

    for idx in effect_order:
        try:
            if idx==0 and glitch_controls.get('pixel_sorting',True):
                img=ov_pixel_sort(img,features,seed+1000,glitch_intensity)
            elif idx==1 and glitch_controls.get('digital_corruption',True):
                img=ov_corruption(img,features,seed+2000,glitch_intensity)
            elif idx==2 and glitch_controls.get('chromatic_aberration',True):
                img=ov_channel_bleed(img,features,seed+3000,glitch_intensity)
        except: pass

    if rng.random()<0.7:  img=ov_databend(img,seed+4000,glitch_intensity)
    if rng.random()<0.6:  img=ov_scan_interference(img,seed+5000,glitch_intensity)
    if rng.random()<0.65: img=ov_geometric(img,features,seed+6000,palette)

    if glitch_controls.get('black_and_white',False):
        bw=glitch_controls.get('bw_intensity',0.5)
        img=Image.blend(img,img.convert('L').convert('RGB'),bw)

    # Finalizzazione per emozione
    em=features['emotion']
    if em in ('Aggressive','Ritmico'):
        img=ImageEnhance.Contrast(img).enhance(rng.uniform(1.3,1.8))
        img=ImageEnhance.Color(img).enhance(rng.uniform(1.4,2.0))
    elif em in ('Ambient','Calmo'):
        img=img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5,1.5)))
        img=ImageEnhance.Color(img).enhance(rng.uniform(0.7,1.1))
    elif em=='Energetico':
        img=ImageEnhance.Sharpness(img).enhance(rng.uniform(1.5,2.5))
        img=ImageEnhance.Color(img).enhance(rng.uniform(1.3,1.8))
    else:
        img=ImageEnhance.Contrast(img).enhance(rng.uniform(1.1,1.4))

    notes=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    desc=[
        f"🎵 BPM: {features['bpm']:.1f} → Velocità pattern",
        f"🔊 RMS: {features['rms']:.3f} → Intensità glitch",
        f"🎼 Centroide: {features['spectral_centroid']:.0f} Hz → Tipo distorsione",
        f"⚡ Percussivo: {features['percussive_energy']:.3f} → Pixel sorting",
        f"🎹 Armonico: {features['harmonic_energy']:.3f} → Aberrazione cromatica",
        f"🎸 Nota dominante: {notes[features['tonal_centroid']%12]} → Palette colori",
        f"😊 Emozione: {features['emotion']} → Stile generale",
    ]
    return img, desc, palette

# ═══════════════════════════════════════════
# GENERA VIDEO
# ═══════════════════════════════════════════

def generate_video(features, seed, size, audio_bytes, fps, duration_sec,
                   glitch_controls, glitch_intensity, progress_cb=None,
                   cover_img_bytes=None):
    # Dimensioni divisibili per 16
    w, h = make_divisible(size)
    render_size = (w, h)

    rms_curve, onset_curve, zcr_curve, actual_dur = extract_frame_curves(
        audio_bytes, fps, duration_sec
    )
    total_frames = len(rms_curve)

    mfcc = features['mfcc_features']
    palette_seed = seed ^ int(abs(mfcc[2])*1000) ^ int(abs(mfcc[5])*100)
    palette = build_palette(features, palette_seed)

    # BASE: usa la copertina generata se disponibile
    if cover_img_bytes is not None:
        try:
            base_bg = Image.open(io.BytesIO(cover_img_bytes)).convert('RGB').resize(render_size, Image.LANCZOS)
        except Exception:
            base_bg = pick_background(render_size, features, seed, palette)
    else:
        base_bg = pick_background(render_size, features, seed, palette)

    # Secondo layer per blend dinamico
    s2 = seed + 999983
    palette2 = build_palette(features, s2)
    base_bg2 = pick_background(render_size, features, s2, palette2)

    # File temporanei
    tmp_video = tempfile.NamedTemporaryFile(suffix='_noaudio.mp4', delete=False)
    tmp_final  = tempfile.NamedTemporaryFile(suffix='_final.mp4',   delete=False)
    tmp_audio  = tempfile.NamedTemporaryFile(suffix='.mp3',          delete=False)
    tmp_video.close(); tmp_final.close(); tmp_audio.close()

    # Scrivi audio su disco per moviepy
    with open(tmp_audio.name, 'wb') as f:
        f.write(audio_bytes)

    writer = imageio.get_writer(tmp_video.name, fps=fps, codec='libx264',
                                 quality=7, macro_block_size=16)

    for i in range(total_frames):
        rms_v   = float(rms_curve[i])
        onset_v = float(onset_curve[i])
        zcr_v   = float(zcr_curve[i])

        # Ogni frame ha seed unico ma determinisitco
        frame_seed = (seed + i * 7919) & 0x7FFFFFFF

        # 1. Base: blend dinamico tra i due background guidato dall'onset
        blend_alpha = 0.15 + onset_v * 0.45   # 0.15 → 0.60
        try:
            frame = Image.blend(base_bg, base_bg2, blend_alpha)
        except:
            frame = base_bg.copy()

        # 2. Aberrazione cromatica — intensità proporzionale all'onset
        if glitch_controls.get('chromatic_aberration', True):
            dyn_features = features.copy()
            dyn_features['spectral_bandwidth'] = features['spectral_bandwidth'] * (0.5 + onset_v * 2.0)
            dyn_features['rms'] = rms_v * 0.3
            try:
                frame = ov_channel_bleed(frame, dyn_features, frame_seed, intensity=0.3+onset_v*2.5*glitch_intensity)
            except: pass

        # 3. Corruzioni digitali — solo sui picchi (onset > soglia smoothed)
        if glitch_controls.get('digital_corruption', True) and onset_v > 0.55:
            dyn_f = features.copy()
            dyn_f['zero_crossing_rate'] = zcr_v * 0.5
            dyn_f['rms'] = rms_v
            try:
                frame = ov_corruption(frame, dyn_f, frame_seed+100,
                                      intensity=(onset_v-0.55)*3.0*glitch_intensity)
            except: pass

        # 4. Pixel sorting — intensità dal RMS smoothed
        if glitch_controls.get('pixel_sorting', True) and rms_v > 0.3:
            dyn_f = features.copy()
            dyn_f['onset_strength'] = onset_v * features['onset_strength']
            dyn_f['percussive_energy'] = rms_v * features['percussive_energy'] * 3
            try:
                frame = ov_pixel_sort(frame, dyn_f, frame_seed+200,
                                      intensity=rms_v*glitch_intensity*0.8)
            except: pass

        # 5. Scan interference — leggera, sempre presente ma variabile
        try:
            frame = ov_scan_interference(frame, frame_seed+300, intensity=0.1+rms_v*0.5*glitch_intensity)
        except: pass

        # 6. Databend — solo su picchi forti
        if onset_v > 0.75:
            try:
                frame = ov_databend(frame, frame_seed+400, intensity=onset_v*glitch_intensity)
            except: pass

        # 7. Saturazione e contrasto oscillano col volume
        try:
            sat_factor = 0.8 + rms_v * 1.5 * glitch_intensity
            frame = ImageEnhance.Color(frame).enhance(sat_factor)
            contrast_factor = 0.9 + onset_v * 0.8
            frame = ImageEnhance.Contrast(frame).enhance(contrast_factor)
        except: pass

        # Resize se necessario
        if frame.size != render_size:
            frame = frame.resize(render_size, Image.LANCZOS)

        writer.append_data(np.array(frame))

        if progress_cb and i % max(1, total_frames//100) == 0:
            progress_cb(i / total_frames)

    writer.close()

    # Attach audio con moviepy
    try:
        video_clip = VideoFileClip(tmp_video.name)
        audio_clip = AudioFileClip(tmp_audio.name)
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclipped(0, video_clip.duration)
        final = video_clip.with_audio(audio_clip)
        final.write_videofile(tmp_final.name, codec='libx264', audio_codec='aac',
                              logger=None, fps=fps)
        video_clip.close(); audio_clip.close()
        os.unlink(tmp_video.name)
        os.unlink(tmp_audio.name)
        return tmp_final.name
    except Exception as e:
        # Fallback: video senza audio
        os.unlink(tmp_audio.name)
        return tmp_video.name

# ═══════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════

def generate_report(features, filename, regen, glitch_controls, glitch_intensity,
                    video_info=None):
    notes=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    dn=notes[features['tonal_centroid']%12]
    rf="sussurrato" if features['rms']<0.02 else ("delicato" if features['rms']<0.05 else ("potente" if features['rms']<0.08 else "esplosivo"))
    vol_num = str(regen).zfill(2)

    video_section = ""
    if video_info:
        video_section = f"""
:: VIDEO GENERATO
    Durata          →  {video_info['duration_str']}
    Frame totali    →  {video_info['total_frames']}
    FPS             →  {video_info['fps']}
    Risoluzione     →  {video_info['resolution']}
    Audio incluso   →  SI
    Smoothing       →  Hanning window ({video_info['fps']//2+1} campioni)
    Peak sincro     →  Onset strength → corruzioni
                       RMS smoothed  → pixel sorting
                       Blend alpha   → aberrazione cromatica
"""

    return f"""[GlitchCoverStudio] // VOL_{vol_num} // H.264 // PNG

:: STILE: Glitch Generativo / Algorithmic Art
:: VARIANTE: #{regen}
:: SIGNATURE: {features['audio_signature']}

Ogni Pixel e ogni Frame sono il Riflesso di un Dato Sonoro Reale.

:: DATI AUDIO
    BPM                  →  {features['bpm']:.1f}
    RMS Energy           →  {features['rms']:.4f}  [{rf}]
    Centroide Spettrale  →  {features['spectral_centroid']:.0f} Hz
    Banda Spettrale      →  {features['spectral_bandwidth']:.0f} Hz
    Spectral Rolloff     →  {features['spectral_rolloff']:.0f} Hz
    Zero Crossing Rate   →  {features['zero_crossing_rate']:.4f}
    Energia Percussiva   →  {features['percussive_energy']:.4f}
    Energia Armonica     →  {features['harmonic_energy']:.4f}
    Onset Strength       →  {features['onset_strength']:.2f}
    Nota Dominante       →  {dn}
    Emozione             →  {features['emotion']}
{video_section}
:: Regia e Algoritmo: Loop507

#GlitchArt #CoverArt #AlgorithmicArt #AudioVisual #Glitch
#SoundDesign #ComputationalMinimalism #SignalCorruption
#NewMediaArt #DataNoise #NoiseMusic #AlgorithmicVideo
#{features['emotion']} #BPM{int(features['bpm'])} #Loop507
"""

# ═══════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════

for k,v in [('regen_count',0),('img_bytes',None),('img_format_saved','PNG'),
            ('img_emotion',''),('report_text',None),('audio_filename',''),
            ('descrizione',[]),('features_saved',None),('palette_saved',None),
            ('seed_saved',0),('video_bytes',None),('video_path',None)]:
    if k not in st.session_state: st.session_state[k]=v

# ═══════════════════════════════════════════
# UI
# ═══════════════════════════════════════════

audio_file = st.file_uploader("🎵 Carica il tuo brano (MP3, WAV, OGG)", type=["mp3","wav","ogg"])

col1,col2,col3 = st.columns(3)
with col1: img_format   = st.selectbox("Formato immagine:", ["PNG","JPEG"])
with col2: aspect_ratio = st.selectbox("Formato dimensioni:", ["Quadrato (1:1)","Verticale (9:16)","Orizzontale (16:9)"])
with col3: show_analysis= st.checkbox("Analisi avanzata", value=True)

st.sidebar.markdown("### 🎛️ Controlli Glitch")
glitch_intensity = st.sidebar.slider("Intensita Glitch", 0.1, 2.0, 1.0, 0.1)
st.sidebar.markdown("#### Effetti")
use_ps  = st.sidebar.checkbox("Pixel Sorting", value=True)
use_dc  = st.sidebar.checkbox("Corruzioni Digitali", value=True)
use_ca  = st.sidebar.checkbox("Aberrazione Cromatica", value=True)
use_bw  = st.sidebar.checkbox("Bianco e Nero", value=False)
bw_int  = st.sidebar.slider("Intensita B&N", 0.1, 1.0, 0.5, 0.1) if use_bw else 0.5

glitch_controls = {'pixel_sorting':use_ps,'digital_corruption':use_dc,
                   'chromatic_aberration':use_ca,'black_and_white':use_bw,'bw_intensity':bw_int}

# ── VIDEO SETTINGS (sidebar) ──────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎬 Impostazioni Video")
fps_choice = st.sidebar.selectbox("FPS:", [24, 30], index=0)
dur_mode   = st.sidebar.radio("Durata video:", ["Brano completo", "Personalizzata"], index=0)
if dur_mode == "Personalizzata":
    dur_choice = st.sidebar.slider("Durata (sec)", 10, 240, 30, 5)
else:
    dur_choice = 0  # 0 = usa durata reale del brano (calcolata dopo l'analisi)
# FPS adattivi automatici per video lunghi
effective_fps = fps_choice if (dur_choice == 0 or dur_choice <= 120) else 15
if dur_choice > 120:
    st.sidebar.info("⚠️ Oltre 2 min: FPS ridotti a 15 automaticamente")
if dur_choice == 0:
    st.sidebar.caption("Durata = lunghezza reale del brano")
else:
    st.sidebar.caption(f"~{dur_choice*effective_fps} frame · stima {dur_choice*effective_fps//60 + 1}-{dur_choice*effective_fps//40 + 2} min")

if audio_file:
    audio_bytes = audio_file.read()
    with st.spinner("🔍 Analizzando caratteristiche audio..."):
        features = analyze_audio_features(audio_bytes)

    if features:
        col_main, col_side = st.columns([2,1])

        with col_main:
            # ── GENERA COPERTINA ──────────────────────
            if st.button("🎨 Genera Copertina Glitch", key="gen_cover"):
                st.session_state.regen_count += 1
                mf = features.copy()
                mf['rms'] *= glitch_intensity
                mf['percussive_energy'] *= glitch_intensity
                mf['onset_strength'] *= glitch_intensity

                seed = (features['file_hash'] + st.session_state.regen_count*99991 +
                        int(time.time()*1000)%100003) & 0x7FFFFFFF
                st.session_state.seed_saved = seed

                size = get_dimensions(aspect_ratio)
                with st.spinner("🎨 Generando copertina..."):
                    t0 = time.time()
                    img, desc, palette = generate_cover(mf, seed, size, glitch_controls, glitch_intensity)
                    gen_time = time.time()-t0

                buf = io.BytesIO()
                if img_format=="PNG": img.save(buf,format='PNG')
                else: img.save(buf,format='JPEG',quality=95)
                buf.seek(0)

                st.session_state.img_bytes        = buf.getvalue()
                st.session_state.img_format_saved = img_format
                st.session_state.img_emotion      = features['emotion']
                st.session_state.audio_filename   = audio_file.name
                st.session_state.descrizione      = desc
                st.session_state.features_saved   = features
                st.session_state.palette_saved    = palette
                st.session_state.video_bytes      = None  # reset video precedente
                st.session_state.report_text      = generate_report(
                    features, audio_file.name, st.session_state.regen_count,
                    glitch_controls, glitch_intensity)

                st.success(f"✅ Copertina generata in {gen_time:.1f}s")

        with col_side:
            st.markdown("### 🎯 Varianti")
            st.markdown(f"**N°** {st.session_state.regen_count}")
            if st.button("🔄 Nuova Variante"):
                st.session_state.regen_count += 1
                st.rerun()
            if st.button("🔄 Reset"):
                for k in ['regen_count','img_bytes','report_text','video_bytes','video_path']:
                    st.session_state[k] = 0 if k=='regen_count' else None
                st.rerun()

        # ── ANALISI ──────────────────────────────────
        if show_analysis and st.session_state.descrizione:
            st.markdown("### 🔍 Analisi Audio")
            for d in st.session_state.descrizione: st.markdown(f"- {d}")
            f=st.session_state.features_saved or features
            ca,cb=st.columns(2)
            with ca:
                st.metric("🎵 BPM",f"{f['bpm']:.1f}")
                st.metric("🔊 RMS",f"{f['rms']:.3f}")
                st.metric("🎼 Centroide",f"{f['spectral_centroid']:.0f} Hz")
                st.metric("😊 Emozione",f['emotion'])
            with cb:
                st.metric("⚡ Percussivo",f"{f['percussive_energy']:.3f}")
                st.metric("🎹 Armonico",f"{f['harmonic_energy']:.3f}")
                st.metric("🌊 ZCR",f"{f['zero_crossing_rate']:.3f}")
                st.metric("📊 Onset",f"{f['onset_strength']:.1f}")

        # ── ANTEPRIMA COPERTINA + GENERA VIDEO ───────
        if st.session_state.img_bytes is not None:
            st.divider()
            st.markdown("### 🖼️ Copertina")
            st.image(st.session_state.img_bytes,
                     caption=f"#{st.session_state.regen_count} — {st.session_state.img_emotion}")

            # Bottone genera video — sotto la copertina
            st.markdown("### 🎬 Genera Video da questa Copertina")
            # Durata effettiva: 0 = brano completo (calcolato da librosa)
            real_dur = float(dur_choice) if dur_choice > 0 else None
            # Stima frame per info box (approssimazione se brano completo)
            est_sec  = dur_choice if dur_choice > 0 else 180
            eff_fps  = fps_choice if est_sec <= 120 else 15
            total_frames_est = est_sec * eff_fps
            min_est = total_frames_est // 60 + 1
            max_est = total_frames_est // 40 + 2
            if dur_choice == 0:
                st.info(f"⏱️ Durata: brano completo · {eff_fps}fps · stima {min_est}-{max_est} min (varia con la durata del brano)")
            else:
                dur_str = f"{dur_choice//60}:{dur_choice%60:02d}" if dur_choice >= 60 else f"{dur_choice}s"
                st.info(f"⏱️ Durata: {dur_str} · {total_frames_est} frame @ {eff_fps}fps · stima {min_est}-{max_est} min")

            if st.button("🎬 Genera Video Glitch", key="gen_video"):
                prog_bar = st.progress(0, text="Generando frame...")
                status   = st.empty()

                def update_progress(p):
                    pct = int(p*100)
                    prog_bar.progress(pct, text=f"Frame {pct}%")

                seed = st.session_state.seed_saved
                size = get_dimensions(aspect_ratio)
                mf   = (st.session_state.features_saved or features).copy()
                mf['rms'] *= glitch_intensity
                mf['percussive_energy'] *= glitch_intensity
                mf['onset_strength'] *= glitch_intensity

                # Durata reale: se brano completo, passa 0 → extract_frame_curves carica tutto
                actual_dur_sec = real_dur if real_dur is not None else 0.0
                actual_eff_fps = fps_choice if (actual_dur_sec == 0 or actual_dur_sec <= 120) else 15

                try:
                    t0 = time.time()
                    video_path = generate_video(
                        mf, seed, size, audio_bytes,
                        fps=actual_eff_fps,
                        duration_sec=actual_dur_sec,
                        glitch_controls=glitch_controls,
                        glitch_intensity=glitch_intensity,
                        progress_cb=update_progress,
                        cover_img_bytes=st.session_state.img_bytes
                    )
                    prog_bar.progress(100, text="Finalizzando...")
                    gen_time = time.time()-t0

                    with open(video_path,'rb') as vf:
                        st.session_state.video_bytes = vf.read()
                    st.session_state.video_path = video_path

                    # Aggiorna report con info video
                    w,h = make_divisible(get_dimensions(aspect_ratio))
                    used_dur = actual_dur_sec if actual_dur_sec > 0 else est_sec
                    used_frames = int(used_dur * actual_eff_fps)
                    mins = int(used_dur//60); secs_r = int(used_dur%60)
                    st.session_state.report_text = generate_report(
                        st.session_state.features_saved or features,
                        audio_file.name, st.session_state.regen_count,
                        glitch_controls, glitch_intensity,
                        video_info=dict(
                            duration_str=f"{mins}:{secs_r:02d}",
                            total_frames=used_frames,
                            fps=actual_eff_fps,
                            resolution=f"{w}x{h}"
                        )
                    )
                    status.success(f"✅ Video generato in {gen_time/60:.1f} minuti!")
                except Exception as e:
                    status.error(f"❌ Errore generazione video: {e}")

            # ── DOWNLOAD ZONE — fuori da tutti i button ──
            st.divider()
            st.markdown("### 💾 Download")

            ext  = "png" if st.session_state.img_format_saved=="PNG" else "jpg"
            mime = "image/png" if st.session_state.img_format_saved=="PNG" else "image/jpeg"
            base = f"glitch_{st.session_state.regen_count}_{st.session_state.img_emotion.lower()}"

            d1,d2,d3 = st.columns(3)
            with d1:
                st.download_button(
                    "🖼️ Copertina", data=st.session_state.img_bytes,
                    file_name=f"{base}.{ext}", mime=mime,
                    use_container_width=True, key="dl_img"
                )
            with d2:
                if st.session_state.video_bytes:
                    st.download_button(
                        "🎬 Video MP4", data=st.session_state.video_bytes,
                        file_name=f"{base}_video.mp4", mime="video/mp4",
                        use_container_width=True, key="dl_video"
                    )
                else:
                    st.button("🎬 Video MP4", disabled=True,
                              use_container_width=True, help="Genera prima il video")
            with d3:
                if st.session_state.report_text:
                    st.download_button(
                        "📄 Report .txt", data=st.session_state.report_text.encode("utf-8"),
                        file_name=f"report_{base}.txt", mime="text/plain",
                        use_container_width=True, key="dl_report"
                    )

    else:
        st.error("❌ Impossibile analizzare il file audio.")
else:
    st.info("👆 Carica un file audio per iniziare")
    st.markdown("""
**Flusso:**
1. Carica il brano
2. Genera la copertina (pochi secondi) — rigenerala fino a che ti piace
3. Clicca **Genera Video** — usa quella copertina come base
4. Scarica copertina, video MP4 con audio, e report

**Il video è sincronizzato all'audio:**
- Aberrazione cromatica → segue l'onset strength
- Corruzioni digitali → esplodono sui picchi
- Pixel sorting → segue il volume (RMS smoothed)
- Saturazione → oscilla col volume
""")

st.markdown("---")
st.markdown("**🔧 Loop507 Studio** · Arte algoritmica da analisi audio reale")
