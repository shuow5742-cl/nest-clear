"""Deterministic scenes shared by the gen1 exporter and tests."""
import cv2
import numpy as np
H, W = 1536, 2048
BARS = {'empty': [], 'center_bar': [(1200,1600,740,780)], 'two_bars': [(1200,1600,740,780),(900,1100,1000,1030)], 'right_edge': [(W-100,W,740,780)], 'big_blob': [(1000,1500,600,1000)]}
COMPLEX = ('noisy_curves','pale_lines','dashed','collinear_cores','fiber_spray','hair_with_tails','edge_curves','mixed_fragments')
SCENES = tuple(BARS) + COMPLEX

def frame_for(name):
    if name in BARS:
        f = np.full((H,W,3),200,np.uint8)
        for x0,x1,y0,y1 in BARS[name]: f[y0:y1,x0:x1] = 20
        return f
    rng = np.random.default_rng(4263 + COMPLEX.index(name))
    gray = np.clip(np.linspace(165,210,W,dtype=np.float32)[None,:] + rng.normal(0,2,(H,W)),0,255).astype(np.uint8)
    f = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    def line(points, val, width): cv2.polylines(f,[np.asarray(points,np.int32)],False,(val,)*3,width,cv2.LINE_8)
    if name == 'noisy_curves':
        for i,w in enumerate((2,4,6,8)): line([(300+x,300+i*200+int(42*np.sin(x/85))) for x in range(0,900,4)],25,w)
    elif name == 'pale_lines':
        for i,v in enumerate((60,80,100,120)): line([(240+x,300+i*240+int(24*np.sin(x/100))) for x in range(0,1200,5)],v,7)
    elif name == 'dashed':
        for i in range(20): cv2.circle(f,(400+i*42,700+int(15*np.sin(i/3))),9,(20,)*3,-1)
    elif name == 'collinear_cores':
        for i in range(12): cv2.ellipse(f,(400+i*75,650+i*8),(24,11),6,0,360,(25,)*3,-1)
    elif name == 'fiber_spray':
        for _ in range(130):
            x,y = rng.integers(600,1300),rng.integers(500,1000)
            line([(x,y),(x+int(rng.integers(-90,91)),y+int(rng.integers(-90,91)))],int(rng.integers(55,120)),int(rng.integers(2,5)))
    elif name == 'hair_with_tails':
        cv2.ellipse(f,(1024,760),(185,48),12,0,360,(25,)*3,-1)
        line([(200,600),(840,725)],70,5); line([(1200,795),(1850,960)],75,5)
    elif name == 'edge_curves':
        line([(x,20+int(18*np.sin(x/60))) for x in range(1000)],35,7)
        line([(2030-int(18*np.sin(y/60)),y) for y in range(500,1500)],65,6)
    else:
        for i in range(9): cv2.ellipse(f,(350+i*100,850+i*8),(30,14),5,0,360,(25,)*3,-1)
        for i in range(8): line([(500+i*30,400),(700+i*30,560)],75,4)
        cv2.ellipse(f,(1500,900),(100,30),30,0,360,(20,)*3,-1)
    return f
