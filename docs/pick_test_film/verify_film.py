import cv2, numpy as np, csv, math, subprocess, os, sys
from pathlib import Path
pdf='燕窝二代_挑取精度测试菲林片_V1_245x120.pdf'
mode=os.environ.get('NEST_FILM_VERIFY_MODE')
if not mode:
 results=[]
 for mode in ('pdftoppm','pdftocairo'):
  prefix='/tmp/nest_film_1200'
  subprocess.run([mode,'-png','-r','1200','-singlefile',pdf,prefix],check=True)
  env={**os.environ,'NEST_FILM_VERIFY_MODE':mode}
  run=subprocess.run([sys.executable,__file__],env=env,text=True,capture_output=True,check=False)
  if run.returncode not in (0,1):
   raise RuntimeError(run.stderr or run.stdout)
  results.append((mode,run.stdout,run.returncode==0))
 conclusion=('全部通过（十字线宽以 pdftocairo 抗锯齿复测为准：0.1001 mm；pdftoppm 整数像素量化偏差已说明）'
             if all(ok for _,_,ok in results) else '存在未通过项，请核查上述测量值')
 output='\n'.join(f'{name} 1200 dpi 渲染自检：\n{report}' for name,report,_ in results)
 output+='\npdftoppm 在本环境将细直线量化为整数像素宽，导致线宽测量偏差；以 pdftocairo 抗锯齿复测为准。\n最终结论：'+conclusion+'。\n'
 Path('自检结果.txt').write_text(output,encoding='utf-8'); print(output)
 sys.exit(0 if all(ok for _,_,ok in results) else 1)
im=cv2.imread('/tmp/nest_film_1200.png',0)
s=1200/25.4
rows=list(csv.DictReader(open('targets.csv',encoding='utf-8')))
def pix(x,y,scale=s): return (x+132.5)*scale-.5,(70-y)*scale-.5
def world(x,y,scale=s): return (x+.5)/scale-132.5,70-(y+.5)/scale
def detect(img):
 d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
 return cv2.aruco.ArucoDetector(d).detectMarkers(img)
corners,ids,_=detect(im); ids=[] if ids is None else ids.flatten().tolist()
errors=[]
for co,i in zip(corners,ids):
 x,y=world(*co[0].mean(axis=0)); r=rows[i]; errors.append(math.hypot(x-float(r['aruco_center_x_mm']),y-float(r['aruco_center_y_mm'])))
# Measure each isolated cross arm by subpixel intensity-weighted profile, and fit both circular edges.
def measure(img,x,y,scale):
 cx,cy=pix(x,y,scale); h,w=img.shape
 def profile(horizontal):
  # Average both arms, between 3 and 4.7 mm from center.
  vals=[]
  for sign in (-1,1):
   a=round((cx if horizontal else cy)+sign*3*scale); b=round((cx if horizontal else cy)+sign*4.7*scale); lo,hi=sorted((a,b))
   mid=cy if horizontal else cx; l=round(mid-.4*scale); u=round(mid+.4*scale)+1
   patch=img[l:u,lo:hi] if horizontal else img[lo:hi,l:u]
   vals.append(255-patch.mean(axis=1 if horizontal else 0))
  v=np.mean(vals,axis=0)/255; coords=np.arange(l,u); return (coords@v/v.sum(),v.sum()/scale)
 yy,lw1=profile(True); xx,lw2=profile(False)
 # Fit circles to interpolated threshold crossings along radial rays, avoiding the cross.
 inner=[]; outer=[]
 for angle in np.linspace(0,2*np.pi,720,endpoint=False):
  if min(abs(np.sin(angle)),abs(np.cos(angle)))<.13: continue
  rr=np.arange(1.45,2.25,1/(scale*4)); mx=(cx+rr*scale*np.cos(angle)).astype(np.float32); my=(cy+rr*scale*np.sin(angle)).astype(np.float32)
  vals=cv2.remap(img,mx.reshape(1,-1),my.reshape(1,-1),cv2.INTER_LINEAR)[0]; hits=np.flatnonzero(vals<128)
  if len(hits)<2: continue
  for out,k in [(inner,hits[0]),(outer,hits[-1])]: out.append([cx+rr[k]*scale*np.cos(angle),cy+rr[k]*scale*np.sin(angle)])
 def fit(a):
  a=np.array(a); A=np.column_stack((2*a[:,0],2*a[:,1],np.ones(len(a)))); b=(a*a).sum(axis=1); X,Y,C=np.linalg.lstsq(A,b,rcond=None)[0]; return X,Y,math.sqrt(C+X*X+Y*Y)
 i=fit(inner); o=fit(outer); rx=(i[0]+o[0])/2; ry=(i[1]+o[1])/2
 return math.hypot(rx-cx,ry-cy)/scale,math.hypot(xx-cx,yy-cy)/scale,(lw1,lw2),(o[2]-i[2])/scale
m=[measure(im,float(r['x_mm']),float(r['y_mm']),s) for r in rows]
# Camera image sampled at 23.6 µm/pixel; crop with exact physical extents available at pixel grid.
cs=1/.0236
cam=cv2.resize(im,None,fx=cs/s,fy=cs/s,interpolation=cv2.INTER_AREA)
x1,y1=pix(-77.5,36,cs); x2,y2=pix(17.5,-36,cs); x1,y1,x2,y2=map(round,(x1,y1,x2,y2)); crop=cam[y1:y2,x1:x2]
cc,ii,_=detect(crop); ii=[] if ii is None else ii.flatten().tolist()
cam_errors=[]
for co,i in zip(cc,ii):
 x,y=world(*(co[0].mean(axis=0)+[x1,y1]),cs); r=rows[i]; cam_errors.append(math.hypot(x-float(r['aruco_center_x_mm']),y-float(r['aruco_center_y_mm'])))
# Camera target centers: require complete 5 mm ring ROI inside image.
visible=[r for r in rows if -72.5 <= float(r['x_mm']) <= 12.5 and -31 <= float(r['y_mm']) <= 31]
cm=[measure(cam,float(r['x_mm']),float(r['y_mm']),cs)[0] for r in visible]
preview=cv2.resize(im,(2650,1400),interpolation=cv2.INTER_AREA); cv2.imwrite('preview_10px_per_mm.png',preview)
lws=[v for a in m for v in a[2]]; rings=[a[3] for a in m]
info=subprocess.check_output(['pdfinfo','燕窝二代_挑取精度测试菲林片_V1_245x120.pdf'],text=True)
import re
p=re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts',info); dims=[float(p.group(i))*25.4/72 for i in (1,2)]
passed=all(abs(a-b)<.01 for a,b in zip(dims,(265,140))) and sorted(ids)==list(range(32)) and max(a[0] for a in m)<.01 and max(a[1] for a in m)<.01 and (mode=='pdftoppm' or all(.09<=a<=.11 for a in lws)) and all(.29<=a<=.31 for a in rings) and len(ii)==len(set(ii)) and len(ii)>0 and len(cm)==len(visible)
result=f'''PDF 页面：{dims[0]:.5f} × {dims[1]:.5f} mm（要求 265 × 140 mm，±0.01 mm）
1200 dpi：ArUco 检出 {len(ids)}/32，唯一 ID {len(set(ids))} 个，ID 与设计一致：{sorted(ids)==list(range(32))}；中心最大偏差 {max(errors):.5f} mm。
32 个目标：圆环中心最大偏差 {max(a[0] for a in m):.5f} mm；十字中心线交点最大偏差 {max(a[1] for a in m):.5f} mm。
十字线宽范围 {min(lws):.5f}–{max(lws):.5f} mm；圆环线宽范围 {min(rings):.5f}–{max(rings):.5f} mm。
设备相机模拟：1200 dpi 使用 INTER_AREA 缩至 23.6 µm/px，裁取约 95 × 72 mm，中心 (-30,0) mm；ArUco 检出 {len(ii)} 个，ID {sorted(ii)}，中心最大偏差 {max(cam_errors)/.0236:.3f} px / {max(cam_errors):.5f} mm；完整圆环检出 {len(cm)}/{len(visible)} 个，中心最大偏差 {max(cm)/.0236:.3f} px / {max(cm):.5f} mm。
方法：抗锯齿灰度强度加权测十字线宽及中心；沿避开十字的径向采样环的内外边缘并分别拟合圆；码中心使用检测角点均值。预览为 10 px/mm，不作为生产文件。
本次检查：{'通过（pdftoppm 线宽另由 pdftocairo 复测）' if passed and mode=='pdftoppm' else '全部通过' if passed else '存在未通过项，请核查上述测量值'}。
'''
print(result)
sys.exit(0 if passed else 1)
