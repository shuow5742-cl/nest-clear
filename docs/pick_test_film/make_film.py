#!/usr/bin/env python3
import csv, subprocess, os
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
P='燕窝二代_挑取精度测试菲林片_V1_245x120.pdf'
pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
subprocess.run(['/home/shuow/nest-clear/.venv/bin/python','-c', '''import cv2, json
D=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
json.dump([(cv2.aruco.generateImageMarker(D,i,6)==0).astype(int).tolist() for i in range(32)],open('/tmp/nest_film_markers.json','w'))'''],check=True)
import json
markers=json.load(open('/tmp/nest_film_markers.json'))
c=canvas.Canvas(P,pagesize=(265*mm,140*mm),pageCompression=1)
def pt(x,y): return ((x+132.5)*mm,(y+70)*mm)
def line(x1,y1,x2,y2,w):
 c.setLineWidth(w*mm); c.line(*pt(x1,y1),*pt(x2,y2))
def text(x,y,s,size,font='CJK'):
 c.setFont(font,size*mm); c.drawString(*pt(x,y),s)
c.setStrokeGray(.5); c.setLineWidth(.1*mm); c.rect(10*mm,10*mm,245*mm,120*mm,stroke=1,fill=0); c.setStrokeGray(0); c.setFillGray(0)
rows=[]
for row,y in enumerate([45,15,-15,-45]):
 for col,x in enumerate([-105,-75,-45,-15,15,45,75,105]):
  i=row*8+col
  c.setLineWidth(.3*mm); c.circle(*pt(x,y),1.85*mm,stroke=1,fill=0)
  line(x-5,y,x+5,y,.1); line(x,y-5,x,y+5,.1)
  for r in range(6):
   for q in range(6):
    if markers[i][r][q]:
     a,b=pt(x+5+q*4/6,y+9-(r+1)*4/6); c.rect(a,b,4/6*mm,4/6*mm,stroke=0,fill=1)
  text(x-7,y-7.5,f'T{i:02}',2,'Helvetica'); text(x-7,y-9.2,f'({x},{y})',1.2,'Helvetica')
  rows.append([i,f'T{i:02}',x,y,i,x+7,y+7])
line(-3,0,3,0,.1); line(0,-3,0,3,.1); text(3.5,-2,'O(0,0)',1.5,'Helvetica')
p=c.beginPath(); p.moveTo(*pt(-120,55)); p.lineTo(*pt(-115,55)); p.lineTo(*pt(-120,50)); p.close(); c.drawPath(p,stroke=0,fill=1)
line(-60,56,-40,56,.3); line(-40,56,-42,57,.3); line(-40,56,-42,55,.3); text(-52,57,'X+',1.5,'Helvetica')
line(-118,20,-118,40,.3); line(-118,40,-119,38,.3); line(-118,40,-117,38,.3); text(-115.5,36,'Y+',1.5,'Helvetica')
header='此面朝上 · X+ 箭头指向设备 X+ 方向 · 燕窝二代挑取精度测试片 V1 2026-11-18'
# Droid Sans Fallback lacks Latin glyphs; use Helvetica for Latin and symbols.
import itertools
runs=[]
for latin, group in itertools.groupby(header, key=lambda ch: ord(ch)<128 or ch=='·'):
 runs.append(('Helvetica' if latin else 'CJK', ''.join(group)))
size=2.5*mm
width=sum(pdfmetrics.stringWidth(s,font,size) for font,s in runs)
t=c.beginText(*pt(30,55.5)); t.moveCursor(-width/2,0)
for font,s in runs:
 t.setFont(font,size); t.textOut(s)
c.drawText(t)
c.showPage(); c.save()
with open('targets.csv','w',newline='',encoding='utf-8') as f:
 w=csv.writer(f); w.writerow(['index','label','x_mm','y_mm','aruco_id','aruco_center_x_mm','aruco_center_y_mm']); w.writerows(rows)
open('说明_给厂家.txt','w',encoding='utf-8').write('''燕窝二代挑取精度测试菲林片 V1  2026-11-18
成品尺寸：245 × 120 mm。PDF 页面 265 × 140 mm，四周各 10 mm 白边。
材质：白色哑光 PET，目标厚度 0.188 mm；可接受 0.125–0.25 mm，请注明实际厚度。
印刷：黑色哑光油墨，按 PDF 1:1 输出，不得缩放；按灰色裁切线裁切，裁切公差 ±0.2 mm。
位置精度：≥200 mm 范围内 ±0.05 mm；线宽 ≥0.1 mm 必须完整清晰。
表面不得反光、起皱。数量 3 张，单独包装，平放不卷曲。
坐标原点为成品中心，X 向右，Y 向上。左上角三角形顶点为 (-120,55)、(-115,55)、(-120,50) mm；X+ 箭头为 (-60,56) 至 (-40,56)，Y+ 箭头为 (-118,20) 至 (-118,40)。顶边文字居中位置为 (30,55.5) mm（文字基线）。
''')
