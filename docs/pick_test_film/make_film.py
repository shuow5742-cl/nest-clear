#!/usr/bin/env python3
import csv, subprocess, os, sys
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
a4 = '--a4' in sys.argv
P = '燕窝二代_挑取测试菲林片_A4打印版_20261123.pdf' if a4 else '燕窝二代_挑取精度测试菲林片_V2_245x120.pdf'
pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
subprocess.run(['/home/shuow/nest-clear/.venv/bin/python','-c', '''import cv2, json
D=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
json.dump([(cv2.aruco.generateImageMarker(D,i,6)==0).astype(int).tolist() for i in range(32)],open('/tmp/nest_film_markers.json','w'))'''],check=True)
import json
markers=json.load(open('/tmp/nest_film_markers.json'))
page_w, page_h = (297, 210) if a4 else (265, 140)
c=canvas.Canvas(P,pagesize=(page_w*mm,page_h*mm),pageCompression=1)
def pt(x,y): return ((x+page_w/2)*mm,(y+page_h/2)*mm)
def line(x1,y1,x2,y2,w):
 c.setLineWidth(w*mm); c.line(*pt(x1,y1),*pt(x2,y2))
def text(x,y,s,size,font='CJK'):
 c.setFont(font,size*mm); c.drawString(*pt(x,y),s)
c.setStrokeGray(.5); c.setLineWidth(.1*mm); c.rect((page_w/2-122.5)*mm,(page_h/2-60)*mm,245*mm,120*mm,stroke=1,fill=0); c.setStrokeGray(0); c.setFillGray(0)
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
line(-60,56,-40,56,.3); line(-40,56,-42,57,.3); line(-40,56,-42,55,.3); text(-52,57,'Y+',1.5,'Helvetica')
line(-118,40,-118,20,.3); line(-118,20,-119,22,.3); line(-118,20,-117,22,.3); text(-115.5,36,'X+',1.5,'Helvetica')
header='此面朝上 · 按箭头对齐设备 X+/Y+ · 挑取精度测试片 V2 2026-11-23'
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
if a4:
 # Page coordinates in mm, outside the 245 x 120 pattern.
 def ruler(x, y, length, vertical=False):
  c.setStrokeGray(0); c.setLineWidth(.12*mm)
  if vertical: c.line(x*mm,y*mm,x*mm,(y+length)*mm)
  else: c.line(x*mm,y*mm,(x+length)*mm,y*mm)
  for i in range(length+1):
   tick = 2.5 if i % 10 == 0 else 1.6 if i % 5 == 0 else .8
   if vertical: c.line(x*mm,(y+i)*mm,(x+tick)*mm,(y+i)*mm)
   else: c.line((x+i)*mm,y*mm,(x+i)*mm,(y+tick)*mm)
   if i % 10 == 0:
    c.setFont('Helvetica',1.5*mm)
    if vertical: c.drawString((x+3)*mm,(y+i-.5)*mm,str(i))
    else: c.drawCentredString((x+i)*mm,(y-2.5)*mm,str(i))
  c.setFont('Helvetica',2*mm)
  if vertical: c.drawString((x-1)*mm,(y+length+3)*mm,f'Y {length} mm')
  else: c.drawCentredString((x+length/2)*mm,(y+5)*mm,f'X {length} mm')
 ruler(48, 23, 100); ruler(183, 23, 50)
 ruler(12, 55, 100, True); ruler(278, 80, 50, True)
 instruction="打印时选择'实际大小/100%'，不要'适合页面'；打印后用钢尺量校验尺 100mm 应为 100.0±0.3mm，不符请调打印设置。贴纸只贴图案外四边，保持平整，不得被针尖压动。"
 parts=[('Helvetica' if latin else 'CJK', ''.join(group)) for latin,group in itertools.groupby(instruction,key=lambda ch: ord(ch)<128 or ch=='±')]
 font_size=1.75*mm
 total=sum(pdfmetrics.stringWidth(s,font,font_size) for font,s in parts)
 assert total < 287*mm
 t=c.beginText((297*mm-total)/2,190*mm)
 for font,s in parts: t.setFont(font,font_size); t.textOut(s)
 c.drawText(t)
 c.setFont('Helvetica',2*mm); c.drawCentredString(148.5*mm,180*mm,'A4 print version 20261123')
c.showPage(); c.save()
if a4: sys.exit(0)
with open('targets.csv','w',newline='',encoding='utf-8') as f:
 w=csv.writer(f); w.writerow(['index','label','x_mm','y_mm','aruco_id','aruco_center_x_mm','aruco_center_y_mm']); w.writerows(rows)
open('说明_给厂家.txt','w',encoding='utf-8').write('''燕窝二代挑取精度测试菲林片 V2  2026-11-23
成品尺寸：245 × 120 mm。PDF 页面 265 × 140 mm，四周各 10 mm 白边。
材质：白色哑光 PET，目标厚度 0.188 mm；可接受 0.125–0.25 mm，请注明实际厚度。
印刷：黑色哑光油墨，按 PDF 1:1 输出，不得缩放；按灰色裁切线裁切，裁切公差 ±0.2 mm。
位置精度：≥200 mm 范围内 ±0.05 mm；线宽 ≥0.1 mm 必须完整清晰。
表面不得反光、起皱。数量 3 张，单独包装，平放不卷曲。
坐标原点为成品中心，X 向右，Y 向上。左上角三角形顶点为 (-120,55)、(-115,55)、(-120,50) mm；设备 Y+ 箭头为 (-60,56) 至 (-40,56)，设备 X+ 箭头为 (-118,40) 至 (-118,20)。顶边文字居中位置为 (30,55.5) mm（文字基线）。
''')
