#!/usr/bin/env python3
from pathlib import Path
import base64
import xml.etree.ElementTree as ET

D=Path(__file__).resolve().parent
OUT=D/"v70_action_selection_architecture.drawio"
IMG=D/"assets"/"v70_observation_montage.jpg"
W,H=2760,1480
C={"ink":"#142B43","muted":"#607084","blue":"#3979AA","blue2":"#7CA9C8","bluef":"#E9F2F8",
"purple":"#6C5FA4","purple2":"#9C8FC4","purplef":"#F1EDF8","teal":"#0C858A","teald":"#075B62",
"deep":"#063E46","tealf":"#E4F4F3","orange":"#E77F2D","orange2":"#F0A25F","orangef":"#FFF0E2",
"red":"#D55259","redf":"#FDECEF","green":"#3E9562","green2":"#75B58B","greenf":"#E7F4EB",
"gold":"#D5A423","goldf":"#FFF5D3","gray":"#8994A3","grayf":"#F2F4F6",
"wall":"#8493A1","floor":"#F6F3ED","wood":"#B98F6C","white":"#FFFFFF"}

def ts(n=20,col=None,b=False,a="center"):
 return f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;fontFamily=Arial;fontSize={n};fontColor={col or C['ink']};fontStyle={1 if b else 0};align={a};verticalAlign=middle;"
def bs(fill,stroke,w=2,rounded=True,shadow=False,extra=""):
 return f"rounded={1 if rounded else 0};arcSize=16;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={w};fontFamily=Arial;fontColor={C['ink']};fontSize=20;shadow={1 if shadow else 0};{extra}"
def es(col,w=4,dash=False,extra=""):
 return f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={col};strokeWidth={w};dashed={1 if dash else 0};dashPattern=9 7;endArrow=blockThin;endFill=1;endSize=11;{extra}"

mf=ET.Element("mxfile",{"host":"drawio","version":"30.2.4","compressed":"false","agent":"Codex drawio-skill"})
dg=ET.SubElement(mf,"diagram",{"id":"v70-dense","name":"v70 Dense Overview"})
gm=ET.SubElement(dg,"mxGraphModel",{"dx":str(W),"dy":str(H),"grid":"1","gridSize":"10","guides":"1","page":"1","pageScale":"1","pageWidth":str(W),"pageHeight":str(H),"background":"#FFFFFF"})
rt=ET.SubElement(gm,"root");ET.SubElement(rt,"mxCell",{"id":"0"});ET.SubElement(rt,"mxCell",{"id":"1","parent":"0"})
q=2
def nid(p="n"):
 global q
 z=f"{p}{q}";q+=1;return z
def v(val,x,y,w,h,style,cid=None):
 z=cid or nid();c=ET.SubElement(rt,"mxCell",{"id":z,"value":val,"style":style,"vertex":"1","parent":"1"});ET.SubElement(c,"mxGeometry",{"x":str(x),"y":str(y),"width":str(w),"height":str(h),"as":"geometry"});return z
def e(src,tgt,style,pts=None):
 c=ET.SubElement(rt,"mxCell",{"id":nid("e"),"value":"","style":style,"edge":"1","parent":"1","source":src,"target":tgt});g=ET.SubElement(c,"mxGeometry",{"relative":"1","as":"geometry"})
 if pts:
  a=ET.SubElement(g,"Array",{"as":"points"})
  for x,y in pts:ET.SubElement(a,"mxPoint",{"x":str(x),"y":str(y)})
def fe(a,b,style,pts=None):
 c=ET.SubElement(rt,"mxCell",{"id":nid("d"),"value":"","style":style,"edge":"1","parent":"1"});g=ET.SubElement(c,"mxGeometry",{"relative":"1","as":"geometry"});ET.SubElement(g,"mxPoint",{"x":str(a[0]),"y":str(a[1]),"as":"sourcePoint"});ET.SubElement(g,"mxPoint",{"x":str(b[0]),"y":str(b[1]),"as":"targetPoint"})
 if pts:
  ar=ET.SubElement(g,"Array",{"as":"points"})
  for x,y in pts:ET.SubElement(ar,"mxPoint",{"x":str(x),"y":str(y)})
def pill(val,x,y,w,h,fill,stroke,n=16,col=None,b=True,cid=None):return v(val,x,y,w,h,bs(fill,stroke,1,True,False,f"arcSize=50;fontSize={n};fontColor={col or C['ink']};fontStyle={1 if b else 0};"),cid)
def dot(x,y,r,fill,stroke=None,sw=1):return v("",x,y,r,r,f"ellipse;html=1;aspect=fixed;fillColor={fill};strokeColor={stroke or fill};strokeWidth={sw};")
def wall(x,y,w,h):return v("",x,y,w,h,bs(C["wall"],C["wall"],0,False))
def mapdraw(x,y,w,h,route=False,small=False):
 v("",x,y,w,h,bs(C["floor"],"#C9D1D8",2,True,False,"arcSize=8;"))
 for gx in range(int(x+35),int(x+w-15),50):fe((gx,y+8),(gx,y+h-8),"html=1;strokeColor=#E4E1D9;strokeWidth=1;endArrow=none;opacity=60;")
 for gy in range(int(y+32),int(y+h-10),45):fe((x+8,gy),(x+w-8,gy),"html=1;strokeColor=#E4E1D9;strokeWidth=1;endArrow=none;opacity=60;")
 t=9 if small else 13
 for X,Y,WW,HH in [(x+7,y+7,w-14,t),(x+7,y+h-7-t,w-14,t),(x+7,y+7,t,h-14),(x+w-7-t,y+7,t,h-14),
 (x+w*.30,y+16,t,h*.29),(x+w*.30,y+h*.47,t,h*.46),(x+w*.69,y+16,t,h*.26),(x+w*.69,y+h*.48,t,h*.45),
 (x+16,y+h*.34,w*.20,t),(x+w*.79,y+h*.35,w*.16,t)]:wall(X,Y,WW,HH)
 for fx,fy,fw,fh,col in [(.05,.09,.18,.12,C["wood"]),(.05,.49,.17,.12,"#B9C4CB"),(.76,.09,.16,.12,"#B9C4CB"),(.76,.60,.15,.13,C["wood"]),(.43,.10,.14,.10,"#B7C8D2")]:
  v("",x+w*fx,y+h*fy,w*fw,h*fh,bs(col,"#FFFFFF",2,True,False,"arcSize=7;"))
 for px,py in [(.12,.28),(.20,.28),(.82,.29),(.84,.79)]:dot(x+w*px,y+h*py,10 if small else 16,"#687A87","#FFFFFF")
 for px,py in [(.25,.77),(.92,.45)]:dot(x+w*px,y+h*py,15 if small else 23,"#79A974","#FFFFFF",2)
 if route:
  P=[(x+w*.14,y+h*.77),(x+w*.32,y+h*.65),(x+w*.47,y+h*.70),(x+w*.62,y+h*.52),(x+w*.83,y+h*.43)]
  for a,b in zip(P,P[1:]):fe(a,b,f"curved=1;html=1;strokeColor={C['green']};strokeWidth={4 if small else 5};endArrow=none;")
  for i,(px,py) in enumerate(P):
   dot(px-8,py-8,16,C["green"],"#FFFFFF",2)
   if i:v(str(i),px-11,py-29,22,17,ts(13,C["green"],True))

v("v70 安全候选规划器",50,20,850,58,ts(42,C["ink"],True,"left"))
v("VISUAL CONTEXT · LATENT FUTURE · RISK-AWARE CROSS-ROUND SEARCH",50,78,1050,28,ts(16,C["muted"],False,"left"))
pill("formal-v70",1120,35,160,38,C["tealf"],C["teal"],16,C["teald"])

v("视觉观测",55,112,180,34,ts(25,C["blue"],True,"left"));pill("4 history",225,110,120,34,C["bluef"],C["blue"],15,C["blue"]);pill("+ goal",355,110,100,34,C["goldf"],C["gold"],15,C["gold"])
obs=v("",45,145,700,260,bs("#FFFFFF","#AEC3D4",3,True,True,"arcSize=12;"),"obs")
uri="data:image/jpeg,"+base64.b64encode(IMG.read_bytes()).decode()
v("",58,157,674,240,f"shape=image;html=1;imageAspect=0;aspect=fixed;image={uri};strokeColor=none;fillColor=none;")
v("",258,157,5,240,bs("#FFFFFF","#FFFFFF",0,False));v("",58,276,400,5,bs("#FFFFFF","#FFFFFF",0,False));v("",458,157,6,240,bs("#FFFFFF","#FFFFFF",0,False))
for z,x,y in [("t−3",68,245),("t−2",268,245),("t−1",68,364),("t",268,364)]:pill(z,x,y,64,25,"#17324D","#FFFFFF",14,"#FFFFFF")
pill("目标 t+4",620,356,96,28,"#A77C16","#FFFFFF",14,"#FFFFFF")

dino=v("",790,150,270,225,bs(C["bluef"],C["blue"],3,True,True,"arcSize=14;"),"dino");v("DINOv3-L/16",815,164,220,36,ts(25,C["ink"],True));pill("冻结",815,330,78,28,"#FFFFFF",C["blue"],14,C["blue"])
for i in range(5):v("",825+i*17,230-i*8,115,82,bs(["#D6E8F3","#C6DEED","#B5D4E7","#9FC6DF","#85B5D3"][i],C["blue"],1,True,False,"arcSize=6;"))
for rr in range(4):
 for cc in range(5):v("",965+cc*14,235+rr*14,9,9,bs(C["blue"],C["blue"],0,False))
v("patch / 210-D",925,304,125,23,ts(15,C["blue"],True))

adapter=v("",1100,205,165,112,f"shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor=#E4EFF6;strokeColor={C['blue']};strokeWidth=2;","adapter");v("Adapter",1125,218,115,30,ts(22,C["ink"],True))
for yy,ww in [(262,72),(279,50),(296,30)]:v("",1147+(72-ww)/2,yy,ww,8,bs(C["blue"],C["blue"],0,True))

for ox,oy,op in [(-24,24,45),(-12,12,68)]:v("",1320+ox,135+oy,330,250,bs(C["purplef"],C["purple2"],2,True,False,f"arcSize=14;opacity={op};"))
ltwm=v("",1320,135,330,250,bs(C["purplef"],C["purple"],3,True,True,"arcSize=14;"),"ltwm");v("LTWM ×3",1360,150,250,38,ts(28,C["purple"],True))
for i in range(3):
 y=210+i*54;pill(f"M{i+1}",1350,y,58,30,"#FFFFFF",C["purple"],15,C["purple"]);fe((1415,y+15),(1515,y+15),f"curved=1;html=1;strokeColor={C['purple2']};strokeWidth=3;endArrow=none;",[(1450,y-5),(1480,y+33)])
 for j in range(4):v("",1530+j*18,y+5,12,20,bs(["#A4D8D6","#79C4C2","#52AAA9","#21898B"][j],"none",0,False))
v("未来 H4",1510,360,115,23,ts(15,C["purple"],True))

pred=v("",1690,150,245,220,bs("#FFFFFF","#AFCED4",2,True,True,"arcSize=14;"),"pred");v("预测 tokens",1720,166,185,34,ts(23,C["teald"],True))
for m in range(3):
 for rr in range(4):
  for cc in range(4):v("",1725+m*68+cc*11,224+rr*11,8,8,bs(["#B9E1DF","#84C9C7","#48A9A9"][m],"none",0,False))
 v(f"m{m+1}",1725+m*68,273,40,20,ts(13,C["muted"],True))
for t in range(4):pill(f"+{t+1}",1725+t*48,318,40,30,C["tealf"],C["teal"],14,C["teal"])
e(obs,dino,es(C["blue"],5,False,"exitX=1;exitY=.52;entryX=0;entryY=.52;"));e(dino,adapter,es(C["blue"],5,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"));e(adapter,ltwm,es(C["purple"],5,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"));e(ltwm,pred,es(C["teal"],5,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"))

h4=v("",2310,105,405,330,bs("#FFFFFF",C["green"],3,True,True,"arcSize=18;"),"h4");v("选择 H4",2340,120,210,42,ts(29,C["green"],True,"left"));pill("4-step plan",2560,123,125,34,C["greenf"],C["green"],15,C["green"]);mapdraw(2335,175,355,205,True,True);v("首动作 a₀",2430,392,160,26,ts(17,C["green"],True))

fusion=v("",900,425,925,145,bs("#FFFFFF",C["teal"],3,True,True,"arcSize=14;"),"fusion");v("候选表征",930,440,180,35,ts(24,C["teald"],True,"left"));pill("815-D",1675,438,120,38,C["teal"],C["teal"],21,"#FFFFFF")
segs=[("动作 242",245,C["orange2"]),("物理 24",75,"#DDB76B"),("DINO 210",215,C["blue2"]),("LTWM 112",120,C["purple2"]),("Δ 112",105,"#4CB0AF"),("集合 112",105,C["green2"]),("3",30,C["ink"])]
xx=925
for lab,ww,col in segs:v(lab,xx,495,ww,50,bs(col,"#FFFFFF",2,False,False,f"fontSize={16 if ww>70 else 13};fontColor=#FFFFFF;fontStyle=1;"));xx+=ww
e(pred,fusion,es(C["teal"],5,False,"exitX=.5;exitY=1;entryX=.72;entryY=0;"),[(1812,400),(1565,400)])

v("",1165,580,490,320,f"ellipse;html=1;fillColor=none;strokeColor={C['teal']};strokeWidth=4;dashed=1;opacity=60;")
core=v("",1215,615,390,250,f"shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fillColor={C['deep']};strokeColor={C['teal']};strokeWidth=6;shadow=1;","core");v("v70",1300,650,220,78,ts(64,"#FFFFFF",True));v("安全候选规划",1260,730,300,46,ts(32,"#FFFFFF",True));v("score · gate · search",1285,792,250,28,ts(18,"#BDE5E3"))

arm=v("",870,620,310,240,bs("#FFFFFF","#8BC0C3",2,True,True,"arcSize=14;"),"arm");v("Arm-D ×3",900,638,230,38,ts(27,C["teald"],True,"left"))
for i in range(3):
 x=900+i*88;v("",x,705,68,112,bs(["#D8EEEE","#BFE3E2","#9DD4D3"][i],C["teal"],2,True,False,"arcSize=9;"))
 for j,ww in enumerate([40,32,22,14]):v("",x+(68-ww)/2,723+j*18,ww,8,bs(C["teal"],C["teal"],0,True))
 v(f"m{i+1}",x+14,796,40,17,ts(13,C["teald"],True))
risk=v("",1640,620,310,240,bs("#FFFFFF","#E6A7AB",2,True,True,"arcSize=14;"),"risk");v("风险 μ / σ",1670,638,220,38,ts(27,C["red"],True,"left"));fe((1670,770),(1800,770),"html=1;strokeColor=#CDD5DD;strokeWidth=2;endArrow=none;");fe((1680,768),(1800,768),f"curved=1;html=1;strokeColor={C['red']};strokeWidth=4;endArrow=none;",[(1715,745),(1740,700),(1770,727)]);fe((1820,770),(1920,770),"html=1;strokeColor=#CDD5DD;strokeWidth=2;endArrow=none;");v("",1852,696,42,74,f"ellipse;html=1;fillColor=#F7C8CC;strokeColor={C['red']};strokeWidth=2;opacity=65;");fe((1873,686),(1873,780),f"html=1;strokeColor={C['red']};strokeWidth=3;dashed=1;endArrow=none;");pill("UCB",1818,798,108,28,C["redf"],C["red"],15,C["red"])
e(fusion,arm,es(C["teal"],4,False,"exitX=.15;exitY=1;entryX=.5;entryY=0;"));e(arm,core,es(C["teal"],5,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"));e(core,risk,es(C["red"],5,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"))

v("安全门",900,895,120,32,ts(22,C["green"],True,"left"));gates=[("physical","✓","physical",1010),("support","≤","support",1205),("dual","p","dual",1510),("harm","!","harm",1705)];G=[]
for gid,gl,lab,x in gates:G.append(v("",x,890,145,112,bs(C["greenf"],C["green"],3,True,True,"arcSize=30;"),gid));v(gl,x+45,900,55,50,ts(35,C["green"],True));v(lab,x+10,952,125,28,ts(17,C["green"],True))
for a,b in zip(G,G[1:]):e(a,b,es(C["green"],3,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"))
e(risk,"physical",es(C["red"],3,False,"exitX=.5;exitY=1;entryX=.5;entryY=0;"),[(1795,875),(1082,875)])

cand=v("",45,505,765,590,bs("#FFFFFF",C["orange"],3,True,True,"arcSize=16;"),"cand");v("K17 候选轨迹",75,525,300,42,ts(30,C["orange"],True,"left"));pill("NoMaD ×16",390,528,150,34,C["orangef"],C["orange"],16,C["orange"]);pill("+ incumbent",550,528,150,34,"#FFFFFF",C["muted"],15,C["muted"]);mapdraw(75,585,705,475)
v("",120,865,82,54,bs(C["ink"],"#FFFFFF",3,True,True,"arcSize=35;"));dot(146,876,28,"#FFFFFF",C["ink"],3)
for A,B in [((130,870),(105,845)),((190,870),(215,845)),((130,914),(105,940)),((190,914),(215,940))]:fe(A,B,f"html=1;strokeColor={C['ink']};strokeWidth=7;endArrow=none;");dot(B[0]-6,B[1]-6,12,C["ink"])
T=[(705,625),(730,670),(742,715),(748,765),(745,815),(728,865),(700,910),(660,960),(610,1005),(550,1035),(490,1040)];K=["#C1CAD2","#EAA064","#D97931","#B6C2CA","#E7802E","#D65158","#EBA56D","#B5C1CA","#D97931","#BAC5CD","#C7D0D7"]
for i,((tx,ty),col) in enumerate(zip(T,K)):fe((200,892),(tx,ty),f"curved=1;html=1;strokeColor={col};strokeWidth={5 if i in (2,4,5) else 3};endArrow=none;opacity=92;",[(320,885+(i-5)*6),(440,825+(i-5)*20),(565,755+(i-5)*33)]);dot(tx-7,ty-7,14,col,"#FFFFFF")
pill("H4",675,1008,70,30,C["orange"],C["orange"],16,"#FFFFFF");e(cand,fusion,es(C["orange"],5,False,"exitX=1;exitY=.30;entryX=0;entryY=.65;"),[(850,680),(850,520)])

cem=v("",1990,475,555,475,bs(C["orangef"],C["orange"],3,True,True,"arcSize=18;"),"cem");v("CEM ×3",2025,495,210,45,ts(31,C["orange"],True,"left"));pill("16 / round",2340,500,160,36,"#FFFFFF",C["orange"],16,C["orange"]);cent=[(2090,700),(2265,700),(2420,700)];sz=[(145,250),(115,205),(88,160)];R=[]
for idx,((cx,cy),(rw,rh)) in enumerate(zip(cent,sz),1):
 rid=v("",cx-rw/2,cy-rh/2,rw,rh,f"ellipse;html=1;fillColor=#FFFFFF;strokeColor={C['orange']};strokeWidth={4 if idx==3 else 2};dashed={1 if idx<3 else 0};",f"round{idx}");R.append(rid);v(f"R{idx}",cx-30,557,60,27,ts(17,C["orange"],True));cloud=[(-42,-78),(8,-88),(37,-48),(-25,-32),(16,2),(-42,33),(5,66),(38,50),(-10,92)];sc=[1,.7,.45][idx-1]
 for j,(dx,dy) in enumerate(cloud):dot(cx+dx*sc-7,cy+dy*sc-7,14,C["green"] if j in (3,4,6,8) else "#BFC8D0","#FFFFFF")
 dot(cx-7,cy-7,14,C["ink"])
for a,b in zip(R,R[1:]):e(a,b,es(C["orange"],4,False,"exitX=1;exitY=.5;entryX=0;entryY=.5;"))
v("收缩搜索",2180,875,180,30,ts(19,C["orange"],True));e("harm",cem,es(C["orange"],5,False,"exitX=1;exitY=.5;entryX=0;entryY=.55;"),[(1900,946),(1960,946),(1960,735)])

arch=v("",950,1135,1190,265,bs(C["greenf"],C["green"],4,True,True,"arcSize=20;"),"arch");v("union65",990,1155,220,50,ts(35,C["green"],True,"left"));v("跨轮安全归档",1200,1164,260,38,ts(24,C["green"],True,"left"));pill("17 + 16 + 16 + 16",1475,1160,245,36,"#FFFFFF",C["green"],17,C["green"])
for i,(name,count,col) in enumerate([("K17","17",C["orange"]),("R1","16",C["orange2"]),("R2","16","#54AAA7"),("R3","16",C["green"])]):
 x=1000+i*205;pill(name,x,1230,160,34,"#FFFFFF",col,17,col)
 for rr in range(2):
  for cc in range(6):dot(x+18+cc*22,1280+rr*24,13,col,"#FFFFFF")
 v(count,x+55,1332,50,22,ts(15,col,True))
best=v("★",1900,1215,120,120,f"ellipse;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor={C['green']};strokeWidth=5;fontColor={C['green']};fontSize=52;fontStyle=1;","best");v("最低能量",1885,1340,150,28,ts(17,C["green"],True));pill("无安全 → incumbent",980,1360,220,28,"#FFFFFF",C["gray"],14,C["gray"],False)
for i,(r,lane) in enumerate(zip(R,[1600,1750,1900])):e(r,arch,es(C["green"],3,False,f"exitX=.5;exitY=1;entryX={.55+i*.125};entryY=0;"),[(cent[i][0],970+i*15),(lane,970+i*15),(lane,1125)])
e(best,h4,es(C["green"],6,False,"exitX=.5;exitY=0;entryX=.5;entryY=1;"),[(1960,1010),(2680,1010),(2680,455),(2512,455)])

ext=v("",2290,1050,420,320,f"rounded=1;arcSize=18;whiteSpace=wrap;html=1;fillColor={C['grayf']};strokeColor={C['gray']};strokeWidth=2;dashed=1;dashPattern=10 7;","ext");v("外部 6B",2320,1070,170,35,ts(24,C["gray"],True,"left"));v("非 formal-v70",2490,1074,170,27,ts(15,C["gray"],False,"left"))
for sid,gl,lab,x in [("exec","a₀","执行首动作",2330),("observe","◎","更新观测",2460),("replan","↻","重规划",2590)]:v(gl,x,1140,82,82,f"ellipse;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor={C['gray']};strokeWidth=2;fontColor={C['gray']};fontSize=30;fontStyle=1;",sid);v(lab,x-16,1232,115,28,ts(16,C["gray"],True))
e("exec","observe",es(C["gray"],2,True,"exitX=1;exitY=.5;entryX=0;entryY=.5;"));e("observe","replan",es(C["gray"],2,True,"exitX=1;exitY=.5;entryX=0;entryY=.5;"));e(h4,"exec",es(C["gray"],2,True,"exitX=1;exitY=.72;entryX=.5;entryY=0;"),[(2740,340),(2740,1025),(2370,1025)]);e("replan",obs,es(C["gray"],2,True,"exitX=1;exitY=.5;entryX=0;entryY=.52;"),[(2735,1180),(2735,1408),(20,1408),(20,280)])

lx=55
for col,lab in [(C["blue"],"视觉"),(C["purple"],"未来"),(C["orange"],"搜索"),(C["red"],"风险"),(C["green"],"安全"),(C["gray"],"外部")]:dot(lx,1435,15,col);v(lab,lx+22,1428,65,26,ts(15,C["muted"],True,"left"));lx+=115
v("场景素材：Shixart1985 / Wikimedia Commons / CC BY 2.0（裁剪）",1650,1425,1030,32,ts(14,C["muted"],False,"right"))
ET.indent(mf,space="  ");OUT.write_text(ET.tostring(mf,encoding="unicode",xml_declaration=True),encoding="utf-8");print(f"Wrote {OUT}")
