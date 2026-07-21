"""The physics dashboard: customer packing clips (left) → the verified skill →
the LeRobot parallel-jaw arm doing the task in MuJoCo physics (center), with the
FSM timeline and physical-placement result.

Honest by construction: the customer-video slots are drop-in local files (we
neither download nor auto-extract skills from them yet — Phase 3), and the sim
is a real physics render, not a cartoon.
"""

from __future__ import annotations

import base64
import json
import os
import platform
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v")


def find_clips(videos_dir: str) -> list[str]:
    if not os.path.isdir(videos_dir):
        return []
    files = [f for f in sorted(os.listdir(videos_dir)) if f.lower().endswith(_VIDEO_EXTS)]
    return files[:5]


def _slots(videos_dir: str, embed: bool) -> list[dict]:
    clips = find_clips(videos_dir)
    slots = []
    for i in range(5):
        if i < len(clips):
            name = clips[i]
            if embed:
                with open(os.path.join(videos_dir, name), "rb") as f:
                    src = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
            else:
                src = "/videos/" + name
            slots.append({"i": i + 1, "name": name, "src": src})
        else:
            slots.append({"i": i + 1, "name": None, "src": None})
    return slots


def runtime_info() -> dict:
    import mujoco
    return {"python": platform.python_version(), "mujoco": mujoco.__version__,
            "platform": platform.platform(terse=True)}


_CSS = """
:root{--ink:#202124;--muted:#5f6368;--line:#e8eaed;--panel:#fbfcfd;--blue:#1a73e8;
  --teal:#12a4a4;--good:#1e8e3e;--warn:#e37400;--mono:'Roboto Mono',ui-monospace,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:400 15px/1.5 -apple-system,'Segoe UI',Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:44px 26px 90px}
h1{font-size:27px;font-weight:500;letter-spacing:-.02em;margin:0}
h1 .dot{color:var(--blue)}
.sub{color:var(--muted)}
.badge{font:600 11px/1 var(--mono);letter-spacing:.08em;color:var(--teal);border:1px solid var(--teal);
  border-radius:999px;padding:5px 9px;text-transform:uppercase;margin-left:10px}
.rt{color:var(--muted);font:400 12.5px/1.6 var(--mono);border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);padding:10px 0;margin:14px 0 8px}
.note{background:#fff8e1;border:1px solid #fde68a;border-radius:10px;padding:12px 14px;
  color:#7a5b00;font-size:13px;margin:16px 0 28px}
.grid{display:grid;grid-template-columns:300px 1fr;gap:26px}
.eyebrow{font:600 12px/1 var(--mono);letter-spacing:.1em;color:var(--blue);text-transform:uppercase;margin-bottom:12px}
.slot{border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:14px;background:var(--panel)}
.slot video{width:100%;display:block;background:#000}
.slot .cap{font:500 12px/1.3 var(--mono);color:var(--muted);padding:8px 10px}
.slot.empty{border-style:dashed;min-height:120px;display:flex;align-items:center;justify-content:center;
  text-align:center;color:var(--muted);font-size:12.5px;padding:16px}
.pack{border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:18px;margin-bottom:22px}
.pack h3{margin:0 0 4px;font-size:17px;font-weight:500;text-transform:capitalize}
.pack video{width:100%;max-width:520px;border-radius:10px;display:block;background:#111}
.chain{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 10px}
.pill{font:600 11px/1 var(--mono);padding:6px 9px;border-radius:999px;background:#eef3fe;color:var(--blue);border:1px solid #d7e3fd}
.pill.ok{background:#e6f4ea;color:var(--good);border-color:#cce8d4}
.pill.term{background:#e6f4ea;color:var(--good);border-color:#cce8d4}
.m{display:flex;gap:18px;flex-wrap:wrap;font:500 13px/1 var(--mono);color:var(--muted);margin-top:6px}
.m b{color:var(--good)} .m .bad{color:var(--warn)}
.foot{color:var(--muted);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:16px}
.marker{margin-top:36px;border-top:1px solid var(--line);padding-top:26px}
.lead{color:var(--muted);max-width:74ch;margin:6px 0 16px}
.mrow{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}
.mrow label{font:500 12.5px/1 var(--mono);color:var(--muted)}
.mrow input,.mrow select{font:400 13px var(--mono);padding:6px 8px;border:1px solid var(--line);border-radius:7px}
.mcanvas{border:1px solid var(--line);border-radius:10px;background:#eee;cursor:crosshair;max-width:100%;touch-action:none}
.btn{font:600 12px var(--mono);color:#fff;background:var(--blue);border:0;border-radius:8px;padding:9px 13px;cursor:pointer}
.btn.alt{background:#3c4043}
.toggle{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.toggle button{border:0;background:#fff;padding:7px 11px;font:600 12px var(--mono);cursor:pointer;color:var(--muted)}
.toggle button.on{background:var(--blue);color:#fff}
.result{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start;margin-top:14px}
.result video{width:320px;border-radius:10px;background:#111}
.status{font:500 13px/1.5 var(--mono);color:var(--muted)} .status b{color:var(--good)}
.realbadge{font:700 10px/1 var(--mono);letter-spacing:.06em;color:#fff;background:var(--teal);
  border-radius:999px;padding:4px 8px;margin-left:10px;text-transform:uppercase;vertical-align:middle}
.pack.real{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal) inset}
.wrow{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.wfig{flex:1;min-width:280px} .wimg{width:100%;border-radius:10px;display:block;background:#111}
.wfig .cap{font:500 12px/1.3 var(--mono);color:var(--muted);padding:6px 2px}
.warrow{font:700 22px var(--mono);color:var(--teal)}
.profile{margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
.plabel{font:600 11px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.spark{width:100%;max-width:340px;border-radius:6px;display:block;margin-bottom:6px}
.pnote{font:400 12px/1.5 var(--muted);color:var(--muted);max-width:70ch;margin-top:4px}
"""

_JS = """
const D = window.__PHYS__;
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e;};
function slots(){
  const col=document.getElementById('videos');
  D.slots.forEach(s=>{
    if(s.src){
      const d=el('div','slot');
      const v=document.createElement('video'); v.src=s.src; v.controls=true; v.loop=true; v.muted=true; v.autoplay=true; v.playsInline=true;
      d.appendChild(v); d.appendChild(el('div','cap','clip '+s.i+' · '+s.name));
      col.appendChild(d);
    } else {
      col.appendChild(el('div','slot empty','drop a customer packing clip →  videos/'+s.i+'.mp4'));
    }
  });
}
function renderPack(stage, title, r, tag){
  const c=el('div','pack'+(tag?' real':''));
  const h=el('h3',null,title); if(tag){const b=el('span','realbadge',tag);h.appendChild(b);} c.appendChild(h);
  const v=document.createElement('video'); v.src='data:video/mp4;base64,'+r.mp4_b64;
  v.controls=true; v.loop=true; v.muted=true; v.autoplay=true; v.playsInline=true;
  c.appendChild(v);
  const chain=el('div','chain');
  r.timeline.filter(e=>e.edge&&e.edge.indexOf('->')>0).forEach(e=>{
    const to=e.edge.split('->')[1];
    chain.appendChild(el('span','pill'+(e.outcome==='ok'?(to==='VERIFIED'?' term':' ok'):''), to));
  });
  c.appendChild(chain);
  const m=el('div','m');
  m.innerHTML='<span>'+r.final_state+'</span>'+
    '<span>packed in carton: '+(r.inside_carton?'<b>yes</b>':'<span class=bad>no</span>')+'</span>'+
    '<span>first-attempt: '+(r.first_attempt?'yes':'+'+r.recoveries+' recovery')+'</span>';
  c.appendChild(m);
  stage.appendChild(c);
}
function packs(){
  const stage=document.getElementById('stage');
  if(D.arm){Object.entries(D.arm.kinds).forEach(([kind,r])=>
    renderPack(stage, kind+'  ·  SO-101 5-DOF model', r, 'SO-101 model · IK'));}
}
function marker(){
  const cv=document.getElementById('mcanvas'), ctx=cv.getContext('2d');
  let frame=null, prod=null, cart=null, mode='product', drag=null, grabClip=null, grabTime=0;
  const clipSel=document.getElementById('clipSel');
  D.slots.filter(s=>s.src).forEach(s=>{const o=document.createElement('option');o.value=s.src;o.textContent='clip '+s.i+' · '+s.name;clipSel.appendChild(o);});
  if(!clipSel.options.length){const o=document.createElement('option');o.textContent='(no clips in ./videos)';o.disabled=true;clipSel.appendChild(o);}
  const vid=document.createElement('video'); vid.muted=true; vid.style.display='none'; document.body.appendChild(vid);
  function redraw(){
    ctx.clearRect(0,0,cv.width,cv.height);
    if(frame){ctx.putImageData(frame,0,0);} else {ctx.fillStyle='#e6e6e6';ctx.fillRect(0,0,cv.width,cv.height);ctx.fillStyle='#999';ctx.font='13px monospace';ctx.fillText('load a frame, grab from a clip, or mark on blank',20,140);}
    const box=(r,col)=>{if(!r)return;ctx.strokeStyle=col;ctx.lineWidth=2;ctx.strokeRect(r[0],r[1],r[2]-r[0],r[3]-r[1]);};
    box(prod,'#1a73e8'); box(cart,'#b8860b');
  }
  function toC(e){const b=cv.getBoundingClientRect();return [(e.clientX-b.left)*cv.width/b.width,(e.clientY-b.top)*cv.height/b.height];}
  cv.onmousedown=e=>{const p=toC(e);drag=[p[0],p[1],p[0],p[1]];};
  cv.onmousemove=e=>{if(!drag)return;const p=toC(e);drag[2]=p[0];drag[3]=p[1];
    const r=[Math.min(drag[0],drag[2]),Math.min(drag[1],drag[3]),Math.max(drag[0],drag[2]),Math.max(drag[1],drag[3])];
    if(mode==='product')prod=r; else cart=r; redraw();};
  window.addEventListener('mouseup',()=>{drag=null;});
  document.getElementById('markProd').onclick=()=>{mode='product';markProd.classList.add('on');markCart.classList.remove('on');};
  document.getElementById('markCart').onclick=()=>{mode='carton';markCart.classList.add('on');markProd.classList.remove('on');};
  document.getElementById('frameFile').onchange=e=>{const f=e.target.files[0];if(!f)return;const img=new Image();
    img.onload=()=>{ctx.drawImage(img,0,0,cv.width,cv.height);frame=ctx.getImageData(0,0,cv.width,cv.height);redraw();};img.src=URL.createObjectURL(f);};
  document.getElementById('grabBtn').onclick=()=>{const src=clipSel.value;if(!src)return;vid.src=src;
    vid.onloadeddata=()=>{vid.currentTime=Math.min(0.6,(vid.duration||1)/3);};
    vid.onseeked=()=>{try{ctx.drawImage(vid,0,0,cv.width,cv.height);frame=ctx.getImageData(0,0,cv.width,cv.height);
      grabClip=src.split('/').pop();grabTime=vid.currentTime;redraw();
      document.getElementById('resStatus').textContent='grabbed '+grabClip+' @ '+grabTime.toFixed(2)+'s — drag a rough box, then SAM2 refine.';
    }catch(err){document.getElementById('resStatus').textContent='could not grab frame: '+err;}};
    vid.load();};
  document.getElementById('sam2Btn').onclick=async()=>{
    const st=document.getElementById('resStatus'); const seed=mode==='product'?prod:cart;
    if(!grabClip){st.textContent='grab a clip frame first — SAM2 runs server-side on the clip.';return;}
    if(!seed){st.textContent='drag a rough '+mode+' box first; SAM2 will tighten it to the object.';return;}
    const bf=[seed[0]/cv.width,seed[1]/cv.height,seed[2]/cv.width,seed[3]/cv.height];
    st.textContent='SAM2 segmenting '+mode+' …';
    try{
      const resp=await fetch('/segment',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({clip:grabClip,time:grabTime,box_frac:bf})});
      const j=await resp.json();
      if(j.ok){const f=j.bbox_frac,r=[f[0]*cv.width,f[1]*cv.height,f[2]*cv.width,f[3]*cv.height];
        if(mode==='product')prod=r;else cart=r;redraw();
        st.innerHTML='SAM2 <b>'+mode+'</b> mask · score '+j.score+' — now build &amp; pack.';}
      else st.textContent='✗ '+j.error;
    }catch(err){st.textContent='segment failed: '+err;}
  };
  document.getElementById('runBtn').onclick=async()=>{
    const st=document.getElementById('resStatus'), rv=document.getElementById('resVid');
    if(!prod){st.textContent='draw the product box first (blue).';return;}
    const ann={sku:'marked',image:{w:cv.width,h:cv.height},scale_m_per_px:parseFloat(scaleIn.value),
      product:{kind:kindSel.value,bbox_px:prod.map(v=>Math.round(v)),height_m:parseFloat(heightIn.value)}};
    if(cart)ann.carton={bbox_px:cart.map(v=>Math.round(v))};
    st.textContent='building skill + running the SO-101 model in MuJoCo …'; rv.style.display='none';
    try{
      const resp=await fetch('/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(ann)});
      const j=await resp.json();
      if(j.ok){rv.style.display='block';rv.src='data:video/mp4;base64,'+j.mp4_b64;rv.play();
        st.innerHTML='<b>'+j.final_state+'</b> · SO-101 model · packed in carton: '+
          (j.inside_carton?'yes':'no')+' · product half-size '+j.size.join(' × ')+' m';}
      else{st.textContent='✗ '+j.error;}
    }catch(err){st.textContent='request failed: '+err;}
  };
  redraw();
}
function watchPanel(){
  if(!D.watch){return;}
  const host=document.getElementById('watchstage'); const w=D.watch;
  const card=el('div','pack real');
  card.appendChild(el('h3',null,'watched: '+w.clip+'  ·  frame '+w.frame_idx));
  const row=el('div','wrow');
  const fig1=el('div','wfig'); const img=document.createElement('img');
  img.src='data:image/png;base64,'+w.overlay_png_b64; img.className='wimg';
  fig1.appendChild(img); fig1.appendChild(el('div','cap','SAM2 masks on the real clip — carton (amber) + product (green)'));
  const fig2=el('div','wfig'); const v=document.createElement('video');
  v.src='data:video/mp4;base64,'+w.pack_mp4_b64; v.controls=true; v.loop=true; v.muted=true; v.autoplay=true; v.playsInline=true; v.className='wimg';
  fig2.appendChild(v); fig2.appendChild(el('div','cap','SO-101 packs the '+w.kind+' it inferred, in MuJoCo'));
  row.appendChild(fig1); row.appendChild(el('div','warrow','→')); row.appendChild(fig2);
  card.appendChild(row);
  const m=el('div','m');
  m.innerHTML='<span>SAM2 score carton '+w.carton_score+' · product '+w.product_score+'</span>'+
    '<span>'+w.final_state+'</span>'+
    '<span>packed in carton: '+(w.inside_carton?'<b>yes</b>':'<span class=bad>no</span>')+'</span>'+
    '<span>mapped product half '+w.product_half_m.join(' × ')+' m</span>';
  card.appendChild(m);
  if(w.profile){const p=w.profile;
    const pr=el('div','profile');
    const spk=document.createElement('img'); spk.src='data:image/png;base64,'+w.sparkline_png_b64; spk.className='spark';
    pr.appendChild(el('div','plabel','carton-region packing activity (cv2 frame-diff)'));
    pr.appendChild(spk);
    const lc={LOW:'#e37400',MEDIUM:'#1a73e8',HIGH:'#1e8e3e'}[p.confidence_label]||'#5f6368';
    const pm=el('div','m');
    pm.innerHTML='<span>≈<b>'+p.n_events+'</b> candidate place events</span>'+
      '<span>confidence <b style="color:'+lc+'">'+p.confidence_label+'</b> ('+p.confidence+')</span>'+
      '<span>active '+Math.round(p.active_fraction*100)+'% of '+p.duration_s+'s</span>';
    pr.appendChild(pm);
    pr.appendChild(el('div','pnote',p.note));
    card.appendChild(pr);}
  host.appendChild(card);
}
slots(); packs(); watchPanel(); marker();
"""


def render_physics_page(slots: list[dict], runtime: dict,
                        arm_showcase: dict | None = None, watch: dict | None = None) -> str:
    data = json.dumps({"slots": slots, "arm": arm_showcase, "watch": watch})
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>morrow · customer video → LeRobot physics</title><style>{_CSS}</style></head>
<body><div class="wrap">
<header><h1>morrow<span class="dot">.</span></h1>
<span class="sub">customer packing → verified skill → LeRobot physics sim</span>
<span class="badge">mujoco · parallel jaw</span></header>
<div class="rt">python {runtime['python']} · mujoco {runtime['mujoco']} · {runtime['platform']}</div>
<div class="note"><b>Honest scope:</b> the customer clips on the left are your drop-in files —
we do <i>not</i> download them or yet auto-extract a skill from video (that's the hard Phase-3 step).
The center panels are the <i>SO-101 (MuJoCo Menagerie model, 5-DOF)</i> — the LeRobot arm's real MJCF
run in MuJoCo, <b>not</b> a physical arm: an orientation-aware IK drives its position actuators tool-down
and the parallel jaw grasps by friction — grasp is verified from two-finger contact, lift from real
height, placement from footprint overlap. Its usable top-down reach is small (x≈0.21–0.32 m,
|y|&lt;0.10 m), so the cell sits in that envelope.</div>
<div class="grid">
  <div><div class="eyebrow">Customer packing (drop-in)</div><div id="videos"></div></div>
  <div><div class="eyebrow">LeRobot SO-101 physics — arm doing the task</div><div id="stage"></div></div>
</div>
<section class="marker">
  <div class="eyebrow">Watch a real clip → workflow → SO-101 (SAM2)</div>
  <p class="lead">This is the full <b>watch→do</b> path on a real customer clip: SAM2 (real weights)
  segments the carton and an operator-seeded product on a frame, and the SO-101 then packs the product
  it inferred, verified in physics. Honest limits: monocular clips carry no metric scale (operator sets
  m/px), and fully-unattended video tracking drifts under hand occlusion — so segmentation is
  operator-seeded, not magic. Omitted here if SAM2 weights aren't installed.</p>
  <div id="watchstage"></div>
</section>
<section class="marker">
  <div class="eyebrow">Mark ANY clip → SAM2 refine → SO-101 physics</div>
  <p class="lead">Grab a frame from any clip, drag a <i>rough</i> box over the product (or carton), then
  <b>SAM2 refine</b> to snap it to the real object (runs server-side on the clip — needs weights).
  Set the table scale (metres per pixel) and product height, then build a physics skill and watch the
  <b>SO-101 model</b> pack it (box/cylinder). Honest: monocular clips have no metric scale, so you supply
  m/px — SAM2 gives the pixel geometry, not depth.</p>
  <div class="mrow">
    <input type="file" id="frameFile" accept="image/*">
    <select id="clipSel"></select>
    <button class="btn alt" id="grabBtn">grab frame</button>
    <span class="toggle"><button id="markProd" class="on">mark product</button><button id="markCart">mark carton</button></span>
    <button class="btn" id="sam2Btn">SAM2 refine</button>
  </div>
  <canvas id="mcanvas" class="mcanvas" width="480" height="270"></canvas>
  <div class="mrow">
    <label>kind <select id="kindSel"><option>box</option><option>cylinder</option></select></label>
    <label>scale m/px <input id="scaleIn" type="number" step="0.0001" value="0.0007" style="width:92px"></label>
    <label>height m <input id="heightIn" type="number" step="0.01" value="0.06" style="width:74px"></label>
    <button class="btn" id="runBtn">build &amp; pack in physics</button>
  </div>
  <div class="result"><video id="resVid" controls loop muted playsinline style="display:none"></video>
    <div class="status" id="resStatus"></div></div>
</section>
<div class="foot">Frames rendered by MuJoCo; the FSM (compile → run_skill) is byte-identical to the
analytic sim and to the eventual bench. Cartesian EE waypoints map to a SO-101 model via IK.</div>
</div>
<script>window.__PHYS__ = {data};</script><script>{_JS}</script></body></html>"""


def serve_physics(host: str = "127.0.0.1", port: int = 8001, videos_dir: str = "videos",
                  arm_kinds=("box",)) -> None:
    from .showcase import build_arm_showcase, build_watch_showcase
    watch = None
    try:
        print("watching a real clip with SAM2 → SO-101 (skipped if no weights) ...", flush=True)
        watch = build_watch_showcase()
    except Exception as e:  # SAM2 optional / clip-specific — never block the dashboard
        print(f"  watch panel skipped: {e}", flush=True)
    print("rendering SO-101 model packs (5-DOF IK) ...", flush=True)
    arm_showcase = build_arm_showcase(kinds=arm_kinds)
    page = render_physics_page(_slots(videos_dir, embed=False),
                               runtime_info(), arm_showcase, watch).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(page, "text/html; charset=utf-8")
            elif self.path.startswith("/videos/"):
                name = os.path.basename(self.path[len("/videos/"):])
                fp = os.path.join(videos_dir, name)
                if name and os.path.isfile(fp):
                    with open(fp, "rb") as f:
                        self._send(f.read(), "video/mp4")
                else:
                    self.send_error(404)
            else:
                self.send_error(404)

        def _body(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length) or b"{}")

        def _segment(self, req):
            # SAM2 on an operator click/box, live, on any local clip frame.
            from .watch import have_sam2, segment_click
            if not have_sam2():
                return {"ok": False, "error": "SAM2 weights not installed (set $MORROW_SAM2_CKPT)"}
            name = os.path.basename(req.get("clip", ""))
            fp = os.path.join(videos_dir, name)
            if not (name and os.path.isfile(fp)):
                return {"ok": False, "error": f"clip {name!r} not found in ./{videos_dir}"}
            out = segment_click(fp, float(req.get("time", 0.0)),
                                box_frac=req.get("box_frac"), point_frac=req.get("point_frac"))
            return {"ok": True, **out}

        def _annotate(self, ann):
            from .showcase import _inside_carton, _mp4_b64
            from .watch import pack_annotation_on_arm  # the SO-101 model is the only embodiment
            frames, size, _c, result, world = pack_annotation_on_arm(
                ann, seed=int(ann.get("seed", 0)), capture=True)
            return {"ok": True, "mp4_b64": _mp4_b64(frames), "embodiment": "so101",
                    "final_state": result.final_state, "success": result.success,
                    "inside_carton": _inside_carton(world),
                    "size": [round(v, 3) for v in size]}

        def do_POST(self):
            if self.path not in ("/annotate", "/segment"):
                self.send_error(404)
                return
            try:
                req = self._body()
                body = self._segment(req) if self.path == "/segment" else self._annotate(req)
            except Exception as e:  # oversize gripper, bad JSON, unsupported kind -> honest error
                body = {"ok": False, "error": str(e)}
            self._send(json.dumps(body).encode(), "application/json")

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"morrow physics cell on http://{host}:{port}  (drop clips in ./{videos_dir}/)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
