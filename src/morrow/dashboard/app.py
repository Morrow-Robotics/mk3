"""Render the investor sequence as a single self-contained HTML page.

No build step, no framework, no external assets — the data is embedded as JSON
and vanilla JS builds the DOM. Aesthetic borrows the restrained,
scientific-white look of a DeepMind results page: lots of whitespace, thin
type, one blue accent, monospaced numbers.
"""

from __future__ import annotations

import json

_CSS = """
:root{
  --ink:#202124; --muted:#5f6368; --line:#e8eaed; --bg:#ffffff; --panel:#fbfcfd;
  --blue:#1a73e8; --teal:#12a4a4; --good:#1e8e3e; --warn:#e37400; --bad:#d93025;
  --mono:'Roboto Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:400 15px/1.5 -apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:56px 28px 96px}
header{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:30px;font-weight:500;letter-spacing:-.02em;margin:0}
h1 .dot{color:var(--blue)}
.sub{color:var(--muted);font-size:15px}
.badge{font:600 11px/1 var(--mono);letter-spacing:.08em;color:var(--teal);
  border:1px solid var(--teal);border-radius:999px;padding:5px 9px;text-transform:uppercase}
.runtime{color:var(--muted);font:400 12.5px/1.6 var(--mono);margin:14px 0 40px;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:12px 0}
.runtime b{color:var(--ink);font-weight:500}
section{margin:40px 0}
.eyebrow{font:600 12px/1 var(--mono);letter-spacing:.1em;color:var(--blue);
  text-transform:uppercase;margin-bottom:14px}
h2{font-size:20px;font-weight:500;margin:0 0 4px}
.lead{color:var(--muted);margin:0 0 20px;max-width:70ch}
.claim{font-size:19px;font-weight:400;line-height:1.5;border-left:3px solid var(--blue);
  padding:6px 0 6px 18px;color:var(--ink)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
.grid{display:grid;gap:14px}
.chain{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:18px}
.pill{font:600 12px/1 var(--mono);padding:8px 11px;border-radius:999px;
  background:#eef3fe;color:var(--blue);border:1px solid #d7e3fd;white-space:nowrap}
.pill.term{background:#e6f4ea;color:var(--good);border-color:#cce8d4}
.arrow{color:var(--muted);font-size:13px}
.edge{display:grid;grid-template-columns:190px 90px 1fr;gap:10px;align-items:center;
  padding:9px 0;border-top:1px solid var(--line);font-size:13.5px}
.edge:first-child{border-top:0}
.edge .t{font:600 12.5px/1 var(--mono)}
.chip{display:inline-block;font:500 11.5px/1 var(--mono);padding:5px 8px;border-radius:7px;
  background:#f1f3f4;color:var(--muted)}
.chip.f{background:#e8f0fe;color:var(--blue)}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.metric .n{font:500 30px/1.1 var(--mono);letter-spacing:-.02em}
.metric .l{color:var(--muted);font-size:12.5px;margin-top:6px}
.metric .n.good{color:var(--good)} .metric .n.blue{color:var(--blue)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:11px 10px;border-bottom:1px solid var(--line)}
th{font:600 11px/1 var(--mono);letter-spacing:.06em;color:var(--muted);text-transform:uppercase}
td .mono{font-family:var(--mono)}
.bar{height:7px;border-radius:999px;background:#eef0f2;overflow:hidden;margin-top:5px}
.bar>span{display:block;height:100%;background:var(--blue)}
.bar.replay>span{background:var(--bad)}
.runs{display:flex;flex-wrap:wrap;gap:10px}
.run{font:600 12px/1 var(--mono);padding:9px 12px;border-radius:10px;border:1px solid var(--line)}
.run.ok{background:#e6f4ea;color:var(--good);border-color:#cce8d4}
.run.fa{background:#e8f0fe;color:var(--blue);border-color:#d7e3fd}
.tl{margin:0;padding:0;list-style:none;font:400 13px/1.6 var(--mono)}
.tl li{padding:7px 0;border-top:1px solid var(--line);color:var(--muted)}
.tl li:first-child{border-top:0}
.tl .ok{color:var(--good)} .tl .rec{color:var(--warn);font-weight:600}
.foot{color:var(--muted);font-size:12.5px;margin-top:48px;border-top:1px solid var(--line);padding-top:16px}
button{font:600 12px/1 var(--mono);color:#fff;background:var(--blue);border:0;border-radius:9px;
  padding:10px 14px;cursor:pointer}
"""

_JS = """
const D = window.__DATA__;
const pct = x => (x*100).toFixed(1)+'%';
const el = (t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e;};

function chain(graph){
  const box=el('div','chain');
  graph.edges.forEach((e,i)=>{
    if(i===0) box.appendChild(el('span','pill',e.from));
    box.appendChild(el('span','arrow','\\u2192'));
    box.appendChild(el('span','pill'+(e.to==='VERIFIED'?' term':''),e.to));
  });
  return box;
}
function edges(graph){
  const box=el('div','card');
  graph.edges.forEach(e=>{
    const row=el('div','edge');
    row.appendChild(el('span','t',e.from+' \\u2192 '+e.to));
    row.appendChild(el('span','chip f',e.frame));
    const r=el('div');
    r.appendChild(el('span','chip','verify: '+e.success));
    r.appendChild(document.createTextNode(' '));
    r.appendChild(el('span','chip','recover: '+e.recovery));
    row.appendChild(r);
    box.appendChild(row);
  });
  return box;
}
function timeline(run){
  const ul=el('ul','tl');
  run.timeline.forEach(ev=>{
    const li=el('li');
    if(ev.outcome==='ok') li.innerHTML='<span class="ok">\\u2713</span> '+ev.edge+' \\u00b7 attempt '+ev.attempts;
    else if(ev.outcome==='recover') li.innerHTML='<span class="rec">\\u21ba recovery</span> '+ev.edge+' \\u00b7 '+ev.action+' after '+ev.attempts+' tries';
    else li.innerHTML='<span class="rec">\\u2715 flagged</span> '+ev.edge;
    ul.appendChild(li);
  });
  return ul;
}
function metric(n,l,cls){const m=el('div','metric');m.appendChild(el('div','n '+(cls||''),n));m.appendChild(el('div','l',l));return m;}

function render(){
  const root=document.getElementById('app'); root.innerHTML='';

  // compiled state graph
  let s=el('section');
  s.appendChild(el('div','eyebrow','1 \\u00b7 Compiled skill'));
  s.appendChild(el('h2','A demonstration became a verified state machine'));
  const g=D.primary_graph;
  s.appendChild(el('p','lead','SKU "'+g.sku_id+'"  \\u00b7  hash '+g.hash+'  \\u00b7  '+g.descriptor.kind+', symmetry '+g.descriptor.symmetry+'. Grasp verified by hardware; placement by vision. Each edge carries its own recovery.'));
  s.appendChild(chain(g)); s.appendChild(edges(g));
  root.appendChild(s);

  // randomized runs
  s=el('section');
  s.appendChild(el('div','eyebrow','2 \\u00b7 Transfer'));
  s.appendChild(el('h2','Same skill, randomized product and carton'));
  s.appendChild(el('p','lead','The product is moved and rotated and the carton is jittered. No replay \\u2014 every edge is re-instantiated from a fresh scene.'));
  const runs=el('div','runs');
  D.runs.forEach((r,i)=>{
    const cls = r.first_attempt ? 'run fa' : 'run ok';
    runs.appendChild(el('div',cls,'run '+(i+1)+': '+r.final_state+(r.first_attempt?' \\u00b7 first try':' \\u00b7 +'+r.retries+' retry')));
  });
  s.appendChild(runs);
  if(D.runs.length){
    s.appendChild(el('p','lead','Trace of run 1 \\u2014 every transition re-perceives before it acts:'));
    s.appendChild(timeline(D.runs[0]));
  }
  root.appendChild(s);

  // forced failure + recovery
  s=el('section');
  s.appendChild(el('div','eyebrow','3 \\u00b7 Recovery'));
  s.appendChild(el('h2','A forced grasp failure, recovered from the current state'));
  s.appendChild(el('p','lead','The first full round of grasps is made to miss. The cell detects it from the vacuum signal, runs the transition\\u2019s recovery, and finishes \\u2014 no human.'));
  s.appendChild(timeline(D.forced_failure));
  root.appendChild(s);

  // second sku
  s=el('section');
  s.appendChild(el('div','eyebrow','4 \\u00b7 Changeover'));
  s.appendChild(el('h2','A second SKU, onboarded without writing code'));
  const g2=D.second_graph;
  s.appendChild(el('p','lead','SKU "'+g2.sku_id+'" ('+g2.descriptor.kind+'), hash '+g2.hash+'. Demonstrated, compiled, and run: '+D.second_run.final_state+'.'));
  s.appendChild(chain(g2));
  root.appendChild(s);

  // benchmark
  s=el('section');
  s.appendChild(el('div','eyebrow','5 \\u00b7 Evaluation'));
  s.appendChild(el('h2','Frozen benchmark \\u00b7 n='+D.benchmark.n+' randomized trials each'));
  const bench=(data)=>{
    const t=el('table');
    t.innerHTML='<thead><tr><th>SKU class</th><th>onboarding</th><th>morrow final</th><th>first-attempt</th><th>human</th><th>open-loop replay</th></tr></thead>';
    const tb=el('tbody');
    Object.entries(data.kinds).forEach(([k,r])=>{
      const m=r.morrow;
      const tr=el('tr');
      tr.innerHTML =
        '<td><b>'+k+'</b></td>'+
        '<td class="mono">'+r.onboarding.n_demos+' demos \\u00b7 '+r.onboarding.code_changes+' code</td>'+
        '<td class="mono">'+pct(m.final_success_rate)+'<div class="bar"><span style="width:'+(m.final_success_rate*100)+'%"></span></div></td>'+
        '<td class="mono">'+pct(m.first_attempt_rate)+'<div class="bar"><span style="width:'+(m.first_attempt_rate*100)+'%"></span></div></td>'+
        '<td class="mono">'+pct(m.human_intervention_rate)+'</td>'+
        '<td class="mono">'+pct(r.baseline_open_loop_success_rate)+'<div class="bar replay"><span style="width:'+(r.baseline_open_loop_success_rate*100)+'%"></span></div></td>';
      tb.appendChild(tr);
    });
    t.appendChild(tb); const c=el('div','card'); c.appendChild(t); return c;
  };
  s.appendChild(bench(D.benchmark));
  if(D.benchmark_stress){
    s.appendChild(el('p','lead','Under stress \\u2014 rotated carton, intermittent low-confidence frames, and occlusion noise. Final success holds; more of the work is autonomous recovery.'));
    s.appendChild(bench(D.benchmark_stress));
  }
  root.appendChild(s);
}
render();
"""


def render_page(sequence: dict, runtime: dict) -> str:
    data = json.dumps(sequence)
    rt = runtime
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>morrow · demonstration to verified skill</title>
<style>{_CSS}</style></head><body><div class="wrap">
<header>
  <h1>morrow<span class="dot">.</span></h1>
  <span class="sub">demonstration → verified skill · mk3</span>
  <span class="badge">simulation</span>
</header>
<div class="runtime">
  <b>runtime</b> &nbsp; python {rt['python']} &nbsp;·&nbsp; numpy {rt['numpy']}
  &nbsp;·&nbsp; {rt['platform']} &nbsp;·&nbsp; {rt['cores']} cores
</div>
<p class="claim">Demonstrate packing a product once. Morrow compiles it into a verified skill
that transfers to new poses, recovers from failed grasps on its own, and takes a new SKU
without a line of code.</p>
<div id="app"></div>
<div class="foot">
  Numbers are from the analytic simulator, not a physical cell — they show the mechanism
  (perception, hardware-verified grasp, per-transition recovery), not industrial reliability.
  The robot and perceiver are boundaries; the bench adapter is the one piece still to build.
</div>
</div>
<script>window.__DATA__ = {data};</script>
<script>{_JS}</script>
</body></html>"""
