import { useCanvasState, useHostTheme } from "cursor/canvas";

type Theme = ReturnType<typeof useHostTheme>;
type View = "index" | "radar" | "xray" | "gate" | "press" | "scope" | "courier" | "merge" | "phaser" | "dataset" | "tree";

const MECHANISMS: Array<{ id: Exclude<View, "index">; mark: string; title: string }> = [
  { id: "radar", mark: "R", title: "Connection radar" },
  { id: "xray", mark: "X", title: "Registration X-ray" },
  { id: "gate", mark: "G", title: "Start gate" },
  { id: "press", mark: "P", title: "MessagePack press" },
  { id: "scope", mark: "H", title: "Heartbeat scope" },
  { id: "courier", mark: "C", title: "Partition courier" },
  { id: "merge", mark: "M", title: "Merge machine" },
  { id: "phaser", mark: "Φ", title: "Phaser clock" },
  { id: "dataset", mark: "D", title: "Dataset floodgate" },
  { id: "tree", mark: "T", title: "Aggregator tree" },
];

const CSS = `
* { box-sizing: border-box; }
button, input { font: inherit; }
.root { min-height: 100%; padding: 18px; overflow-x: hidden; }
.workbench { width: min(1240px, 100%); min-height: calc(100vh - 36px); margin: auto; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.eyebrow { font: 700 9px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .15em; text-transform: uppercase; }
.title { margin: 5px 0 6px; font: 620 clamp(20px, 3vw, 28px)/1.05 ui-sans-serif, system-ui, sans-serif; letter-spacing: -.035em; }
.sentence { margin: 0; max-width: 680px; font: 400 12px/1.5 ui-sans-serif, system-ui, sans-serif; }
.tiny { font: 500 9px/1.35 ui-monospace, SFMono-Regular, Menlo, monospace; }
.nav-layout { display: grid; grid-template-columns: 42px minmax(0,1fr); min-height: calc(100vh - 36px); }
.marks { display: flex; flex-direction: column; align-items: center; gap: 6px; padding-right: 7px; border-right: 1px solid; }
.mark, .back, .control, .index-mech, .endpoint, .cell, .feed, .gen, .chunk, .stage-button {
  color: inherit; background: transparent; cursor: pointer; border-radius: 3px;
}
.mark, .back { width: 28px; height: 28px; border: 1px solid transparent; font: 700 10px/1 ui-monospace, monospace; }
.back { margin-bottom: 7px; border-bottom-color: currentColor; font-size: 15px; }
.mark[aria-current="page"] { opacity: 1; }
.mark:not([aria-current="page"]) { opacity: .38; }
.control { min-height: 31px; padding: 6px 10px; border: 1px solid; font-size: 10px; }
.control[aria-pressed="true"] { font-weight: 700; }
.control:disabled { opacity: .35; cursor: default; }
.mark:focus-visible, .back:focus-visible, .control:focus-visible, .index-mech:focus-visible, .endpoint:focus-visible,
.cell:focus-visible, .feed:focus-visible, .chunk:focus-visible, .stage-button:focus-visible,
.trace-step button:focus-visible, .courier-packet:focus-visible, input:focus-visible {
  outline: 2px solid currentColor; outline-offset: 3px;
}
.instrument { min-width: 0; min-height: calc(100vh - 36px); position: relative; overflow: hidden; }
.index { min-height: calc(100vh - 36px); position: relative; }
.index-head { position: relative; z-index: 2; max-width: 620px; }
.constellation { min-height: 690px; position: relative; margin-top: -10px; }
.constellation-lines { position: absolute; inset: 3%; width: 94%; height: 94%; pointer-events: none; }
.index-core { position: absolute; left: 50%; top: 47%; width: 92px; height: 92px; transform: translate(-50%,-50%); border: 1px solid; border-radius: 50%; display: grid; place-items: center; }
.index-core::after { content: ""; width: 38px; height: 38px; border: 1px solid; border-radius: 50%; }
.index-mech { position: absolute; width: 170px; padding: 6px; text-align: left; border: 0; }
.index-mech svg { width: 100%; height: 58px; display: block; }
.index-mech strong { display: block; font-size: 10px; font-weight: 630; }
.index-mech small { font: 500 8px ui-monospace, monospace; opacity: .55; }
.i0{left:2%;top:5%}.i1{left:31%;top:0}.i2{right:3%;top:6%}.i3{left:0;top:36%}.i4{right:0;top:35%}
.i5{left:3%;bottom:7%}.i6{left:31%;bottom:0}.i7{right:3%;bottom:7%}.i8{left:43%;top:21%;transform:translateX(-50%)}.i9{right:25%;bottom:21%}
.radar-page { min-height: 100%; display: grid; place-items: center; }
.radar-copy { position: absolute; left: 5%; top: 4%; z-index: 2; max-width: 390px; }
.radar-field { width: min(720px, 82vw); aspect-ratio: 1; border: 1px solid; border-radius: 50%; position: relative; margin-top: 20px; }
.radar-ring { position: absolute; border: 1px solid; border-radius: 50%; inset: 14%; pointer-events: none; }
.radar-ring.r2 { inset: 29%; }.radar-ring.r3 { inset: 43%; }
.radar-center { position: absolute; left: 50%; top: 50%; width: 76px; height: 76px; transform: translate(-50%,-50%); border: 1px solid; border-radius: 50%; display: grid; place-items: center; text-align: center; }
.radar-sweep { position: absolute; left: 50%; top: 50%; width: 42%; border-top: 2px solid; transform-origin: left center; }
.scan-once { animation: scan-once .7s ease-out 1 both; }
.endpoint { position: absolute; width: 104px; min-height: 46px; border: 1px solid; padding: 5px; font: 500 8px/1.3 ui-monospace, monospace; transition: left .3s ease, top .3s ease; }
.e0{left:7%;top:27%}.e1{left:79%;top:22%}.e2{left:74%;top:75%}.e3{left:12%;top:78%}
.radar-console { position: absolute; right: 4%; bottom: 3%; max-width: 260px; text-align: right; }
.xray-page { display: grid; grid-template-columns: .78fr 1.22fr; min-height: 100%; }
.xray-envelope { padding: clamp(24px,5vw,64px); border-right: 1px solid; display: flex; flex-direction: column; justify-content: center; }
.envelope-stack { min-height: 300px; position: relative; margin-top: 28px; }
.envelope-layer { position: absolute; left: 0; right: 12%; padding: 15px; border: 1px solid; font: 500 9px ui-monospace, monospace; transition: transform .25s ease; }
.xray-trace { padding: clamp(24px,5vw,70px); display: grid; align-content: center; gap: 16px; }
.trace-step { display: grid; grid-template-columns: 30px 1fr; gap: 12px; align-items: center; }
.trace-step button { width: 28px; height: 28px; border: 1px solid; border-radius: 50%; background: transparent; color: inherit; cursor: pointer; }
.trace-line { margin-left: 13px; height: 26px; border-left: 1px solid; }
.reply-slab { margin-top: 18px; padding: 18px; border: 1px solid; }
.gate-page { min-height: 100%; display: grid; grid-template-rows: auto 1fr auto; padding: clamp(22px,5vw,62px); }
.race { align-self: center; position: relative; min-height: 330px; border-bottom: 2px solid; display: grid; grid-template-columns: repeat(4,1fr) 110px; align-items: end; gap: 12px; }
.race-lane { min-height: 250px; position: relative; border-right: 1px dashed; display: flex; align-items: flex-end; justify-content: center; padding-bottom: 20px; }
.cell { width: 58px; height: 58px; border: 1px solid; border-radius: 50%; font: 700 10px ui-monospace, monospace; transition: transform .25s ease; }
.cell.arrived { transform: translateY(-150px); }
.gate-post { height: 280px; border-left: 4px solid; position: relative; }
.gate-post::before { content: ""; position: absolute; top: 0; left: -22px; width: 44px; border-top: 4px solid; }
.gate-trigger { position: absolute; right: 0; top: 34px; width: 96px; }
.gate-readout { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.press-page { min-height: 100%; display: grid; grid-template-columns: minmax(180px,.55fr) minmax(280px,1fr) minmax(180px,.55fr); }
.press-copy { padding: 36px 24px; align-self: start; }
.press-column { border-left: 1px solid; border-right: 1px solid; padding: 26px; display: flex; flex-direction: column; align-items: stretch; justify-content: center; gap: 0; }
.press-platen { height: 52px; border: 1px solid; display: grid; place-items: center; }
.press-chamber { min-height: 260px; border-left: 1px solid; border-right: 1px solid; padding: 20px; display: grid; align-content: center; gap: 12px; }
.byte-rack { display: grid; grid-template-columns: repeat(8,1fr); gap: 3px; }
.byte { border: 1px solid; min-height: 28px; display: grid; place-items: center; font: 500 8px ui-monospace, monospace; }
.press-controls { padding: 36px 22px; align-self: end; display: grid; gap: 7px; }
.stage-button { border: 0; border-left: 2px solid; padding: 8px; text-align: left; font-size: 10px; }
.scope-page { min-height: 100%; display: grid; grid-template-columns: minmax(170px,.3fr) 1fr; }
.scope-console { padding: 30px 22px; border-right: 1px solid; display: flex; flex-direction: column; gap: 18px; }
.scope-stack { padding: 28px; display: grid; align-content: center; gap: 18px; }
.scope-channel { min-height: 138px; position: relative; border-bottom: 1px solid; overflow: hidden; }
.scope-channel svg { position: absolute; inset: 18px 0 0; width: 100%; height: 95px; }
.scope-tag { position: absolute; left: 0; top: 0; z-index: 1; }
.scope-readout { position: absolute; right: 0; top: 0; }
.courier-page { min-height: 100%; position: relative; padding: clamp(25px,5vw,62px); display: grid; grid-template-rows: auto 1fr; }
.route-map { position: relative; min-height: 520px; margin-top: 18px; }
.route-line { position: absolute; left: 15%; right: 15%; top: 50%; border-top: 1px dashed; }
.route-stop { position: absolute; top: 50%; transform: translate(-50%,-50%); width: 150px; min-height: 116px; border: 1px solid; padding: 14px; background: inherit; }
.origin{left:14%}.relay{left:50%;border-radius:50%;width:118px;min-height:118px;text-align:center}.controller{left:86%}
.courier-packet { position: absolute; top: calc(50% - 18px); width: 36px; height: 36px; border: 2px solid; transform: rotate(45deg); transition: left .35s ease; }
.courier-packet span { display:block; transform:rotate(-45deg); font:700 8px monospace; margin:10px 0 0 5px; }
.merge-page { min-height: 100%; position: relative; display: grid; place-items: center; }
.merge-copy { position: absolute; left: 4%; top: 4%; max-width: 360px; }
.merge-dial { width: min(620px,76vw); aspect-ratio:1; border:1px solid; border-radius:50%; position:relative; margin-top:55px; }
.merge-hub { position:absolute; inset:34%; border:1px solid; border-radius:50%; display:grid; place-items:center; text-align:center; }
.feed { position:absolute; width:86px; height:54px; border:1px solid; font:600 9px ui-monospace,monospace; }
.f0{left:43%;top:3%}.f1{right:5%;top:43%}.f2{left:43%;bottom:3%}.f3{left:5%;top:43%}
.merge-output { position:absolute; right:4%; bottom:5%; text-align:right; max-width:260px; }
.phaser-page { min-height: 100%; display: grid; grid-template-columns: 1fr minmax(250px,.42fr); }
.clock-face { align-self:center; justify-self:center; width:min(650px,70vw); aspect-ratio:1; border:1px solid; border-radius:50%; position:relative; }
.clock-center { position:absolute; inset:35%; border:1px solid; border-radius:50%; display:grid; place-items:center; text-align:center; }
.gen { position:absolute; width:46px; height:46px; border:1px solid; border-radius:50%; font:700 9px monospace; display:grid; place-items:center; }
.phaser-log { border-left:1px solid; padding:35px 24px; display:flex; flex-direction:column; justify-content:center; gap:9px; }
.log-row { display:grid; grid-template-columns:34px 1fr; gap:9px; align-items:center; font:500 9px monospace; }
.dataset-page { min-height:100%; display:grid; grid-template-rows:auto 1fr; padding:26px; }
.flood-instrument { display:grid; grid-template-columns:180px 1fr; min-height:560px; margin-top:16px; }
.reservoir { border:1px solid; padding:16px; display:flex; flex-direction:column; gap:7px; }
.chunk { min-height:36px; border:1px solid; font:600 9px monospace; }
.channels { display:grid; grid-template-rows:repeat(3,1fr); border-top:1px solid; border-bottom:1px solid; }
.channel { position:relative; display:grid; grid-template-columns:90px repeat(6,1fr); align-items:center; border-bottom:1px solid; }
.channel:last-child{border-bottom:0}.channel-slot{text-align:center;font:500 8px/1.35 monospace}.owned{font-weight:800}
.tree-page { min-height:100%; display:grid; grid-template-columns:minmax(260px,1fr) 250px; }
.topology { display:grid; place-items:center; padding:20px; }
.topology svg { width:100%; max-height:80vh; }
.topology-compact { display:none; width:100%; padding:18px 12px; }
.compact-node { border:1px solid; padding:9px 6px; text-align:center; font:650 10px/1.25 ui-monospace,monospace; }
.compact-link { min-height:34px; display:grid; place-items:center; font:500 9px/1.2 ui-monospace,monospace; }
.compact-branch { display:grid; grid-template-columns:repeat(2,1fr); gap:28px; }
.compact-cells { display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin-top:34px; }
.tree-console { border-left:1px solid; padding:30px 22px; display:flex; flex-direction:column; justify-content:center; gap:18px; }
@keyframes scan-once { from{transform:rotate(-70deg)} to{transform:rotate(55deg)} }
@keyframes pulse-once { 0%{opacity:.2;transform:scale(.7)} 60%{opacity:1;transform:scale(1.2)} 100%{opacity:.55;transform:scale(1)} }
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media (max-width:760px){
  .root{padding:10px}.nav-layout{grid-template-columns:1fr}.marks{position:sticky;top:0;z-index:5;flex-direction:row;overflow-x:auto;border-right:0;border-bottom:1px solid;padding:5px 0}.back{margin:0 5px 0 0;border-bottom:0;border-right:1px solid}
  .instrument{min-height:calc(100vh - 70px)}.constellation{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;min-height:0;margin-top:20px}.constellation-lines,.index-core{display:none}.index-mech{position:static;width:auto;transform:none!important;border-bottom:1px solid}
  .radar-copy{position:relative;left:auto;top:auto;padding:18px}.radar-field{width:min(92vw,580px);margin:0}.radar-console{position:relative;right:auto;bottom:auto;padding:15px;text-align:left}.endpoint{width:82px}.e0{left:1%}.e1{left:77%}
  .xray-page{grid-template-columns:1fr}.xray-envelope{border-right:0;border-bottom:1px solid;padding:24px}.xray-trace{padding:24px}.gate-page{padding:20px}.race{grid-template-columns:repeat(4,1fr) 58px;min-height:300px}.race-lane{min-height:230px}.gate-trigger{width:78px;font-size:8px}.cell{width:46px;height:46px}.cell.arrived{transform:translateY(-130px)}
  .press-page{grid-template-columns:1fr}.press-copy{padding:22px}.press-column{border:1px solid;padding:18px}.press-controls{padding:20px;grid-template-columns:repeat(2,1fr)}
  .scope-page{grid-template-columns:1fr}.scope-console{border-right:0;border-bottom:1px solid;padding:20px}.scope-stack{padding:18px}.scope-channel{min-height:120px}
  .route-map{min-height:620px}.route-line{left:50%;right:auto;top:12%;bottom:12%;border-top:0;border-left:1px dashed}.route-stop{left:50%!important;top:auto;transform:translate(-50%,0)}.origin{top:4%}.relay{top:39%}.controller{top:75%}.courier-packet{left:calc(50% - 18px)!important;top:31%;transition:top .3s}.courier-packet.mid{top:56%}.courier-packet.done{top:72%}
  .merge-copy{position:relative;left:auto;top:auto;padding:18px}.merge-page{display:block}.merge-dial{width:min(94vw,560px);margin:15px auto}.merge-output{position:relative;right:auto;bottom:auto;padding:18px;text-align:left}
  .phaser-page{grid-template-columns:1fr}.clock-face{width:min(92vw,580px);margin:20px auto}.phaser-log{border-left:0;border-top:1px solid;padding:22px}.flood-instrument{grid-template-columns:1fr}.reservoir{display:grid;grid-template-columns:repeat(3,1fr)}.channels{min-height:430px}.channel{grid-template-columns:62px repeat(6,1fr)}
  .tree-page{grid-template-columns:1fr}.tree-console{border-left:0;border-top:1px solid;padding:22px}.topology svg{max-height:55vh}
}
@media (max-width:480px){
  .constellation{grid-template-columns:1fr}.index-mech{min-height:92px}.index-mech svg{height:48px}
  .endpoint{font-size:7px;width:68px;min-height:40px}.radar-center{width:58px;height:58px}.race{gap:4px}.cell{width:38px;height:38px;font-size:8px}
  .byte-rack{grid-template-columns:repeat(6,1fr)}.feed{width:66px;height:45px}.flood-instrument{font-size:8px}.reservoir{grid-template-columns:repeat(2,1fr)}
  .topology{padding:4px}.topology svg{display:none}.topology-compact{display:block}.tree-console{padding:18px 14px}.compact-cells{grid-template-columns:repeat(2,1fr);margin-top:24px}.compact-branch{gap:12px}
}
`;

function Glyph({ id, t }: { id: Exclude<View, "index">; t: Theme }) {
  const a=t.category.cyan,s=t.stroke.primary,f=t.fill.tertiary;
  if(id==="radar")return <svg viewBox="0 0 150 58"><circle cx="75" cy="29" r="23" fill="none" stroke={s}/><path d="M75 29l21-14" stroke={a}/><circle cx="75" cy="29" r="3" fill={a}/></svg>;
  if(id==="xray")return <svg viewBox="0 0 150 58"><path d="M15 12h52v34H15zM82 18h53v23H82z" fill={f} stroke={s}/><path d="M28 22h27M28 31h20M67 29h15M95 29h26" stroke={a}/></svg>;
  if(id==="gate")return <svg viewBox="0 0 150 58"><path d="M12 47h126M82 8v39" stroke={s}/><circle cx="27" cy="38" r="5" fill={a}/><circle cx="50" cy="27" r="5" fill={a}/></svg>;
  if(id==="press")return <svg viewBox="0 0 150 58"><path d="M55 6h40v13H55zM62 19v30h26V19" fill={f} stroke={s}/><path d="M68 29h14M68 36h14" stroke={a}/></svg>;
  if(id==="scope")return <svg viewBox="0 0 150 58"><path d="M8 30h25l7-16 12 31 10-23 11 8h69" fill="none" stroke={a}/><path d="M8 48h134" stroke={s}/></svg>;
  if(id==="courier")return <svg viewBox="0 0 150 58"><path d="M12 39h126" stroke={s}/><rect x="22" y="18" width="35" height="21" fill={f} stroke={a}/><path d="M66 29h45m0 0-8-7m8 7-8 7" stroke={a} fill="none"/></svg>;
  if(id==="merge")return <svg viewBox="0 0 150 58"><circle cx="77" cy="29" r="15" fill={f} stroke={a}/><path d="M10 10l52 13M10 47l52-12M140 10L92 23M140 47L92 35" stroke={s}/></svg>;
  if(id==="phaser")return <svg viewBox="0 0 150 58"><circle cx="75" cy="29" r="23" fill="none" stroke={s}/><path d="M75 29V11M75 29l15 10" stroke={a}/></svg>;
  if(id==="dataset")return <svg viewBox="0 0 150 58"><path d="M15 8h25v42H15zM40 16h95M40 29h95M40 42h95" stroke={s} fill={f}/><circle cx="70" cy="16" r="4" fill={a}/><circle cx="98" cy="29" r="4" fill={a}/></svg>;
  return <svg viewBox="0 0 150 58"><circle cx="75" cy="8" r="5" fill={a}/><path d="M75 13v10L42 36M75 23l33 13M42 36L20 52M42 36l20 16M108 36L88 52M108 36l21 16" stroke={s}/></svg>;
}

function Index({ open }: { open: (view: View) => void }) {
  const t=useHostTheme();
  return <div className="index">
    <header className="index-head"><div className="eyebrow" style={{color:t.category.cyan}}>AIPerf cellular transport</div><h1 className="title" style={{color:t.text.primary}}>Velo mechanisms</h1><p className="sentence" style={{color:t.text.secondary}}>Ten interactive instruments expose how cellular identity, synchronization, distribution, and reduction cross the Velo plane.</p></header>
    <div className="constellation">
      <svg className="constellation-lines" viewBox="0 0 1000 650" preserveAspectRatio="none"><path d="M500 305L125 82M500 305L365 48M500 305L860 88M500 305L90 292M500 305L915 290M500 305L125 560M500 305L380 610M500 305L862 557M500 305L435 180M500 305L708 465" fill="none" stroke={t.stroke.tertiary} strokeDasharray="2 8"/></svg>
      <div className="index-core" style={{borderColor:t.stroke.primary,color:t.category.cyan,background:t.bg.elevated}} aria-hidden="true"/>
      {MECHANISMS.map((m,i)=><button key={m.id} className={`index-mech i${i}`} onClick={()=>open(m.id)}><Glyph id={m.id} t={t}/><strong style={{color:t.text.primary}}>{m.title}</strong><small style={{color:t.text.tertiary}}>{m.mark} / {String(i+1).padStart(2,"0")}</small></button>)}
    </div>
  </div>;
}

function Radar() {
  const t=useHostTheme();
  const [sweep,setSweep]=useCanvasState<number>("velo3.radar.sweep",0);
  const [locked,setLocked]=useCanvasState<number>("velo3.radar.locked",-1);
  const endpoints=["tcp://host:port","uds://path","loopback","ephemeral"];
  const discovered=Math.min(4,Math.max(0,sweep));
  return <article className="instrument radar-page">
    <header className="radar-copy"><div className="eyebrow" style={{color:t.category.cyan}}>R / connection radar</div><h1 className="title">Resolve a controller</h1><p className="sentence" style={{color:t.text.secondary}}>A known TCP or UDS endpoint enters the hello exchange and becomes routable controller PeerInfo.</p></header>
    <div className="radar-field" style={{borderColor:t.stroke.primary}}>
      <i className="radar-ring" style={{borderColor:t.stroke.tertiary}}/><i className="radar-ring r2" style={{borderColor:t.stroke.tertiary}}/><i className="radar-ring r3" style={{borderColor:t.stroke.tertiary}}/>
      {sweep>0&&locked<0&&<span key={sweep} className="radar-sweep scan-once" style={{borderColor:t.category.cyan}}/>}
      <button className="radar-center control" style={{borderColor:t.category.cyan,color:t.category.cyan,background:t.bg.elevated}} onClick={()=>locked>=0?setLocked(-1):setSweep(Math.min(4,sweep+1))}>{locked>=0?"Reset lock":discovered<4?"Sweep sector":"All resolved"}</button>
      {endpoints.map((label,i)=><button key={label} className={`endpoint e${i}`} disabled={i>=discovered} aria-pressed={locked===i} onClick={()=>setLocked(i)} style={{borderColor:locked===i?t.category.cyan:t.stroke.primary,color:i<discovered?t.text.primary:t.text.tertiary,background:t.bg.elevated}}>{i<discovered?label:"unresolved"}</button>)}
    </div>
    <div className="radar-console tiny" style={{color:locked>=0?t.category.cyan:t.text.tertiary}}>{locked>=0?`${endpoints[locked]} → _hello → register_peer(controller)`:`${discovered}/4 sectors resolved · each sweep is user-triggered`}</div>
  </article>;
}

function Xray() {
  const t=useHostTheme();
  const [opened,setOpened]=useCanvasState<boolean>("velo3.xray.opened",false);
  const [trace,setTrace]=useCanvasState<number>("velo3.xray.trace",0);
  const steps=["decode CellRegister","register_peer(cell)","spec_for(cell_id)","encode RegisterReply"];
  return <article className="instrument xray-page">
    <section className="xray-envelope" style={{borderColor:t.stroke.primary}}><div className="eyebrow" style={{color:t.category.cyan}}>X / registration X-ray</div><h1 className="title">Open the request</h1><p className="sentence" style={{color:t.text.secondary}}>The controller learns cell identity, selects a pre-sliced protocol-v2 envelope, and returns it with the run-wide START handle.</p><button className="control" aria-pressed={opened} onClick={()=>setOpened(!opened)} style={{borderColor:opened?t.category.cyan:t.stroke.primary,color:opened?t.category.cyan:t.text.primary,marginTop:20,alignSelf:"flex-start"}}>{opened?"Close envelope":"Dissect envelope"}</button>
      <div className="envelope-stack">{["handler / aiperf.cell.register","cell_peer / MessagePack bytes","cell_id / u32"].map((x,i)=><div key={x} className="envelope-layer" style={{top:30+i*58,borderColor:i===trace?t.category.cyan:t.stroke.primary,color:i===trace?t.category.cyan:t.text.secondary,background:t.bg.elevated,transform:opened?`translate(${i*12}px,${i*7}px)`:"none"}}>{x}</div>)}</div>
    </section>
    <section className="xray-trace">{steps.map((x,i)=><div key={x}><div className="trace-step"><button aria-label={`Trace ${x}`} aria-pressed={trace===i} onClick={()=>setTrace(i)} style={{borderColor:trace===i?t.category.cyan:t.stroke.primary,color:trace===i?t.category.cyan:t.text.tertiary}}>{i+1}</button><div><div className="eyebrow" style={{color:trace===i?t.category.cyan:t.text.tertiary}}>{x}</div>{trace===i&&<div className="tiny" style={{color:t.text.secondary,marginTop:5}}>{i===0?"raw payload → CellRegister":i===1?"establish return route":i===2?"pure lookup by cell ID":"envelope bytes + EventHandle"}</div>}</div></div>{i<3&&<div className="trace-line" style={{borderColor:t.stroke.tertiary}}/>}</div>)}
      <div className="reply-slab" style={{borderColor:trace===3?t.category.cyan:t.stroke.primary,background:t.bg.elevated}}><div className="eyebrow" style={{color:t.text.tertiary}}>RegisterReply</div><div className="tiny" style={{color:t.text.primary,marginTop:8}}>envelope: protocol-v2 bytes</div><div className="tiny" style={{color:t.text.primary}}>start_event: EventHandle</div></div>
    </section>
  </article>;
}

function Gate() {
  const t=useHostTheme();
  const [arrivals,setArrivals]=useCanvasState<boolean[]>("velo3.gate.arrivals",[false,false,false,false]);
  const [released,setReleased]=useCanvasState<boolean>("velo3.gate.released",false);
  const all=arrivals.every(Boolean);
  const arrive=(i:number)=>setArrivals(arrivals.map((v,n)=>n===i?true:v));
  return <article className="instrument gate-page">
    <header><div className="eyebrow" style={{color:t.category.cyan}}>G / start gate</div><h1 className="title">Asynchronous arrival. One release.</h1><p className="sentence" style={{color:t.text.secondary}}>Each cell registers once; the Nth registration releases the controller barrier, and START wakes every awaiting cell.</p></header>
    <div className="race" style={{borderColor:t.stroke.primary}}>{[0,1,2,3].map(i=><div className="race-lane" key={i} style={{borderColor:t.stroke.tertiary}}><button className={`cell ${arrivals[i]?"arrived":""}`} aria-pressed={arrivals[i]} onClick={()=>arrive(i)} disabled={arrivals[i]||released} style={{borderColor:arrivals[i]?t.category.cyan:t.stroke.primary,color:arrivals[i]?t.category.cyan:t.text.primary,background:t.bg.elevated}}>c{i}</button></div>)}<div className="gate-post" style={{borderColor:all?t.category.cyan:t.stroke.primary}}><button className="control gate-trigger" disabled={!all||released} onClick={()=>setReleased(true)} style={{borderColor:t.category.cyan,color:t.category.cyan}}>Trigger START</button></div></div>
    <footer className="gate-readout"><span className="tiny" style={{color:released?t.category.cyan:t.text.tertiary}}>{released?"all awaiters → Ready":`${arrivals.filter(Boolean).length} / 4 registered`}</span><button className="control" onClick={()=>{setArrivals([false,false,false,false]);setReleased(false)}} style={{borderColor:t.stroke.primary}}>Reset apparatus</button></footer>
  </article>;
}

function Press() {
  const t=useHostTheme();
  const [stage,setStage]=useCanvasState<number>("velo3.press.stage",0);
  const bytes=["83","a7","63","6f","75","6e","74","2a","cb","7f","f8","00","00","00","00","00"];
  return <article className="instrument press-page">
    <header className="press-copy"><div className="eyebrow" style={{color:t.category.cyan}}>P / MessagePack press</div><h1 className="title">Typed state becomes raw bytes</h1><p className="sentence" style={{color:t.text.secondary}}>rmp-serde preserves NaN and infinity in a Velo raw payload, then reconstructs the cellular value at the handler.</p></header>
    <section className="press-column" style={{borderColor:t.stroke.primary}}>
      <div className="press-platen" style={{borderColor:stage===1?t.category.cyan:t.stroke.primary,color:stage===1?t.category.cyan:t.text.tertiary}}>rmp_serde::to_vec</div>
      <div className="press-chamber" style={{borderColor:t.stroke.primary}}>
        {stage===0&&<div className="tiny"><b>CellMessage::Heartbeat</b><br/>count: 42<br/>ttft: NaN<br/>max: +∞</div>}
        {stage===1&&<div style={{height:80,borderTop:`18px solid ${t.category.cyan}`,borderBottom:`18px solid ${t.category.cyan}`,animation:"pulse-once .55s ease-out"}}/>}
        {stage===2&&<div className="byte-rack">{bytes.map((b,i)=><span className="byte" key={`${b}${i}`} style={{borderColor:i>7?t.category.cyan:t.stroke.primary,color:i>7?t.category.cyan:t.text.secondary}}>{b}</span>)}</div>}
        {stage===3&&<div className="tiny" style={{color:t.category.cyan}}><b>Decoded Heartbeat</b><br/>count: 42<br/>ttft: NaN<br/>max: +∞</div>}
      </div>
      <div className="press-platen" style={{borderColor:stage===3?t.category.cyan:t.stroke.primary,color:stage===3?t.category.cyan:t.text.tertiary}}>rmp_serde::from_slice</div>
    </section>
    <aside className="press-controls">{["Load typed value","Apply pressure","Inspect raw bytes","Reconstruct"].map((x,i)=><button key={x} className="stage-button" aria-pressed={stage===i} onClick={()=>setStage(i)} style={{borderColor:stage===i?t.category.cyan:t.stroke.primary,color:stage===i?t.category.cyan:t.text.secondary}}>{i+1} / {x}</button>)}</aside>
  </article>;
}

function Scope() {
  const t=useHostTheme();
  const [ticks,setTicks]=useCanvasState<number>("velo3.scope.ticks",4);
  const [failed,setFailed]=useCanvasState<boolean>("velo3.scope.failed",false);
  const safeTicks=Math.min(32,Math.max(1,ticks));
  const visible=Math.min(8,safeTicks);
  return <article className="instrument scope-page">
    <aside className="scope-console" style={{borderColor:t.stroke.primary}}><div><div className="eyebrow" style={{color:t.category.cyan}}>H / heartbeat scope</div><h1 className="title">Read the live pulse</h1><p className="sentence" style={{color:t.text.secondary}}>Fire-and-forget snapshots expose counters, sketches, observation time, lag, and a missing cell.</p></div><button className="control" onClick={()=>setTicks(Math.min(32,safeTicks+1))} style={{borderColor:t.stroke.primary}}>Emit heartbeat</button><button className="control" aria-pressed={failed} onClick={()=>setFailed(!failed)} style={{borderColor:failed?t.category.cyan:t.stroke.primary,color:failed?t.category.cyan:t.text.primary}}>{failed?"Restore cell 2":"Fail cell 2"}</button><div className="tiny" style={{color:t.text.tertiary}}>issued {safeTicks*12}<br/>completed {safeTicks*11}<br/>liveness {failed?"2 / 3":"3 / 3"}</div></aside>
    <section className="scope-stack">{[0,1,2].map(cell=>{const dead=failed&&cell===2;return <div className="scope-channel" key={cell} style={{borderColor:t.stroke.primary}}><div className="scope-tag eyebrow" style={{color:dead?t.text.tertiary:t.text.primary}}>CH {cell} / cell {cell}</div><div className="scope-readout tiny" style={{color:dead?t.category.cyan:t.text.tertiary}}>{dead?"lag ↑ · pulse missing":`observed_at +${safeTicks}s`}</div><svg viewBox="0 0 800 95" preserveAspectRatio="none">{Array.from({length:visible},(_,i)=>i).map(i=>dead&&i>3?null:<path key={i} d={`M${i*100} 52h23l8-30 13 58 13-42 12 14h31`} fill="none" stroke={t.category.cyan} strokeWidth="2" opacity={.35+i/visible*.65}/>)}</svg></div>})}</section>
  </article>;
}

function Courier() {
  const t=useHostTheme();
  const [position,setPosition]=useCanvasState<0|1|2>("velo3.courier.position",0);
  const [acked,setAcked]=useCanvasState<boolean>("velo3.courier.acked",false);
  const [attempt,setAttempt]=useCanvasState<number>("velo3.courier.attempt",1);
  const deliver=()=>{setPosition(2);setAcked(false)};
  return <article className="instrument courier-page">
    <header><div className="eyebrow" style={{color:t.category.cyan}}>C / partition courier</div><h1 className="title">Ship through a fresh return route</h1><p className="sentence" style={{color:t.text.secondary}}>The terminal unary payload includes the fresh shipper PeerInfo and a records or folded-store partition; the controller registers that peer before acknowledging.</p></header>
    <div className="route-map"><div className="route-line" style={{borderColor:t.stroke.primary}}/>
      <section className="route-stop origin" style={{borderColor:position===0?t.category.cyan:t.stroke.primary,background:t.bg.editor}}><div className="eyebrow">fresh ship Velo</div><div className="tiny" style={{margin:"10px 0",color:t.text.secondary}}>cell_peer<br/>partition</div><button className="control" onClick={()=>setPosition(1)} disabled={position!==0} style={{borderColor:t.category.cyan,color:t.category.cyan}}>Send toward controller</button></section>
      <section className="route-stop relay" style={{borderColor:position===1?t.category.cyan:t.stroke.primary,background:t.bg.editor}}><div className="tiny">raw unary<br/>Velo route</div>{position===1&&<button className="control" onClick={deliver} style={{borderColor:t.category.cyan,color:t.category.cyan,marginTop:10}}>Deliver</button>}</section>
      <section className="route-stop controller" onDragOver={(e:{preventDefault:()=>void})=>e.preventDefault()} onDrop={deliver} style={{borderColor:position===2?t.category.cyan:t.stroke.primary,background:t.bg.editor}}><div className="eyebrow">controller handler</div><div className="tiny" style={{margin:"10px 0",color:t.text.secondary}}>{position===2?"register_peer(shipper)":"await payload"}</div><button className="control" disabled={position!==2||acked} onClick={()=>setAcked(true)} style={{borderColor:t.category.cyan,color:t.category.cyan}}>{acked?"ACK returned":"Return CellAck"}</button></section>
      <button draggable={position<2} disabled={position===2} aria-label="Move partition toward controller" onClick={()=>position===0?setPosition(1):deliver()} onKeyDown={(e:{key:string;preventDefault:()=>void})=>{if(e.key===" "){e.preventDefault();position===0?setPosition(1):deliver()}}} onDragStart={(e:{dataTransfer:{setData:(a:string,b:string)=>void}})=>e.dataTransfer.setData("text/plain","partition")} className={`courier-packet ${position===1?"mid":position===2?"done":""}`} style={{left:position===0?"22%":position===1?"58%":"78%",borderColor:t.category.cyan,color:t.category.cyan,background:t.bg.editor}}><span>P</span></button>
      <button className="control" style={{position:"absolute",right:0,bottom:0,borderColor:t.stroke.primary}} onClick={()=>{setPosition(0);setAcked(false);setAttempt(Math.min(99,attempt+1))}}>Retry · attempt {attempt}</button>
    </div>
  </article>;
}

function Merge() {
  const t=useHostTheme();
  const [mode,setMode]=useCanvasState<"records"|"store">("velo3.merge.mode","records");
  const [fed,setFed]=useCanvasState<boolean[]>("velo3.merge.fed",[false,false,false,false]);
  const labels=mode==="records"?["#8","#2","#11","#4"]:["Σ c0","Σ c1","Σ c2","Σ c3"];
  const count=fed.filter(Boolean).length;
  return <article className="instrument merge-page">
    <header className="merge-copy"><div className="eyebrow" style={{color:t.category.cyan}}>M / merge machine</div><h1 className="title">Feed the associative center</h1><p className="sentence" style={{color:t.text.secondary}}>Raw records restore global dispatch order; folded stores append exact algebra and merge approximate t-digests.</p></header>
    <div className="merge-dial" style={{borderColor:t.stroke.primary}}>{labels.map((x,i)=><button key={x} className={`feed f${i}`} aria-pressed={fed[i]} onClick={()=>setFed(fed.map((v,n)=>n===i?!v:v))} style={{borderColor:fed[i]?t.category.cyan:t.stroke.primary,color:fed[i]?t.category.cyan:t.text.primary,background:t.bg.elevated}}>{x}</button>)}<div className="merge-hub" style={{borderColor:count===4?t.category.cyan:t.stroke.primary,background:t.bg.elevated}}><div className="tiny" style={{color:count===4?t.category.cyan:t.text.tertiary}}>{count}/4 inputs<br/>{mode==="records"?"sort ordinal":"append_store"}</div></div></div>
    <aside className="merge-output"><button className="control" aria-pressed={mode==="store"} onClick={()=>{setMode(mode==="records"?"store":"records");setFed([false,false,false,false])}} style={{borderColor:t.category.cyan,color:t.category.cyan}}>{mode==="records"?"Switch to folded stores":"Switch to exact records"}</button><p className="tiny" style={{color:count===4?t.category.cyan:t.text.tertiary}}>{count===4?(mode==="records"?"output: #2 · #4 · #8 · #11":"output: exact count/sum/extrema · approximate percentiles"):"Select radial feeds to complete the reduction."}</p></aside>
  </article>;
}

function Phaser() {
  const t=useHostTheme();
  const [generation,setGeneration]=useCanvasState<number>("velo3.phaser.generation",2);
  const [attachGeneration,setAttachGeneration]=useCanvasState<number|null>("velo3.phaser.attach",null);
  const safeGeneration=Math.min(12,Math.max(1,generation));
  const events=Array.from({length:safeGeneration},(_,i)=>i+1);
  return <article className="instrument phaser-page">
    <section className="clock-face" style={{borderColor:t.stroke.primary}}>{events.map((g,i)=>{const angle=(i/Math.max(8,safeGeneration))*Math.PI*2-Math.PI/2;const left=50+42*Math.cos(angle),top=50+42*Math.sin(angle);const replay=attachGeneration!==null&&g<=attachGeneration;const live=attachGeneration!==null&&g>attachGeneration;return <span key={g} className="gen" aria-label={`Generation ${g}, ${replay?"replay":live?"live":"history"}`} style={{left:`calc(${left}% - 23px)`,top:`calc(${top}% - 23px)`,borderColor:replay?t.stroke.primary:live?t.category.cyan:t.stroke.tertiary,color:live?t.category.cyan:t.text.primary,background:t.bg.elevated}}>g{g}</span>})}<div className="clock-center" style={{borderColor:t.category.cyan,background:t.bg.elevated}}><div><div className="eyebrow" style={{color:t.category.cyan}}>generation {safeGeneration}</div><button className="control" disabled={safeGeneration>=12} onClick={()=>setGeneration(Math.min(12,safeGeneration+1))} style={{borderColor:t.stroke.primary,marginTop:9}}>Advance</button></div></div></section>
    <aside className="phaser-log" style={{borderColor:t.stroke.primary}}><div><div className="eyebrow" style={{color:t.category.cyan}}>Φ / phaser clock</div><h1 className="title">Replay, then live</h1><p className="sentence" style={{color:t.text.secondary}}>Attach captures the current generation; that entire prefix returns in the unary reply, and only later generations arrive by active-message push.</p></div><button className="control" disabled={attachGeneration!==null} onClick={()=>setAttachGeneration(safeGeneration)} style={{borderColor:t.category.cyan,color:t.category.cyan}}>Attach subscriber now</button>{attachGeneration!==null&&events.map(g=><div className="log-row" key={g}><span style={{color:g<=attachGeneration?t.text.tertiary:t.category.cyan}}>g{g}</span><span style={{color:g<=attachGeneration?t.text.secondary:t.category.cyan}}>{g<=attachGeneration?"reply replay":"live push"}</span></div>)}</aside>
  </article>;
}

function Dataset() {
  const t=useHostTheme();
  const [published,setPublished]=useCanvasState<number>("velo3.dataset.published",2);
  const [attach,setAttach]=useCanvasState<Array<number|null>>("velo4.dataset.attach",[null,null,null]);
  const safePublished=Math.min(6,Math.max(0,published));
  return <article className="instrument dataset-page">
    <header><div className="eyebrow" style={{color:t.category.cyan}}>D / dataset floodgate</div><h1 className="title">Broadcast once. Retain by modulo.</h1><p className="sentence" style={{color:t.text.secondary}}>MessagePack plus zstd carries the published prefix as replay at attachment and every later chunk as live; each cell retains only request_id % 3 == cell_id.</p><div className="tiny" style={{color:t.text.tertiary,marginTop:8}}>H = history before attach · R = reply replay · L = live push · own/pass = modulo decision</div><button className="control" onClick={()=>setAttach([null,null,null])} style={{borderColor:t.stroke.primary,marginTop:12}}>Reset subscriptions</button></header>
    <div className="flood-instrument"><aside className="reservoir" style={{borderColor:t.stroke.primary}}>{Array.from({length:6},(_,i)=>i).map(i=><button key={i} className="chunk" disabled={i>safePublished} aria-pressed={i<safePublished} onClick={()=>setPublished(Math.min(6,Math.max(safePublished,i+1)))} style={{borderColor:i<safePublished?t.category.cyan:t.stroke.primary,color:i<safePublished?t.category.cyan:t.text.tertiary,background:t.bg.elevated}}>zpack chunk {i}</button>)}<button className="control" onClick={()=>setPublished(Math.min(6,safePublished+1))} disabled={safePublished===6} style={{borderColor:t.category.cyan,color:t.category.cyan}}>Open floodgate</button></aside>
      <section className="channels" style={{borderColor:t.stroke.primary}}>{[0,1,2].map(cell=>{const boundary=attach[cell];return <div className="channel" key={cell} style={{borderColor:t.stroke.tertiary}}><button className="control" disabled={boundary!==null} onClick={()=>setAttach(attach.map((v,n)=>n===cell?safePublished:v))} style={{borderColor:boundary!==null?t.category.cyan:t.stroke.primary,color:boundary!==null?t.category.cyan:t.text.primary}}>cell {cell}<br/><span className="tiny">{boundary===null?`attach now @ ${safePublished}`:`attached @ ${boundary}`}</span></button>{Array.from({length:6},(_,id)=>id).map(id=>{const publishedNow=id<safePublished;const kind=boundary===null?"H":id<boundary?"R":"L";const owns=id%3===cell;return <div key={id} className={`channel-slot ${owns?"owned":""}`} style={{color:publishedNow?(owns?t.category.cyan:t.text.tertiary):t.stroke.tertiary}}>{publishedNow?<>{kind} · {owns?"own":"pass"}<br/>{id}</>:"·"}</div>})}</div>})}</section>
    </div>
  </article>;
}

function Tree() {
  const t=useHostTheme();
  const [shape,setShape]=useCanvasState<"flat"|"tree">("velo3.tree.shape","tree");
  const [payload,setPayload]=useCanvasState<number>("velo3.tree.payload",64);
  const safePayload=Math.min(96,Math.max(8,payload));
  const cells=[0,1,2,3,4,5,6,7];
  return <article className="instrument tree-page">
    <section className="topology"><svg viewBox="0 0 760 650"><circle cx="380" cy="70" r="34" fill={t.bg.elevated} stroke={t.category.cyan}/><text x="380" y="75" textAnchor="middle" fill={t.text.primary} className="tiny">controller</text>{shape==="tree"&&<><circle cx="245" cy="280" r="30" fill={t.bg.elevated} stroke={t.stroke.primary}/><circle cx="515" cy="280" r="30" fill={t.bg.elevated} stroke={t.stroke.primary}/><text x="245" y="284" textAnchor="middle" fill={t.text.primary} className="tiny">agg 0</text><text x="515" y="284" textAnchor="middle" fill={t.text.primary} className="tiny">agg 1</text><path d="M260 253L360 99M500 253L400 99" stroke={t.category.cyan} strokeWidth={Math.max(1,safePayload/32)}/></>}{cells.map((c,i)=>{const x=65+i*90,parent=i<4?245:515;return <g key={c}><circle cx={x} cy="555" r={Math.max(8,safePayload/9)} fill={t.fill.tertiary} stroke={t.stroke.primary}/><text x={x} y="559" textAnchor="middle" fill={t.text.primary} className="tiny">c{c}</text><path d={shape==="tree"?`M${x} ${542-safePayload/9}L${parent} 310`:`M${x} ${542-safePayload/9}L365 102`} fill="none" stroke={shape==="tree"?t.stroke.primary:t.category.cyan} strokeWidth={Math.max(1,safePayload/32)}/></g>})}</svg><div className="topology-compact"><div className="compact-node" style={{borderColor:t.category.cyan,color:t.category.cyan}}>controller<br/>{shape==="tree"?"2 stores in":"8 partitions in"}</div><div className="compact-link" style={{color:t.text.tertiary}}>{shape==="tree"?"subtree reduction":"global-order merge"}</div>{shape==="tree"&&<div className="compact-branch"><div className="compact-node" style={{borderColor:t.stroke.primary}}>aggregator 0<br/>cells 0–3</div><div className="compact-node" style={{borderColor:t.stroke.primary}}>aggregator 1<br/>cells 4–7</div></div>}<div className="compact-cells">{cells.map(c=><div className="compact-node" key={c} style={{borderColor:t.stroke.primary}}>cell {c}<br/>{safePayload}u</div>)}</div></div></section>
    <aside className="tree-console" style={{borderColor:t.stroke.primary}}><div><div className="eyebrow" style={{color:t.category.cyan}}>T / aggregator tree</div><h1 className="title">Collapse payload upward</h1><p className="sentence" style={{color:t.text.secondary}}>Folded stores can reduce through aggregators; retained raw records stay flat so the controller can restore global dispatch order.</p></div><div><button className="control" aria-pressed={shape==="flat"} onClick={()=>setShape("flat")} style={{borderColor:shape==="flat"?t.category.cyan:t.stroke.primary,color:shape==="flat"?t.category.cyan:t.text.primary,marginRight:6}}>Flat records</button><button className="control" aria-pressed={shape==="tree"} onClick={()=>setShape("tree")} style={{borderColor:shape==="tree"?t.category.cyan:t.stroke.primary,color:shape==="tree"?t.category.cyan:t.text.primary}}>Folded tree</button></div><label className="eyebrow" style={{color:t.text.tertiary}}>payload volume<input aria-label="Payload volume" type="range" min="8" max="96" value={safePayload} onChange={(e:{target:{value:string}})=>setPayload(Number(e.target.value))} style={{width:"100%",marginTop:10}}/></label><div className="tiny" style={{color:t.category.cyan}}>{shape==="tree"?"8 cell stores → 2 subtree stores → 1 report":"8 raw partitions → controller global-order merge"}</div></aside>
  </article>;
}

function Nav({ view,setView }: { view: Exclude<View,"index">; setView:(view:View)=>void }) {
  const t=useHostTheme();
  return <div className="nav-layout"><nav className="marks" aria-label="Velo mechanisms" style={{borderColor:t.stroke.tertiary}}><button className="back" aria-label="Back to mechanism index" onClick={()=>setView("index")}>←</button>{MECHANISMS.map(m=><button key={m.id} className="mark" aria-current={view===m.id?"page":undefined} aria-label={m.title} title={m.title} onClick={()=>setView(m.id)} style={{borderColor:view===m.id?t.category.cyan:"transparent",color:view===m.id?t.category.cyan:t.text.primary}}>{m.mark}</button>)}</nav>{view==="radar"?<Radar/>:view==="xray"?<Xray/>:view==="gate"?<Gate/>:view==="press"?<Press/>:view==="scope"?<Scope/>:view==="courier"?<Courier/>:view==="merge"?<Merge/>:view==="phaser"?<Phaser/>:view==="dataset"?<Dataset/>:<Tree/>}</div>;
}

export default function VeloMechanisms() {
  const t=useHostTheme();
  const [view,setView]=useCanvasState<View>("velo3.view","index");
  return <div className="root" style={{background:t.bg.editor,color:t.text.primary}}><style>{CSS}</style><div className="workbench">{view==="index"?<Index open={setView}/>:<Nav view={view} setView={setView}/>}</div></div>;
}
