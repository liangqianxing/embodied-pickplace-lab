'use strict';
const $ = (id) => document.getElementById(id);
const labels = {random:'随机策略',expert:'脚本专家',bc32:'BC · 32',bc256:'BC · 256',dagger:'DAgger'};
const colors = {green:'#386049',orange:'#e77d44',ink:'#263d30',muted:'#8b927f',line:'#dde1d0'};
let report, replays = [], current, frameIndex = 0, playing = false, lastTime = 0, elapsed = 0;
const scene = $('scene'), sceneContext = scene.getContext('2d');
const lossCanvas = $('loss-chart'), lossContext = lossCanvas.getContext('2d');
function percent(value) { return `${(value*100).toFixed(value === 1 || value === 0 ? 0 : 1)}%`; }
function ci(row) { return row.confidence_interval.map(percent).join(' – '); }
async function fetchJSON(url) { const response = await fetch(url); const data = await response.json(); if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`); return data; }
function showError(message) { $('error-banner').textContent = message; $('error-banner').hidden = false; }
function fitCanvas(canvas, context) { const ratio = window.devicePixelRatio || 1; const width = canvas.clientWidth, height = canvas.clientHeight; canvas.width = width * ratio; canvas.height = height * ratio; context.setTransform(ratio,0,0,ratio,0,0); return [width,height]; }
function setPlayback(value) { playing = value; $('play').textContent = value ? 'Ⅱ' : '▶'; $('play').setAttribute('aria-label',value ? '暂停轨迹' : '播放轨迹'); }
function loadReplay(replay, autoplay = true) { current = replay; frameIndex = 0; elapsed = 0; $('timeline').max = replay.frames.length - 1; $('timeline').value = 0; $('replay-badge').textContent = `${replay.success ? '成功' : '未完成'} · ${labels[replay.policy]} · #${replay.seed}`; $('replay-badge').className = `badge ${replay.success ? '' : 'failure'}`; setPlayback(autoplay); drawScene(); }
function drawScene() {
  const w = scene.clientWidth, h = scene.clientHeight, ctx = sceneContext;
  ctx.clearRect(0,0,w,h);
  const scale = Math.min(w / 1.12, h / 0.88);
  const origin = [w * .51,h * .73];
  const project = ([x,y,z]) => [origin[0]+(x-y)*.85*scale,origin[1]+(x+y)*.39*scale-z*scale];
  function polygon(points,fill,stroke) { ctx.beginPath(); points.map(project).forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.8;ctx.stroke();} }
  function box(x,y,z,dx,dy,dz,palette) { polygon([[x-dx,y-dy,z-dz],[x+dx,y-dy,z-dz],[x+dx,y-dy,z+dz],[x-dx,y-dy,z+dz]],palette[0]);polygon([[x+dx,y-dy,z-dz],[x+dx,y+dy,z-dz],[x+dx,y+dy,z+dz],[x+dx,y-dy,z+dz]],palette[1]);polygon([[x-dx,y-dy,z+dz],[x+dx,y-dy,z+dz],[x+dx,y+dy,z+dz],[x-dx,y+dy,z+dz]],palette[2]); }
  function line(a,b,color,width=1) { const p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke(); }
  box(0,0,-.035,.38,.32,.025,['#c8cdb8','#b8c0aa','#e1e5d1']);
  for(let x=-.35;x<=.35;x+=.05)line([x,-.32,-.009],[x,.32,-.009],'#c6ceb5',.5);
  for(let y=-.30;y<=.30;y+=.05)line([-.38,y,-.009],[.38,y,-.009],'#c6ceb5',.5);
  const metal=['#627961','#394f3c','#8c9d79'];
  box(-.32,.25,.22,.012,.012,.23,metal);box(.32,.25,.22,.012,.012,.23,metal);box(0,.25,.45,.335,.015,.016,metal);
  if(!current)return;
  const frame=current.frames[frameIndex], [gx,gy,gz]=frame.grip, [cx,cy,cz]=frame.cube, [tx,ty]=frame.goal;
  polygon([[tx-.045,ty-.045,-.007],[tx+.045,ty-.045,-.007],[tx+.045,ty+.045,-.007],[tx-.045,ty+.045,-.007]],'#89a37755','#6d8b57');
  const target=project([tx,ty,-.006]);ctx.fillStyle='#526c40';ctx.font='7px "Avenir Next",sans-serif';ctx.textAlign='center';ctx.fillText('TARGET',target[0],target[1]+4);
  if(frameIndex>0){ctx.beginPath();current.frames.slice(0,frameIndex+1).forEach((f,i)=>{const p=project(f.cube);i?ctx.lineTo(...p):ctx.moveTo(...p)});ctx.strokeStyle='#db8d5080';ctx.setLineDash([3,4]);ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);}
  polygon([[cx-.027,cy-.027,-.005],[cx+.027,cy-.027,-.005],[cx+.027,cy+.027,-.005],[cx-.027,cy+.027,-.005]],'#57654a25');
  box(cx,cy,cz,.025,.025,.025,['#d36b35','#bb512b','#f2a066']);
  line([gx,gy,gz+.07],[gx,gy,.45],'#748b68',5);line([gx,gy,.45],[gx,.25,.45],'#9cae89',4);
  box(gx,gy,gz,.078,.04,.014,['#3c5744','#243e2f','#6a8263']);
  const halfOpening=frame.opening/2;
  box(gx-.01-halfOpening,gy,gz-.055,.01,.032,.045,['#57715a','#314934','#8a9d79']);
  box(gx+.01+halfOpening,gy,gz-.055,.01,.032,.045,['#57715a','#314934','#8a9d79']);
  $('opening').textContent=`${(frame.opening*1000).toFixed(0)} mm`;
  $('contact').textContent=`${frame.contacts} / 2`;
  $('distance').textContent=`${(Math.hypot(cx-tx,cy-ty)*100).toFixed(1)} cm`;
  $('time').textContent=`${frame.time.toFixed(2)} s`;
  $('timeline').value=frameIndex;
}
function drawLoss(){
  const ctx=lossContext,w=lossCanvas.clientWidth,h=lossCanvas.clientHeight;ctx.clearRect(0,0,w,h);
  if(!report)return;const data=report.training[$('curve-policy').value];if(!data)return;
  const history=data.history,pad={l:49,r:21,t:24,b:33};const plotW=w-pad.l-pad.r,plotH=h-pad.t-pad.b;
  const all=history.flatMap(r=>[r.train_loss,r.val_loss]);const min=Math.floor(Math.log10(Math.min(...all))),max=Math.ceil(Math.log10(Math.max(...all)));
  const x=(epoch)=>pad.l+(epoch-1)/Math.max(1,history.length-1)*plotW;
  const y=(loss)=>pad.t+(max-Math.log10(loss))/Math.max(1,max-min)*plotH;
  ctx.font='8px "Avenir Next",sans-serif';ctx.textAlign='right';ctx.fillStyle=colors.muted;
  for(let tick=min;tick<=max;tick++){const yy=y(10**tick);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.strokeStyle=colors.line;ctx.lineWidth=.7;ctx.stroke();ctx.fillText(`1e${tick}`,pad.l-9,yy+3);}
  ctx.textAlign='center';for(const epoch of [1,Math.round(history.length/2),history.length])ctx.fillText(epoch,x(epoch),h-15);ctx.fillText('EPOCH',w/2,h-1);
  for(const [key,color] of [['train_loss',colors.green],['val_loss',colors.orange]]){ctx.beginPath();history.forEach((r,i)=>i?ctx.lineTo(x(r.epoch),y(r[key])):ctx.moveTo(x(r.epoch),y(r[key])));ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.stroke();}
  $('training-detail').replaceChildren();for(const [name,value]of [['最佳 epoch',data.best_epoch],['验证损失',data.best_val_loss.toFixed(5)],['训练耗时',`${data.seconds.toFixed(1)} s`]]){const span=document.createElement('span');span.append(`${name} `);const b=document.createElement('b');b.textContent=value;span.append(b);$('training-detail').append(span);}
}
function fillMetrics(){
  const rows=report.evaluation.results, row=(name,dist)=>rows.find(r=>r.policy===name&&r.distribution===dist);
  for(const dist of ['id','ood']){const r=row('dagger',dist);if(r){$(`metric-${dist}`).textContent=percent(r.success_rate);$(`metric-${dist}-ci`).textContent=`${r.successes} / ${r.episodes} · 95% CI ${ci(r)}`;}}
  $('metric-demos').textContent=report.dataset.train_episodes;$('metric-transitions').textContent=`+ ${report.dataset.validation_episodes} 条验证 · ${report.dataset.transitions.toLocaleString()} 个样本`;
  $('metric-trials').textContent=rows.reduce((n,r)=>n+r.episodes,0).toLocaleString();
  $('version').textContent=`MUJOCO ${report.versions.mujoco} / CPU`;
  $('generated-at').textContent=`GENERATED ${report.created_at.slice(0,10)} · CST`;
  const bars=$('comparison-bars'),tbody=$('results-table');bars.replaceChildren();tbody.replaceChildren();
  for(const name of Object.keys(labels)){const a=row(name,'id'),b=row(name,'ood');if(!a||!b)continue;const element=document.createElement('div');element.className='bar-row';
    const label=document.createElement('span');label.className='bar-name';label.textContent=labels[name];element.append(label);
    const tracks=document.createElement('div');tracks.className='bar-tracks';for(const [r,color]of [[a,colors.green],[b,colors.orange]]){const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=percent(r.success_rate);fill.style.background=color;track.append(fill);tracks.append(track);}element.append(tracks);
    const value=document.createElement('span');value.className='bar-percent';value.textContent=percent(a.success_rate);element.append(value);bars.append(element);
    const tr=document.createElement('tr');const nameCell=document.createElement('td');nameCell.textContent=labels[name];tr.append(nameCell);for(const r of[a,b]){const td=document.createElement('td');td.textContent=`${r.successes}/${r.episodes} · ${percent(r.success_rate)}`;const small=document.createElement('small');small.textContent=ci(r);td.append(small);tr.append(td);}tbody.append(tr);
  }
  const failures=row('dagger','ood')?.failures||{};$('failure-counts').replaceChildren();for(const [key,name]of [['no_lift','未完成抬升'],['misplaced_or_dropped','偏离 / 掉落'],['release_or_settle','释放 / 稳定']]){const div=document.createElement('div');div.textContent=name;const strong=document.createElement('strong');strong.textContent=failures[key]??'—';div.append(strong);$('failure-counts').append(div);}
}
function animate(time){if(playing&&current){elapsed+=Math.min(time-lastTime,100)*Number($('speed').value);if(elapsed>=50){frameIndex=Math.min(current.frames.length-1,frameIndex+Math.floor(elapsed/50));elapsed%=50;if(frameIndex===current.frames.length-1)setPlayback(false);drawScene();}}lastTime=time;requestAnimationFrame(animate);}
$('play').addEventListener('click',()=>{if(!current)return;if(frameIndex===current.frames.length-1)frameIndex=0;setPlayback(!playing);drawScene();});
$('timeline').addEventListener('input',()=>{setPlayback(false);frameIndex=Number($('timeline').value);drawScene();});
$('curve-policy').addEventListener('change',drawLoss);
$('replay').addEventListener('change',()=>loadReplay(replays[Number($('replay').value)]));
$('next-seed').addEventListener('click',()=>{$('seed').value=(Number($('seed').value)+1)%2147483648;});
$('run-form').addEventListener('submit',async(event)=>{event.preventDefault();$('run').disabled=true;$('run-status').textContent='正在加载策略并运行 MuJoCo…';try{const params=new URLSearchParams({policy:$('policy').value,distribution:$('distribution').value,seed:$('seed').value});const replay=await fetchJSON(`/api/rollout?${params}`);loadReplay(replay);$('run-status').textContent=`${replay.success?'任务完成':'未完成任务'} · ${replay.steps} 步 · 仿真 ${replay.simulated_seconds.toFixed(2)} 秒`;$('error-banner').hidden=true;}catch(error){$('run-status').textContent=`运行失败：${error.message}`;}finally{$('run').disabled=false;}});
$('show-render').addEventListener('click',()=>{setPlayback(false);$('render-dialog').showModal();});$('close-render').addEventListener('click',()=>$('render-dialog').close());
$('render-dialog').addEventListener('click',(e)=>{if(e.target===$('render-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
for(const link of document.querySelectorAll('nav a'))link.addEventListener('click',()=>{document.querySelector('nav a.active')?.classList.remove('active');link.classList.add('active');});
new ResizeObserver(()=>{fitCanvas(scene,sceneContext);fitCanvas(lossCanvas,lossContext);drawScene();drawLoss();}).observe(document.querySelector('main'));
async function init(){try{[report,replays]=await Promise.all([fetchJSON('/api/report'),fetchJSON('/api/replays')]);fillMetrics();$('replay').replaceChildren();replays.forEach((r,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${labels[r.policy]} · ${r.distribution.toUpperCase()} · ${r.success?'成功':'失败'} #${r.seed}`;$('replay').append(option);});const selected=Math.max(0,replays.findIndex(r=>r.policy==='dagger'&&r.distribution==='id'&&r.success));$('replay').value=selected;loadReplay(replays[selected],!matchMedia('(prefers-reduced-motion: reduce)').matches);drawLoss();try{const media=await fetchJSON('/media/demo.json');$('render-caption').textContent=`固定评测样例：${labels[media.policy]} · 场景 #${media.seed} · ${media.success?'成功':'未完成'}`;}catch{$('render-caption').textContent='未生成渲染，请运行 pickplace render。';}}catch(error){showError(`实验数据未加载：${error.message}。请先完成 collect → train → dagger → evaluate → report，再通过 pickplace serve 打开页面。`);$('run').disabled=true;}}
fitCanvas(scene,sceneContext);fitCanvas(lossCanvas,lossContext);requestAnimationFrame(animate);init();
