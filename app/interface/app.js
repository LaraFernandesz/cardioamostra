'use strict';
export default function(component) {
  const {parentElement: root, data, setStateValue} = component;
  // O Streamlit chama esta função de novo a cada rerun sem desmontar o componente;
  // sem esta limpeza, os ouvintes se acumulam (ex.: um clique empilhava duas entradas no histórico).
  if (typeof root.__cardioCleanup === 'function') root.__cardioCleanup();
  const abort = new AbortController();
  const on = (element,event,handler) => element.addEventListener(event,handler,{signal:abort.signal});
  const $ = id => root.querySelector(`#${id}`);
  const fmt = (n, digits = 0) => Number(n).toLocaleString('pt-BR', {minimumFractionDigits:digits, maximumFractionDigits:digits});
  const techniques = [
    ['nenhuma', 'Nenhuma correção', 'Treina com os dados como estão e decide com limiar de 0,5.', 'Ponto de partida'],
    ['pesos', 'Pesos de classe', 'Dá mais peso aos erros na classe minoritária, sem criar nem remover exemplos.', 'Ajusta o peso dos erros'],
    ['limiar', 'Ajuste de limiar', 'Muda o corte da decisão, escolhido no treino. A amostra e as probabilidades permanecem iguais.', 'Muda a decisão'],
    ['subamostragem', 'Subamostragem aleatória', 'Remove exemplos da classe majoritária ao acaso até equilibrar as duas classes.', 'Remove exemplos'],
    ['sobreamostragem', 'Sobreamostragem aleatória', 'Repete exemplos da classe minoritária ao acaso até equilibrar as duas classes.', 'Repete exemplos'],
    ['smote', 'SMOTE', 'Cria exemplos sintéticos entre pessoas vizinhas da classe minoritária.', 'Gera novos exemplos'],
    ['smote_tomek', 'SMOTE + Tomek', 'Cria exemplos com SMOTE e remove pares de vizinhos de classes opostas com Tomek.', 'Gera e remove exemplos']
  ];
  const bases = data.bases;
  const modelNames = data.modelos;
  const state = {base:'brfss', model:'regressao_logistica', technique:'smote', page:data.page || 'laboratorio', stage:100, sort:'recall', calibration:'smote', expanded:false, ...(root.__cardioSelection || data.initial || {}), page:data.page || 'laboratorio'};
  let initializing = true;
  const persist = () => {
    root.__cardioSelection = {...state};
    if (!initializing) setStateValue('selection', {...state});
  };

  let animation = 0, toastTimer, viewBounds;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  $('techniques').innerHTML = techniques.map((t,i) => `<button class="tech-button" data-tech="${t[0]}" aria-pressed="${t[0]===state.technique}"><span class="tech-number">${String(i+1).padStart(2,'0')}</span><span>${t[1]}</span></button>`).join('');
  if (!data?.resultados?.length || !data?.projecoes) {
    root.querySelectorAll('.page-panel').forEach(p => p.hidden = true);
    $('status').hidden = false;
    $('status').innerHTML = '<h3>Não foi possível carregar os resultados.</h3><p>Os resultados estão indisponíveis. Tente carregar o painel novamente.</p><button class="outline-button" id="retry">Tentar novamente</button>';
    $('retry').onclick = () => location.reload();
    root.querySelectorAll('.sidebar button,.sidebar select').forEach(e=>e.disabled=true);
    return;
  }
  const rows = () => data.resultados.filter(r => r.base===state.base && r.modelo===state.model);
  const current = () => rows().find(r => r.estrategia===state.technique);
  const baseline = () => rows().find(r => r.estrategia==='nenhuma');
  const tech = () => techniques.find(t => t[0]===state.technique);
  const outcome = () => state.base==='insuficiencia' ? ['Sem óbito','Óbito'] : ['Sem doença','Com doença'];
  const isResampling = () => !['nenhuma','pesos','limiar'].includes(state.technique);
  function notify(text) {
    clearTimeout(toastTimer); $('toast').textContent = text; $('toast').hidden = false;
    toastTimer = setTimeout(() => $('toast').hidden = true, 4200);
  }
  function stopAnimation() { cancelAnimationFrame(animation); animation=0; }
  function selectTechnique(key) {
    stopAnimation(); state.technique=key; state.stage=100;
    root.querySelectorAll('[data-tech]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tech===key)));
    render(); persist();
  }
  function stageUpdate(value) {
    state.stage = Number(value); $('stage').value=String(value);
    const label = value===0?'Antes':value===100?'Depois':'Transição visual';
    $('stage-label').textContent = isResampling()?`${label} · ${fmt(value)}%`:'A amostra não muda';
    $('stage').setAttribute('aria-valuetext', `${label}, ${value}%`);
    $('plot-state').textContent = !isResampling()?'AMOSTRA ORIGINAL':value===0?'ANTES DA TÉCNICA':value===100?`DEPOIS · ${tech()[1]}`:'TRANSIÇÃO VISUAL';
    draw();
  }
  function renderBars(r) {
    const c=r.contagens, max=Math.max(c.linhas_antes,c.linhas_depois);
    function bar(title,total,positive,added) {
      const negative=total-positive;
      return `<div><div class="bar-label"><strong>${title}</strong><span>${fmt(total)} no treino</span></div><div class="population-track" style="width:${total/max*100}%" aria-label="${title}: ${fmt(total)} pessoas no treino, ${fmt(positive)} da classe positiva"><span class="bar-negative" style="width:${negative/total*100}%"></span><span class="bar-positive" style="width:${(positive-added)/total*100}%"></span>${added?`<span class="bar-synthetic" style="width:${added/total*100}%"></span>`:''}</div></div>`;
    }
    $('population-bars').innerHTML=bar('Antes',c.linhas_antes,c.doentes_antes,0)+bar('Depois',c.linhas_depois,c.doentes_depois,Math.max(0,c.doentes_depois-c.doentes_antes));
  }
  function renderMetrics(r,b) {
    const death=state.base==='insuficiencia';
    const metrics=[['recall','Recall',death?'Óbitos identificados':'Casos positivos encontrados',2],['precisao','Precisão','Alertas que eram positivos',2],['roc_auc','ROC-AUC','Capacidade de ordenar',3],['brier','Brier','Erro das probabilidades · menor é melhor',3]];
    $('metrics').innerHTML=metrics.map(([key,name,desc,digits])=>{
      const delta=r[key]-b[key], equal=Number(Math.abs(delta).toFixed(digits))===0;
      const good=key==='brier'?delta<0:delta>0;
      return `<div class="metric"><div><div class="metric-name">${name}</div><div class="metric-description">${desc}</div></div><div class="metric-value">${fmt(r[key],digits)}</div><div class="metric-bottom"><span class="metric-baseline">Sem correção: ${fmt(b[key],digits)}</span><span class="delta ${equal?'':good?'good':'bad'}">${equal?'sem mudança':`${delta>0?'↑':'↓'} ${fmt(Math.abs(delta),digits)} · ${good?'melhora':'piora'}`}</span></div></div>`;
    }).join('');
  }
  function renderInsight(r,b) {
    const more=r.recall>b.recall+.01, worse=r.brier>b.brier+.005;
    let title, text;
    if(state.technique==='nenhuma'){
      title='O ponto de partida para toda comparação.';
      text='Escolha uma técnica para observar o que muda na amostra, na detecção e no erro das probabilidades.';
    } else if(state.technique==='limiar') {
      title='A decisão muda. As probabilidades permanecem iguais.';
      text=`O recall passa de ${fmt(b.recall,2)} para ${fmt(r.recall,2)}, mantendo o Brier em ${fmt(r.brier,3)}. Mais detecção também pode trazer mais alarmes falsos.`;
    } else if(more&&worse){
      title='Mais casos encontrados, maior erro nas probabilidades.';
      text=`O recall sobe para ${fmt(r.recall,2)}, enquanto o Brier passa de ${fmt(b.brier,3)} para ${fmt(r.brier,3)}. Vale comparar esse ganho com o ajuste de limiar.`;
    } else {
      title='Observe a detecção e as probabilidades em conjunto.';
      text=`Nesta configuração, o recall é ${fmt(r.recall,2)} e o Brier é ${fmt(r.brier,3)}. Compare as técnicas antes de interpretar uma métrica isolada.`;
    }
    $('insight-title').textContent=title; $('insight-text').textContent=text;
    $('explore-threshold').hidden=state.technique==='limiar';
  }
  function impact(r) {
    const positive=1000*r.prevalencia;
    const tp=r.recall*positive;
    return [Math.round(tp),Math.round(positive-tp),Math.round(r.precisao>0?tp/r.precisao-tp:0)];
  }
  function renderImpact(r,b) {
    const values=impact(r), before=impact(b), death=state.base==='insuficiencia';
    const labels=death?['óbitos identificados','óbitos não identificados','alarmes falsos']:['casos encontrados','casos não encontrados','alarmes falsos'];
    $('impact-values').innerHTML=values.map((n,i)=>`<div class="impact-item"><strong>≈ ${fmt(n)}</strong><span>${labels[i]}</span><small>antes: ${fmt(before[i])}</small></div>`).join('');
  }
  function renderComparison() {
    const sort=$('sort').value;
    const sorted=rows().slice().sort((a,b)=>sort==='brier'?a[sort]-b[sort]:b[sort]-a[sort]);
    $('comparison-body').innerHTML=sorted.map(r=>{
      const t=techniques.find(t=>t[0]===r.estrategia);
      return `<tr class="${r.estrategia===state.technique?'selected':''}"><td><span class="table-tech">${t[1]}</span><span class="table-kind">${t[3]}</span></td><td><span class="table-value">${fmt(r.recall,2)}</span><div class="mini-bar" aria-hidden="true"><span style="width:${r.recall*100}%"></span></div></td><td>${fmt(r.precisao,2)}</td><td>${fmt(r.f1,2)}</td><td>${fmt(r.pr_auc,3)}</td><td>${fmt(r.roc_auc,3)}</td><td>${fmt(r.brier,3)}</td><td>${Number.isFinite(r.razao_previsto_observado)?fmt(r.razao_previsto_observado,2)+"×":"—"}</td><td><button class="row-open" data-open="${r.estrategia}" aria-label="Explorar ${t[1]}">↗</button></td></tr>`;
    }).join('');
  }
  function renderCalibration() {
    const available=['smote','smote_tomek','subamostragem','sobreamostragem','pesos'].filter(key=>rows().some(r=>r.estrategia===key));
    if(!available.includes(state.calibration))state.calibration=available[0];
    $('calibration-tech').replaceChildren(...available.map(key=>new Option(techniques.find(t=>t[0]===key)[1],key)));
    $('calibration-tech').value=state.calibration;
    const selected=data.curvas.filter(r=>r.base===state.base&&r.modelo===state.model);
    const reference=selected.filter(r=>r.estrategia==='nenhuma').sort((a,b)=>a.prevista-b.prevista);
    const corrected=selected.filter(r=>r.estrategia===state.calibration).sort((a,b)=>a.prevista-b.prevista);
    const missing=!reference.length||!corrected.length;
    $('calibration-empty').hidden=!missing;$('calibration-content').hidden=missing;
    $('calibration-empty').textContent='Não há curvas de calibração disponíveis para esta configuração.';
    if(missing)return;
    const name=techniques.find(t=>t[0]===state.calibration)[1];
    $('calibration-name').textContent=name;
    const x=p=>60+p*420,y=p=>275-p*235;
    const maxPeople=Math.max(...reference.concat(corrected).map(r=>r.pessoas));
    let svg='<title>Calibração: nenhuma correção e '+name+'</title>';
    for(const n of [0,.25,.5,.75,1]){
      svg+=`<path d="M60 ${y(n)}H480 M${x(n)} 40V275" stroke="#e4e3df" fill="none"/><text x="${x(n)}" y="297" text-anchor="middle">${fmt(n,2)}</text><text x="48" y="${y(n)+5}" text-anchor="end">${fmt(n,2)}</text>`;
    }
    svg+='<text x="270" y="325" text-anchor="middle">Probabilidade prevista</text><text x="60" y="24">Proporção observada</text>';
    svg+=`<path d="M60 275L480 40" stroke="#687178" stroke-dasharray="5 5" fill="none"/>`;
    for(const [points,label,color,symbol] of [[reference,'Nenhuma correção','#2876b5','circle'],[corrected,name,'#b45f20','square']]){
      svg+=`<polyline points="${points.map(p=>`${x(p.prevista)},${y(p.observada)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;
      for(const p of points){
        const radius=3+5*Math.sqrt(p.pessoas/maxPeople);
        const title=`${label}: prevista ${fmt(p.prevista,3)}, observada ${fmt(p.observada,3)}, ${fmt(p.pessoas)} pessoas`;
        svg+=symbol==='circle'?`<circle cx="${x(p.prevista)}" cy="${y(p.observada)}" r="${radius}" fill="${color}" stroke="#fff" stroke-width="1.5"><title>${title}</title></circle>`:`<rect x="${x(p.prevista)-radius}" y="${y(p.observada)-radius}" width="${radius*2}" height="${radius*2}" rx="1" fill="${color}" stroke="#fff" stroke-width="1.5"><title>${title}</title></rect>`;
      }
    }
    $('calibration-chart').innerHTML=svg;
    $('calibration-values').innerHTML=[[reference,'Nenhuma correção'],[corrected,name]].flatMap(([points,label])=>points.map(p=>`<tr><td>${label}</td><td>${fmt(p.prevista,3)}</td><td>${fmt(p.observada,3)}</td><td>${fmt(p.pessoas)}</td></tr>`)).join('');
  }
  function computeBounds() {
    // Limites comuns às sete técnicas; nenhum ponto é descartado ou recortado.
    const all=Object.values(data.projecoes[state.base]?.tecnicas || {}).flat();
    if(!all.length){viewBounds=[-1,1,-1,1];return;}
    const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
    viewBounds=[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];
  }
  function draw() {
    const canvas=$('scatter'); const box=canvas.getBoundingClientRect();
    if(!box.width||!box.height)return;
    const dpr=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.round(box.width*dpr);canvas.height=Math.round(box.height*dpr);
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const width=box.width,height=box.height;
    const [minX,maxX,minY,maxY]=viewBounds;
    const project=p=>[35+(p[0]-minX)/(maxX-minX||1)*(width-60),height-26-(p[1]-minY)/(maxY-minY||1)*(height-51)];
    const points=data.projecoes[state.base].tecnicas[state.technique];
    const original=data.projecoes[state.base].tecnicas.nenhuma || [];
    if(!points)return;
    const stage=state.stage/100;
    const drawPoint=(p,i)=>{
      const [x,y]=project(p),synthetic=p[2]==='Sintético',discard=p[2]==='Descartado';
      const positive=synthetic||p[2]==='Com doença (real)'||(discard&&original[i]?.[2]==='Com doença (real)');
      if(synthetic&&stage===0)return;
      const opacity=synthetic?.65*stage:discard?.5*(1-stage)+.12*stage:positive?.85:.32;
      const radius=synthetic?2.5:positive?2.5:1.8;
      const enlarged=state.technique==='sobreamostragem'&&p[3]>1?Math.min(6.5,2.5+Math.sqrt(p[3]-1)*.6*stage):radius;
      ctx.globalAlpha=opacity;ctx.beginPath();ctx.arc(x,y,enlarged,0,Math.PI*2);
      if(synthetic||(discard&&stage>.5)){ctx.strokeStyle=positive?'#9e3151':'#8d9f9f';ctx.lineWidth=.9;ctx.stroke();}
      else{ctx.fillStyle=positive?'#a03051':'#839d9c';ctx.fill();}
    };
    points.forEach((p,i)=>{if(p[2]==='Sem doença'||p[2]==='Descartado')drawPoint(p,i);});
    points.forEach((p,i)=>{if(p[2]!=='Sem doença'&&p[2]!=='Descartado')drawPoint(p,i);});
    ctx.globalAlpha=1;
  }
  function render() {
    const r=current(),b=baseline(),t=tech(),labels=outcome();
    if(!r||!b){$('status').hidden=false;$('status').textContent='Não há resultados para esta configuração.';root.querySelectorAll('.page-panel').forEach(p=>p.hidden=true);return;}
    $('status').hidden=true;
    $('base-detail').textContent=`${bases[state.base][1] ? fmt(bases[state.base][1])+' pessoas · ' : ''}${fmt(r.prevalencia*100,2)}% ${state.base==='insuficiencia'?'de óbitos':'com doença'}`;
    $('prevalence-bar').style.width=`${r.prevalencia*100}%`;
    $('negative-label').textContent=labels[0];$('positive-label').textContent=labels[1];
    $('context').textContent=`${bases[state.base][0]} / ${modelNames[state.model]}`;
    $('mobile-summary').textContent=`${bases[state.base][0]} · ${t[1]}`;
    $('tech-title').textContent=t[1];$('tech-description').textContent=t[2];$('tech-tag').textContent=t[3];
    $('tech-index').textContent=`${String(techniques.indexOf(t)+1).padStart(2,'0')} / 07`;
    $('legend').innerHTML=`<span><i class="dot"></i>${labels[0]}</span><span><i class="dot positive"></i>${labels[1]} · real</span>${state.technique.includes('smote')?'<span><i class="dot synthetic"></i>Sintético</span>':''}${['subamostragem','smote_tomek'].includes(state.technique)?'<span><i class="dot discarded"></i>Removido</span>':''}${state.technique==='sobreamostragem'?'<span>Maior ponto = mais cópias</span>':''}`;
    const total=data.projecoes[state.base].total;
    $('plot-note').textContent=`${fmt(total)} pessoas na amostra original. PCA calculada no servidor; animação explicativa. Barras: médias do treino completo.`;
    $('scatter').setAttribute('aria-label',`Projeção PCA de ${fmt(total)} pessoas. ${t[1]}. ${t[2]} Eixos estáveis entre técnicas. ${labels[0]} em cinza; ${labels[1]} em bordô; sintéticos com contorno aberto.`);
    const projection=data.projecoes[state.base];
    const problem=projection.erro || projection.erros?.[state.technique];
    $('projection-error').hidden=!problem;
    $('projection-error').textContent=problem || '';
    root.querySelectorAll('.plot-wrap,.legend,.transform-control,.plot-note').forEach(e=>e.hidden=!!problem);
    ['stage','before','after','replay'].forEach(id=>$(id).disabled=!isResampling()||!!problem);
    root.querySelectorAll('[data-tech]').forEach(button=>{
      button.disabled=!rows().some(r=>r.estrategia===button.dataset.tech);
      button.setAttribute('aria-pressed',String(button.dataset.tech===state.technique));
    });
    renderBars(r);renderMetrics(r,b);renderInsight(r,b);renderImpact(r,b);renderComparison();renderCalibration();stageUpdate(state.stage);
  }
  function route() {
    const key=location.hash.slice(1);
    if(key==='conteudo')return;
    state.page=['laboratorio','comparativo','metodologia'].includes(key)?key:(data.page || 'laboratorio');
    const info={laboratorio:['01 / EXPLORAR E ENTENDER','Laboratório de amostragem','O que muda quando você muda a amostra?'],comparativo:['02 / COLOCAR LADO A LADO','Comparativo de técnicas','O ganho em detecção conta apenas parte da história.'],metodologia:['03 / CONHECER O PROCESSO','Por trás do experimento','Dados, escolhas e limites que precisam estar visíveis.']}[state.page];
    root.querySelectorAll('.page-panel').forEach(p=>p.hidden=p.id!==state.page);
    root.querySelectorAll('[data-page]').forEach(a=>{if(a.dataset.page===state.page)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
    $('page-eyebrow').textContent=info[0];$('page-title').innerHTML=`${info[1]}<span>.</span>`;$('page-description').textContent=info[2];
    document.title=`CardioAmostra — ${state.page==='laboratorio'?'Laboratório':state.page==='comparativo'?'Comparativo':'Metodologia'}`;
    if(state.page==='laboratorio')draw();
    persist();
  }
  function normalize() {
    if(!bases[state.base])state.base=Object.keys(bases)[0];
    const available=data.resultados.filter(r=>r.base===state.base);
    if(!available.some(r=>r.modelo===state.model))state.model=available[0].modelo;
    if(!rows().some(r=>r.estrategia===state.technique))state.technique='nenhuma';
    state.stage=Number.isFinite(Number(state.stage))?Math.max(0,Math.min(100,Number(state.stage))):100;
    $('base').value=state.base;$('modelo').value=state.model;
    [...$('modelo').options].forEach(o=>o.disabled=!available.some(r=>r.modelo===o.value));
  }
  function navigate(page) {
    history.pushState(null,'',`#${page}`);route();
    $('conteudo').focus({preventScroll:true});
    $('conteudo').scrollIntoView({block:'start',behavior:'instant'});
  }
  on(root,'click',e=>{
    const link=e.target.closest('a[href^="#"]');
    if(!link)return;
    e.preventDefault();
    const target=link.getAttribute('href').slice(1);
    if(target==='conteudo'){$('conteudo').focus();return;}
    navigate(target);
  });
  on($('techniques'),'click',e=>{const b=e.target.closest('[data-tech]');if(b)selectTechnique(b.dataset.tech);});
  $('mobile-config').onclick=()=>{
    state.expanded=root.querySelector('.sidebar').classList.toggle('expanded');
    $('mobile-config').setAttribute('aria-expanded',String(state.expanded));
    $('mobile-config').textContent=state.expanded?'Concluir ↑':'Configurar ↓';
    persist();
  };
  on($('base'),'change',e=>{stopAnimation();state.base=e.target.value;state.stage=100;normalize();computeBounds();render();persist();});
  on($('modelo'),'change',e=>{stopAnimation();state.model=e.target.value;normalize();render();persist();});
  on($('stage'),'input',e=>{stopAnimation();stageUpdate(Number(e.target.value));});
  on($('stage'),'change',persist);
  $('before').onclick=()=>{stopAnimation();stageUpdate(0);persist();};
  $('after').onclick=()=>{stopAnimation();stageUpdate(100);persist();};
  $('replay').onclick=()=>{
    stopAnimation();
    if(reduced.matches){stageUpdate(100);persist();return;}
    const start=performance.now();stageUpdate(0);
    function frame(now){const progress=Math.min(1,(now-start)/900);stageUpdate(Math.round(progress*100));if(progress<1)animation=requestAnimationFrame(frame);else persist();}
    animation=requestAnimationFrame(frame);
  };
  $('reset').onclick=()=>{state.base='brfss';state.model='regressao_logistica';state.technique='smote';normalize();computeBounds();selectTechnique(state.technique);};
  $('explore-threshold').onclick=()=>{selectTechnique('limiar');$('tech-title').scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'center'});};
  on($('sort'),'change',()=>{state.sort=$('sort').value;renderComparison();persist();});
  on($('calibration-tech'),'change',()=>{state.calibration=$('calibration-tech').value;renderCalibration();persist();});
  on($('comparison-body'),'click',e=>{const b=e.target.closest('[data-open]');if(b){selectTechnique(b.dataset.open);navigate('laboratorio');}});
  $('export').onclick=()=>{
    const columns=['base','modelo','estrategia','recall','precisao','f1','pr_auc','roc_auc','brier','razao_previsto_observado'];
    const csv='\uFEFF'+columns.join(';')+'\r\n'+rows().map(r=>columns.map(c=>typeof r[c]==='number'?String(r[c]).replace('.',','):r[c]??'').join(';')).join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download=`cardioamostra_${state.base}_${state.model}.csv`;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);notify('CSV preparado com os resultados desta configuração.');
  };
  $('help').onclick=()=>$('guide').showModal();
  $('close-guide').onclick=()=>$('guide').close();$('understood').onclick=()=>$('guide').close();
  on($('guide'),'click',e=>{if(e.target===$('guide')){const b=$('guide').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)$('guide').close();}});
  on(window,'hashchange',route);
  on(window,'popstate',route);
  const observer=new ResizeObserver(()=>draw());observer.observe($('scatter'));
  $('base').replaceChildren(...Object.entries(bases).map(([k,v])=>new Option(v[0],k)));
  $('modelo').replaceChildren(...Object.entries(modelNames).map(([k,v])=>new Option(v,k)));
  normalize();
  $('sort').value=state.sort;
  if(!$('sort').value){$('sort').value='recall';state.sort='recall';}
  root.querySelector('.sidebar').classList.toggle('expanded',state.expanded);
  $('mobile-config').setAttribute('aria-expanded',String(state.expanded));
  $('mobile-config').textContent=state.expanded?'Concluir ↑':'Configurar ↓';
  computeBounds();render();route();initializing=false;
  const cleanup=()=>{root.__cardioSelection={...state};abort.abort();observer.disconnect();stopAnimation();clearTimeout(toastTimer);if(root.__cardioCleanup===cleanup)root.__cardioCleanup=null;};
  root.__cardioCleanup=cleanup;
  return cleanup;
}
