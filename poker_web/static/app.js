const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const num = n => Number(n || 0).toLocaleString('en-US');
const signed = n => (n > 0 ? '+' : '') + num(n);
const SUITS = {s:'♠', h:'♥', d:'♦', c:'♣'};
const STREETS = {preflop:'翻牌前', flop:'翻牌', turn:'转牌', river:'河牌', showdown:'本手结束'};
const ACTIONS = {fold:'弃牌', check:'过牌', call:'跟注', raise:'加注至', all_in:'全押至', blind:'盲注', small_blind:'小盲', big_blind:'大盲', ante:'前注', select:'已锁定选牌', reveal:'亮牌', show:'摊牌', muck:'盖牌', win:'收获底池'};
let modes, mode = 'classic', state = null, busy = false, paused = false, timer, toastTimer;
let preferences = {}, choice = {discard:null, reveal:null};
try { preferences = JSON.parse(localStorage.getItem('river-preferences') || '{}'); } catch { /* Fresh settings. */ }

function savePreferences() {
  preferences.name = $('player-name').value;
  preferences.mode = mode;
  preferences.speed = $('speed').value;
  preferences[mode] = [...document.querySelectorAll('.policy-select')].map(e => e.value);
  try { localStorage.setItem('river-preferences', JSON.stringify(preferences)); } catch { /* Storage is optional. */ }
}

function card(code, {back=false, publicCard=false, placeholder=false}={}) {
  if (placeholder) return '<div class="card placeholder" aria-hidden="true"></div>';
  if (back || !code) return '<div class="card back" aria-label="未公开的底牌"></div>';
  const rank = code[0] === 'T' ? '10' : code[0], suit = SUITS[code[1]] || '';
  return `<div class="card ${'hd'.includes(code[1]) ? 'red' : ''} ${publicCard ? 'public' : ''}" aria-label="${esc(rank+suit)}"><span class="rank">${esc(rank)}</span><span class="suit">${suit}</span><span class="corner">${suit}</span>${publicCard ? '<span class="public-indicator">明牌</span>' : ''}</div>`;
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('toast').hidden = true, 6500);
}

async function api(path, body) {
  const response = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || '连接中断，请重试。');
  return data;
}

function renderSettings(nextMode, count=null) {
  if (modes[mode]) savePreferences();
  mode = nextMode;
  const config = modes[mode];
  for (const button of document.querySelectorAll('[data-mode]')) {
    button.classList.toggle('active', button.dataset.mode === mode);
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  }
  const saved = preferences[mode] || config.defaults;
  count = count || Math.min(config.max_players, Math.max(2, saved.length+1));
  $('player-count').innerHTML = Array.from({length:config.max_players-1}, (_,i) => `<option value="${i+2}" ${i+2===count?'selected':''}>${i+2} 人</option>`).join('');
  $('mode-note').textContent = mode === 'classic' ? '100BB 深筹码 · 固定盲注 · 标准规则' : '已复现的 Bay Area 场次 · 每 20 手升盲';
  $('lineup').innerHTML = Array.from({length:count-1}, (_,i) => {
    const value = saved[i] || config.defaults[i % config.defaults.length];
    const selected = config.policies.find(b => b.id === value) || config.policies[0];
    return `<div class="lineup-row"><span class="seat-index">${i+2}</span><select class="policy-select" aria-label="座位 ${i+2} 的策略">${config.policies.map(b => `<option value="${b.id}" ${b.id===selected.id?'selected':''} ${!b.available?'disabled':''}>${esc(b.name)}${!b.available?' · 模型缺失':''}</option>`).join('')}</select><p class="policy-description" title="${esc(selected.description)}">${esc(selected.kind)}${selected.id==='tournament_specialist'?' · 含已知缺陷':''}</p></div>`;
  }).join('');
  document.querySelectorAll('.policy-select').forEach(select => select.addEventListener('change', () => {
    const b = config.policies.find(b => b.id === select.value);
    const note = select.nextElementSibling;
    note.textContent = b.kind + (b.id === 'tournament_specialist' ? ' · 含已知缺陷' : '');
    note.title = b.description;
    savePreferences();
  }));
  if (!state) {
    $('game-mode').textContent = config.title;
    $('blind-info').textContent = `盲注 ${num(config.blinds[0])} / ${num(config.blinds[1])}`;
    $('stack-info').textContent = `初始 ${num(config.starting_stack)} 筹码`;
    $('level-info').textContent = mode === 'classic' ? '固定盲注' : '每 20 手升盲';
  }
  savePreferences();
}

function renderSeats() {
  if (!state) return;
  const last = {};
  for (const event of state.events) if (event.seat !== null && ACTIONS[event.kind] && !['blind','small_blind','big_blind'].includes(event.kind)) last[event.seat] = event;
  $('seats').innerHTML = state.seats.map(seat => {
    const e = last[seat.seat];
    let action = seat.status === 'busted' ? '已出局' : seat.status === 'folded' ? '已弃牌' : seat.status === 'all_in' ? 'ALL IN' : e ? ACTIONS[e.kind] + (e.amount ? ' ' + num(e.amount) : '') : '';
    if (state.done && seat.delta) action = signed(seat.delta);
    let cards = seat.cards.map(c => card(c, {publicCard:seat.shown.includes(c) && !state.done})).join('');
    if (!seat.human) cards += Array.from({length:Math.max(0,seat.card_count-seat.cards.length)}, () => card(null,{back:true})).join('');
    return `<div class="seat ${seat.human?'human':''} ${seat.status} ${state.actor===seat.seat?'is-turn':''} ${state.done&&seat.delta>0?'winner':''}" data-seat="${seat.seat}"><div class="seat-cards">${cards}</div><div class="seat-info">${seat.seat===state.button?'<span class="seat-position" title="庄家">D</span>':''}<div class="seat-name" title="${esc(seat.name)}">${seat.human?'● ':''}${esc(seat.name)}</div><div class="seat-stack">${num(seat.stack)}</div></div><div class="seat-action ${seat.bet?'chip-bet':''} ${state.done&&seat.delta>0?'delta-positive':state.done&&seat.delta<0?'delta-negative':''}">${esc(action)}${seat.bet&&!state.done?' · '+num(seat.bet):''}</div></div>`;
  }).join('');
  positionSeats();
}

function positionSeats() {
  if (!state) return;
  const width = $('table-stage').clientWidth;
  const nodes = [...document.querySelectorAll('.seat')];
  const botWidth = nodes.find(n => !n.classList.contains('human'))?.offsetWidth || 122;
  const radius = 50 - ((botWidth/2+5)/width*100);
  nodes.forEach((node, i) => {
    const mobilePositions={
      2:[[50,88],[50,10]],
      3:[[50,88],[78,21],[22,21]],
      4:[[50,88],[86,46],[50,10],[14,46]],
      5:[[50,88],[86,65],[73,16],[27,16],[14,65]],
      6:[[50,88],[86,68],[86,29],[50,10],[14,29],[14,68]],
      7:[[50,88],[86,70],[86,40],[67,12],[33,12],[14,40],[14,70]],
      8:[[50,88],[86,72],[86,49],[82,25],[50,9],[18,25],[14,49],[14,72]],
    };
    if(window.innerWidth<=800){
      const [x,y]=mobilePositions[state.seats.length][i];
      node.style.left=`${x}%`;node.style.top=`${y}%`;return;
    }
    const angle = i/state.seats.length * Math.PI*2;
    node.style.left = `${50+radius*Math.sin(angle)}%`;
    node.style.top = `${49+35*Math.cos(angle)}%`;
  });
}

function renderLog() {
  $('history-hand').textContent = `第 ${state.hand} 手 · ${STREETS[state.street]}`;
  const list = state.events.filter(e => ACTIONS[e.kind] || e.kind === 'board');
  $('action-log').innerHTML = list.map(e => {
    if (e.kind === 'board') return `<li class="board-log">公共牌 <span>${e.cards.map(c=>esc((c[0]==='T'?'10':c[0])+SUITS[c[1]])).join(' ')}</span></li>`;
    const name = e.seat===null ? '牌桌' : state.seats[e.seat]?.name || '';
    const detail = e.cards?.length ? e.cards.map(c=>esc((c[0]==='T'?'10':c[0])+SUITS[c[1]])).join(' ') : e.amount ? num(e.amount) : '';
    return `<li class="${e.seat===0?'hero-log':''}"><span class="log-name" title="${esc(name)}">${esc(name)}</span><span class="log-action ${['raise','all_in','win'].includes(e.kind)?'aggressive':e.kind==='fold'?'passive':''}">${ACTIONS[e.kind]} ${detail}</span></li>`;
  }).join('');
  $('action-log').scrollTop = $('action-log').scrollHeight;
  $('standings').innerHTML = [...state.seats].sort((a,b)=>b.stack-a.stack).map(s=>`<div class="standing"><span>${esc(s.name)}</span><b>${num(s.stack)}</b><small>${s.status==='busted'?'已出局':s.human?'你':'机器人'}</small><small>${state.points?state.points[s.seat].toFixed(2)+' 分':'—'}</small></div>`).join('');
  $('recent-hands').innerHTML = state.recent.length ? state.recent.map(h=>`<div class="recent-hand"><span>第 ${h.hand} 手</span><span class="${h.delta>0?'delta-positive':h.delta<0?'delta-negative':''}">${signed(h.delta)}</span></div>`).join('') : '<p class="subtle">完成第一手后显示。</p>';
}

function renderSelection() {
  const cards = state.seats[0].cards;
  $('selection-cards').innerHTML = cards.map((c,i)=>`<div class="select-slot ${choice.discard===i?'discard':choice.reveal===i?'reveal':''}">${card(c)}<span class="role">${choice.discard===i?'弃掉':choice.reveal===i?'公开':'保留 / 隐藏'}</span><div class="select-buttons"><button data-select-discard="${i}" class="${choice.discard===i?'discard-selected':''}" aria-pressed="${choice.discard===i}" aria-label="弃掉第 ${i+1} 张牌">弃掉</button><button data-select-reveal="${i}" class="${choice.reveal===i?'reveal-selected':''}" aria-pressed="${choice.reveal===i}" aria-label="亮出第 ${i+1} 张牌">亮出</button></div></div>`).join('');
  for (const button of document.querySelectorAll('[data-select-discard]')) button.addEventListener('click',()=>{
    choice.discard=Number(button.dataset.selectDiscard);
    if(choice.reveal===choice.discard)choice.reveal=null;
    renderSelection();
  });
  for (const button of document.querySelectorAll('[data-select-reveal]')) button.addEventListener('click',()=>{
    choice.reveal=Number(button.dataset.selectReveal);
    if(choice.discard===choice.reveal)choice.discard=null;
    renderSelection();
  });
  const ready=choice.discard!==null&&choice.reveal!==null;
  $('confirm-selection').disabled=!ready||busy;
  $('selection-summary').textContent=ready ? `弃 ${cards[choice.discard][0]+SUITS[cards[choice.discard][1]]} · 亮 ${cards[choice.reveal][0]+SUITS[cards[choice.reveal][1]]}`:'请选择弃牌和亮牌';
}

function updateRaise(value) {
  const a=state?.legal.find(a=>a.kind==='raise');
  if(!a)return;
  value=Math.min(a.max,Math.max(a.min,Math.trunc(Number(value)||a.min)));
  $('raise-amount').value=value;
  $('raise-slider').value=value;
  const all=value===state.seats[0].stack+state.seats[0].bet;
  $('raise').innerHTML=`${all?'全押':state.seats.some(s=>s.bet>state.seats[0].bet)?'加注至':'下注至'} ${num(value)} <kbd>R</kbd>`;
}

function renderControls() {
  for(const id of ['idle-actions','betting-controls','selection-controls','hand-result'])$(id).hidden=true;
  $('fast-forward').hidden=!state||state.done||state.actor===0;
  $('pause').disabled=!state||state.done;
  $('pause').textContent=paused?'▶':'Ⅱ';
  $('pause').title=paused?'继续机器人行动':'暂停机器人行动';
  $('pause').setAttribute('aria-label',$('pause').title);
  const yourTurn=state&&!state.done&&state.actor===0;
  $('status').classList.toggle('your-turn',yourTurn);
  let message='准备好了就入座。';
  if(!state){$('idle-actions').hidden=false;return;}
  if(state.done){
    $('hand-result').hidden=false;
    const d=state.seats[0].delta;
    $('result-eyebrow').textContent=state.finished?'TABLE COMPLETE':'HAND COMPLETE';
    $('result-title').textContent=state.finished?'本桌结束':d>0?'这一手，拿下。':d<0?'下一手，重新判断。':'平稳收官。';
    $('result-detail').textContent=`你本手 ${signed(d)} 筹码 · 当前 ${num(state.seats[0].stack)}${state.points?' · '+state.points[0].toFixed(2)+' 分':''}`;
    $('result-chips').innerHTML=state.seats.filter(s=>s.delta!==0).map(s=>`<div class="result-chip">${esc(s.name)}<span class="${s.delta>0?'delta-positive':'delta-negative'}">${signed(s.delta)}</span></div>`).join('');
    $('next-hand').textContent=state.finished?'再开一桌 →':'下一手 →';
    message=state.finished?'本桌已结算。你可以调整阵容，再开一桌。':state.seats[0].stack===0?'你已出局；可继续观看，也可以新开一桌。':'本手已结算。看一眼记录，再开始下一手。';
  }else if(yourTurn&&state.phase==='select'){
    $('selection-controls').hidden=false;
    renderSelection();
    message='轮到你选牌。弃一张，公开另一张，保留最后一张。';
  }else if(yourTurn){
    $('betting-controls').hidden=false;
    const legal=Object.fromEntries(state.legal.map(a=>[a.kind,a]));
    $('fold').disabled=!legal.fold||busy;
    const call=legal.check||legal.call;
    $('call').disabled=!call||busy;
    $('call').innerHTML=`${legal.check?'过牌':`跟注 ${num(legal.call?.amount)}`}${legal.call&&legal.call.amount===state.seats[0].stack?' · 全押':''} <kbd>C</kbd>`;
    $('raise').hidden=!legal.raise;
    $('bet-builder').hidden=!legal.raise;
    $('raise').disabled=busy;
    $('call-info').textContent=legal.check?'可以免费过牌':`跟注需要 ${num(legal.call?.amount)} 筹码`;
    if(legal.raise){
      for(const id of ['raise-amount','raise-slider']){$(id).min=legal.raise.min;$(id).max=legal.raise.max;}
      updateRaise(legal.raise.min);
    }
    message=legal.check?'轮到你了。可以过牌，也可以主动下注。':`轮到你了。需要跟注 ${num(legal.call?.amount)} 筹码。`;
  }else{
    $('idle-actions').hidden=false;
    const name=state.seats[state.actor]?.name||'机器人';
    $('idle-actions').innerHTML=`<span class="keycap">${paused?'Ⅱ':'♧'}</span><div><strong>${esc(name)} ${state.phase==='select'?'正在选牌':'等待行动'}</strong><p>${paused?'已暂停。点击右上角播放按钮继续。':state.seats[0].status==='folded'?'你已弃牌，看看这手如何结束。':'留意对手的下注和公开信息。'}</p></div>`;
    message=paused?'机器人行动已暂停。':state.phase==='select'?'正在收集选牌；所有人锁定后才公开亮牌。':`${name} 的回合`;
  }
  $('status').innerHTML=`<span class="status-dot"></span><span>${esc(message)}</span>`;
  $('turn-note').textContent=state.discarded.length?`你的弃牌：${state.discarded.map(c=>c[0]+SUITS[c[1]]).join(' ')} · 对手不可见`:'♠ 只显示你能看到的牌';
}

function render(next) {
  if(next&&(next.hand!==state?.hand||next.game_id!==state?.game_id||next.phase!==state?.phase))choice={discard:null,reveal:null};
  state=next;
  $('empty-message').hidden=!!state;
  $('table-stage').classList.toggle('empty',!state);
  if(!state){$('board').innerHTML=Array.from({length:5},()=>card(null,{placeholder:true})).join('');renderControls();return;}
  const config=modes[state.mode];
  $('game-mode').textContent=config.title;
  $('table-number').textContent=`第 ${state.hand} / ${state.hand_cap} 手`;
  $('blind-info').textContent=`盲注 ${num(state.blinds[0])} / ${num(state.blinds[1])}`;
  $('stack-info').textContent=`${state.seats.filter(s=>s.status!=='busted').length} / ${state.seats.length} 人在桌`;
  $('level-info').textContent=state.mode==='classic'?'固定盲注':`第 ${state.blind_level} 级`;
  $('street-label').textContent=state.phase==='select'?'弃牌与亮牌 · SELECT':STREETS[state.street];
  $('board').innerHTML=state.board.map(c=>card(c)).join('')+Array.from({length:5-state.board.length},()=>card(null,{placeholder:true})).join('');
  $('pot').innerHTML=`<span class="chip-symbol">◉</span><span>${state.done?'本手已结算':'底池'} <strong>${state.done?'✓':num(state.pot)}</strong></span>`;
  $('start-game').innerHTML='新开一桌 <span>↗</span>';
  $('start-game').title='结束当前牌桌，按左侧配置重新开始';
  renderSeats();renderLog();renderControls();
  if(state.warning&&state.warning!==render.previousWarning)toast(state.warning);
  render.previousWarning=state.warning;
}

function schedule() {
  clearTimeout(timer);
  if(!state||state.done||state.actor===0||busy||paused)return;
  timer=setTimeout(()=>command('step'),Number($('speed').value));
}

async function command(name, action) {
  if(busy||!state)return;
  clearTimeout(timer);busy=true;
  for(const id of ['fold','call','raise','next-hand','confirm-selection','fast-forward'])$(id).disabled=true;
  try{
    const data=await api('/api/command',{command:name,action,version:state.version,game_id:state.game_id});
    busy=false;render(data.state);
  }catch(error){
    paused=true;toast(error.message);
    try{const data=await api('/api/session');busy=false;render(data.state);}catch{busy=false;renderControls();}
  }finally{
    busy=false;$('next-hand').disabled=false;$('fast-forward').disabled=false;schedule();
  }
}

async function startGame() {
  if(busy)return;
  savePreferences();clearTimeout(timer);busy=true;
  $('start-game').disabled=true;$('start-game').classList.add('loading');$('start-game').textContent='正在准备牌桌';
  try{
    const data=await api('/api/game',{mode,bots:preferences[mode],name:$('player-name').value.trim()||'你'});
    paused=false;busy=false;render(data.state);
    if(window.innerWidth<800)document.querySelector('.play-area').scrollIntoView({behavior:'smooth',block:'start'});
  }catch(error){toast(error.message);$('start-game').textContent=state?'新开一桌 ↗':'入座，开始游戏 →';}
  finally{busy=false;$('start-game').disabled=false;$('start-game').classList.remove('loading');schedule();}
}

$('start-game').addEventListener('click',startGame);
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>renderSettings(b.dataset.mode)));
$('player-count').addEventListener('change',()=>renderSettings(mode,Number($('player-count').value)));
$('player-name').addEventListener('change',savePreferences);
$('reset-lineup').addEventListener('click',()=>{preferences[mode]=[...modes[mode].defaults];$('lineup').innerHTML='';renderSettings(mode,modes[mode].max_players);});
$('speed').addEventListener('change',()=>{savePreferences();schedule();});
$('pause').addEventListener('click',()=>{paused=!paused;renderControls();schedule();});
$('fast-forward').addEventListener('click',()=>command('fast'));
$('fold').addEventListener('click',()=>command('action',{kind:'fold'}));
$('call').addEventListener('click',()=>command('action',{kind:state.legal.some(a=>a.kind==='check')?'check':'call'}));
$('raise').addEventListener('click',()=>command('action',{kind:'raise',amount:Number($('raise-amount').value)}));
$('raise-slider').addEventListener('input',()=>updateRaise($('raise-slider').value));
$('raise-amount').addEventListener('change',()=>updateRaise($('raise-amount').value));
document.querySelectorAll('[data-bet]').forEach(b=>b.addEventListener('click',()=>{
  const a=state.legal.find(a=>a.kind==='raise'), own=state.seats[0], cost=state.legal.find(a=>a.kind==='call')?.amount||0;
  const values={min:a.min,half:own.bet+cost+Math.floor((state.pot+cost)/2),pot:own.bet+cost+state.pot+cost,all:a.max};
  updateRaise(values[b.dataset.bet]);
}));
$('confirm-selection').addEventListener('click',()=>command('action',{kind:'select',cards:[choice.discard,choice.reveal]}));
$('next-hand').addEventListener('click',()=>state.finished?startGame():command('next'));
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>{
  document.querySelectorAll('[data-tab]').forEach(t=>t.classList.toggle('active',t===b));
  $('actions-tab').hidden=b.dataset.tab!=='actions';$('session-tab').hidden=b.dataset.tab!=='session';
}));
$('rules-open').addEventListener('click',()=>{
  const config=modes[state?.mode||mode];
  $('rules-title').textContent=config.title;
  $('rules-list').innerHTML=config.rules.map(r=>`<li>${esc(r)}</li>`).join('');
  $('rules-dialog').showModal();
});
$('rules-close').addEventListener('click',()=>$('rules-dialog').close());
$('rules-dialog').addEventListener('click',event=>{if(event.target===$('rules-dialog')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
document.addEventListener('keydown',e=>{
  if(e.altKey||e.ctrlKey||e.metaKey||e.repeat||$('rules-dialog').open||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName))return;
  if(!state||state.actor!==0||state.phase==='select'||busy)return;
  const id={f:'fold',c:'call',r:'raise'}[e.key.toLowerCase()];
  if(id&&!$(id).disabled&&!$(id).hidden){e.preventDefault();$(id).click();}
});
new ResizeObserver(positionSeats).observe($('table-stage'));

async function boot() {
  $('start-game').disabled=true;
  try{
    modes=(await api('/api/catalog')).modes;
    $('player-name').value=preferences.name||'你';
    $('speed').value=['700','300','0'].includes(preferences.speed)?preferences.speed:'700';
    // Preserve saved preferences until the settings DOM is first populated.
    mode=null;
    renderSettings(modes[preferences.mode]?preferences.mode:'classic');
    const restored=await api('/api/session');render(restored.state);schedule();
  }catch(error){toast(error.message);}
  finally{$('start-game').disabled=!modes;}
}
boot();
