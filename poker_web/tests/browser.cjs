/* Browser integration: actual engine responses, both modes, desktop and mobile. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '../..');
const output = path.join(root, '.runtime/web/qa');
fs.mkdirSync(output, {recursive:true});
fs.mkdirSync(path.join(root, '.runtime/tmp'), {recursive:true});
process.env.TMPDIR = path.join(root, '.runtime/tmp');
delete process.env.LD_PRELOAD;
const base = process.env.POKER_WEB_URL || 'http://127.0.0.1:8788';

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || undefined, args:['--no-sandbox']});
  const errors = [], checks = [];
  async function current(page) {
    return (await (await page.request.get(base+'/api/session')).json()).state;
  }
  async function settle(page) {
    await page.waitForFunction(() => !document.querySelector('#next-hand').disabled);
    await page.waitForTimeout(80);
    return current(page);
  }
  async function playHand(page, allowRaise=false) {
    let selections=0,raised=false;
    for(let i=0;i<180;i++) {
      let s=await current(page);
      assert(s);
      for(const seat of s.seats.slice(1))assert.deepEqual(seat.cards,seat.shown,'private bot cards must not be sent');
      if(s.done)return {selections,raised,state:s};
      if(s.actor!==0) {
        await page.locator('#fast-forward').click();
      } else if(s.phase==='select') {
        await page.locator('[data-select-discard="2"]').click();
        await page.locator('[data-select-reveal="0"]').click();
        assert.equal(await page.locator('#confirm-selection').isDisabled(),false);
        await page.screenshot({path:path.join(output,page.viewportSize().width<800?'mobile-selection.png':'desktop-selection.png'),fullPage:true});
        await page.locator('#confirm-selection').click();
        selections++;
      } else if(allowRaise&&!raised&&s.legal.some(a=>a.kind==='raise')) {
        await page.locator('[data-bet="min"]').click();
        await page.locator('#raise').click();raised=true;
      } else {
        await page.locator('#call').click();
      }
      await settle(page);
    }
    throw new Error('Hand did not finish');
  }
  try {
    const context=await browser.newContext({viewport:{width:1500,height:1000}});
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.goto(base);await page.waitForSelector('.policy-select');
    if (base.startsWith('https://')) {
      const cookie=(await context.cookies(base)).find(c=>c.name==='poker_session');
      assert(cookie?.secure && cookie.httpOnly && cookie.sameSite==='Strict');
      assert.equal((await page.request.post(base+'/api/game', {
        headers:{Origin:'https://example.org'},
        data:{mode:'classic',bots:['solver_like'],name:'invalid-origin'},
      })).status(),403);
      checks.push('public HTTPS, Secure HttpOnly SameSite cookie and cross-origin rejection');
    }
    assert.equal(await page.locator('.policy-select').count(),5);
    await page.screenshot({path:path.join(output,'desktop-lobby.png'),fullPage:true});
    await page.locator('#start-game').click();await page.waitForSelector('.seat');
    await page.locator('#pause').click();
    assert.equal(await page.locator('.seat').count(),6);
    assert.equal(await page.locator('.seat.is-turn').count(),1);
    const classic=await playHand(page,true);
    assert(classic.raised);
    assert.equal(classic.state.seats.reduce((n,s)=>n+s.stack,0),60000);
    await page.screenshot({path:path.join(output,'desktop-result.png'),fullPage:true});
    await page.reload();await page.waitForSelector('#hand-result:not([hidden])');
    assert.equal((await current(page)).game_id,classic.state.game_id);
    checks.push('classic six-player play, raise, settlement, hidden cards and refresh resume');
    await page.locator('#next-hand').click();await settle(page);
    assert.equal((await current(page)).hand,2);
    await page.locator('[data-mode="bay_area"]').click();
    assert.equal(await page.locator('.policy-select').count(),7);
    assert.equal(await page.locator('.policy-select option[value="fullhouse_released"]').count(),0);
    await page.locator('#start-game').click();await page.waitForFunction(()=>document.querySelectorAll('.seat').length===8);
    await page.locator('#pause').click();
    let count=0;
    for(let i=0;i<10;i++) {
      const result=await playHand(page);count+=result.selections;
      if(count)break;
      if(result.state.finished){await page.locator('#start-game').click();await page.waitForSelector('.seat');await page.locator('#pause').click();}
      else {await page.locator('#next-hand').click();await settle(page);}
    }
    assert(count>0,'exercise real human discard/reveal phase');
    await page.screenshot({path:path.join(output,'bay-result.png'),fullPage:true});
    checks.push('Bay Area eight-player play, six-way selection controls, sealed cards and settlement');
    await page.locator('#rules-open').click();assert(await page.locator('#rules-dialog').isVisible());await page.locator('#rules-close').click();
    await page.locator('[data-tab="session"]').click();assert.equal(await page.locator('.standing').count(),8);
    const s=await current(page);await page.request.post(base+'/api/leave',{data:{game_id:s.game_id}});
    await context.close();

    const mobile=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
    const mp=await mobile.newPage();mp.on('pageerror',e=>errors.push(e.message));
    await mp.goto(base);await mp.waitForSelector('.policy-select');
    assert.equal(await current(mp),null,'a fresh visitor must not inherit another table');
    await mp.locator('[data-mode="bay_area"]').click();
    await mp.locator('#start-game').click();await mp.waitForSelector('.seat');await mp.locator('#pause').click();
    await mp.locator('.play-area').scrollIntoViewIfNeeded();
    assert(await mp.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1),'mobile horizontal overflow');
    const bounds=await mp.locator('.seat').evaluateAll(nodes=>nodes.map(n=>({left:n.getBoundingClientRect().left,right:n.getBoundingClientRect().right})));
    assert(bounds.every(b=>b.left>=0&&b.right<=390),'mobile seat clipping');
    const overlap=await mp.evaluate(()=>{
      const rects=[...document.querySelectorAll('.seat-info,.seat-cards,.board .card')].map(n=>n.getBoundingClientRect());
      for(let i=0;i<rects.length;i++)for(let j=i+1;j<rects.length;j++){
        const a=rects[i],b=rects[j];
        if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)return true;
      }
      return false;
    });
    assert.equal(overlap,false,'mobile cards and seat labels must not overlap');
    await mp.screenshot({path:path.join(output,'mobile-table.png'),fullPage:true});
    await playHand(mp);
    checks.push('390px mobile layout, eight visible seats and playable controls');
    const ms=await current(mp);await mp.request.post(base+'/api/leave',{data:{game_id:ms.game_id}});
    await mobile.close();
    assert.deepEqual(errors,[]);
    const result={verdict:'pass',url:base,checked_at_utc:new Date().toISOString(),checks,page_errors:errors};
    fs.writeFileSync(path.join(output,'browser-results.json'),JSON.stringify(result,null,2));
    console.log(JSON.stringify(result,null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
