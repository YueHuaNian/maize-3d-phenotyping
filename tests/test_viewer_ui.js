// 用合成数据和模拟 DOM 检验查看器逻辑，不启动浏览器或外部服务。
// 运行命令：node tests/test_viewer_ui.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../src/maize_leaf/viewer.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const leaves = ['convergent', 'expanded', 'drooping'].map((key, i) => ({
  leaf_id: i + 1, support_review_required:i===2,
  classification: {cluster_kmeans: key, cluster_kmeans_lopo_oof: key,
    relative_height: .9 - .3 * i, stem_leaf_angle_deg: 20 + 25 * i,
    projection_ratio: .5, rel_projected_area: .3 + .3*i, pitch_entropy_b12: .7, coverage_observed: .8,
    coverage_extrapolated: .1, ambiguous_r080: i === 1, lopo_switched: i === 2,
    changed_by_height: i === 2, cluster_without_height: 'expanded', distance_ratio_r: .7},
}));
const payload = {plant: 'p1', info: {leaves}, traces: [
  ...leaves.map(l => ({type: 'mesh3d', group: 'surface', leaf_id: l.leaf_id,
    color: '#123456', region_facecolors: ['#abcdef','#fedcba'], face_regions:[0,1], i:[0,0],j:[0,0],k:[0,0], x: [0], y: [0], z: [0]})),
  {type: 'scatter3d', group: 'labels', leaf_ids: [1, 2, 3], x: [0, 1, 2], y: [0, 0, 0], z: [1, .5, 0],
    text: ['L1', 'L2', 'L3'], hovertext: ['a', 'b', 'c'],
    marker: {color: ['blue', 'green', 'orange']}, textfont: {color: ['blue', 'green', 'orange']}},
]};

// 分别验证在线接口与离线内嵌载荷的显示、筛选和快速选叶。
async function check(offline) {
  // 构造最小页面元素、绘制计数和场景缓存，记录不必要的重绘。
  const elements = new Map();
  for (const [, id] of html.matchAll(/id="([^"]+)"/g)) {
    elements.set(id, {value: 'all', checked: true, hidden: true, textContent: '', innerHTML: '',
      handlers: {}, on(event, fn) { this.handlers[event] = fn; },
      querySelector() { return {}; }, querySelectorAll() { return []; }});
  }
  elements.get('chkPoints').checked = false;
  elements.get('isolateLeaf').checked = false;
  elements.get('chkInferred').checked = false;
  const plots = new Map();
  const reactCalls = {plot:0,featurePlot:0};
  let restyleCalls=0;
  let fetchCount = 0;
  const catalog = {p1: [1, 2, 3], p2: [1, 2, 3]};
  const bootstrap = {offline, catalog, initial_plant: 'p1', initial_leaf: null,
    initial_color_by: 'kmeans', has_classification: true,
    classification_metadata: {primary_model: {height_weight: .5}}, payload};
  const context = vm.createContext({
    console, URLSearchParams, Number, Map, Set, requestAnimationFrame:fn=>setTimeout(fn,0),
    document: {getElementById: id => elements.get(id)},
    location: {search: '', href: ''}, history: {replaceState() {}},
    Plotly: {
      react: async (id,data)=>{reactCalls[id]++;plots.set(id,data);},
      restyle: async (id,updates,indices)=>{
        restyleCalls++;
        const data=plots.get(id), selected=indices||data.map((_,i)=>i);
        for(const [key,values] of Object.entries(updates)){
          selected.forEach((idx,j)=>{
            const parts=key.split('.');let target=data[idx];
            for(const part of parts.slice(0,-1))target=target[part]??={};
            target[parts.at(-1)]=Array.isArray(values)?values[j]:values;
          });
        }
        // 模拟渲染过程中触发点击回调，检查重入保护。
        if(id==='featurePlot'&&elements.get(id).handlers.plotly_click){
          elements.get(id).handlers.plotly_click({points:[{customdata:1}]});
        }
        if(id==='plot'&&elements.get(id).handlers.plotly_click){
          elements.get(id).handlers.plotly_click({points:[{data:{meta:{leaf_id:1}}}]});
        }
      },relayout() {},downloadImage() {}
    },
    fetch: async url => {
      fetchCount++;
      return {ok: true, json: async () => url === '/api/catalog' ? bootstrap : payload};
    },
  });
  vm.runInContext(script.replace('__BOOTSTRAP_JSON__', offline ? JSON.stringify(bootstrap) : 'null')
    .replace(/boot\(\);\s*$/, 'globalThis.ready = boot();'), context);
  await context.ready;
  const surfaces = () => plots.get('plot').filter(t => t.type === 'mesh3d' && t.visible!==false);
  assert.equal(surfaces().length, 3);
  assert.match(elements.get('modelNote').textContent, /0\.50/);
  assert.match(elements.get('leafList').innerHTML, /相对高度/);

  await vm.runInContext('state.leaf=2; render()', context);
  assert.equal(surfaces().length, 3); // 选叶保留整株背景和网格缓冲。
  assert.equal(reactCalls.plot,1);
  assert.equal(reactCalls.featurePlot,1);
  assert.equal(vm.runInContext('state.leaf',context),2);
  elements.get('isolateLeaf').checked=true;
  await vm.runInContext('render()',context);
  assert.equal(surfaces().length,1);
  assert.equal(surfaces()[0].meta.leaf_id,2);
  elements.get('isolateLeaf').checked=false;
  assert.match(elements.get('selectionInfo').innerHTML, /相对着生高度/);
  const marker=plots.get('plot').find(t=>t.name==='当前叶片');
  assert.deepEqual(Array.from(marker.x),[1]);

  elements.get('classFilter').value = 'drooping';
  await vm.runInContext('state.leaf=null; render()', context);
  assert.equal(surfaces()[0].meta.leaf_id, 3);
  assert.equal(surfaces().length, 1);

  elements.get('resetFilters').onclick();
  await vm.runInContext('render()', context);
  assert.equal(surfaces().length, 3);
  assert.equal(elements.get('classFilter').value, 'all');
  assert.equal(vm.runInContext('state.leaf', context), null);
  await vm.runInContext("state.colorBy='region'; render()", context);
  assert.equal(surfaces().length, 3);
  assert.deepEqual(Array.from(surfaces()[0].facecolor), ['#abcdef']);
  assert.ok(!('group' in surfaces()[0]));
  assert.ok(!('region_facecolors' in surfaces()[0]));
  elements.get('chkInferred').checked=true;
  await vm.runInContext('render()',context);
  assert.equal(surfaces().length,6);
  elements.get('chkInferred').checked=false;
  await vm.runInContext('render()',context);
  assert.equal(surfaces().length,3);
  assert.equal(fetchCount, offline ? 0 : 2);
  // 一百次快速选叶应合并处理，并保持已有网格引用。
  const before=reactCalls.plot;
  const geometry=surfaces()[0].x;
  await vm.runInContext('Promise.all(Array.from({length:100},(_,i)=>selectLeaf(i%3+1)))',context);
  assert.equal(reactCalls.plot,before);
  assert.equal(surfaces()[0].x,geometry);
  assert.ok(restyleCalls<100,'Rapid clicks must coalesce rather than queue 100 mesh draws');

  if (offline) {
    await vm.runInContext('selectPlant("p2",3)', context);
    assert.match(context.location.href, /^plant_p2\.html\?.*leaf=3/);
  }
}

// 两种运行方式都通过才判定前端逻辑检查成功。
(async () => {
  await check(false);
  await check(true);
  console.log('PASS: stable geometry, coalesced rapid selection, reentrant click guard, live/offline parity, filters and offline navigation.');
})().catch(error => { console.error(error); process.exitCode = 1; });
