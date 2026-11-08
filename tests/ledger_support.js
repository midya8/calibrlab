'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
function equal(a,b,where='root'){if(typeof b==='number'){assert.equal(typeof a,'number',where);assert.ok(Number.isFinite(a),where);if(Number.isInteger(b))assert.ok(Number.isInteger(a),where);assert.ok(Math.abs(a-b)<=1e-12+1e-12*Math.abs(b),`${where}: ${a} != ${b}`);}else if(Array.isArray(b)){assert.ok(Array.isArray(a),where);assert.equal(a.length,b.length,where);b.forEach((v,i)=>equal(a[i],v,where+'['+i+']'));}else if(b&&typeof b==='object'){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),where);for(const k of Object.keys(b))equal(a[k],b[k],where+'.'+k);}else assert.equal(a,b,where);}
const root=path.join(__dirname,'../calibrlab/ledger_data'),ws=fs.readdirSync(root).filter(p=>/^score-\d+\.json$/.test(p)).flatMap(p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8')));
module.exports={equal,ws};
