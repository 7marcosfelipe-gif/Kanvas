#!/usr/bin/env node
// Quick diagnostic: run DagLayout on stress-test and check for group overlaps
const fs = require("fs");

// Stub obsidian require so the plugin module loads
const Module = require("module");
const _orig = Module._resolveFilename;
Module._resolveFilename = (req, ...a) => req === "obsidian" ? req : _orig(req, ...a);
require.extensions[".js"] = require.extensions[".js"];
const fakeObsidian = { Plugin: class {}, Notice: class {}, TFile: class {}, setIcon: () => {}, addIcon: () => {} };
require.cache["obsidian"] = { id: "obsidian", filename: "obsidian", loaded: true, exports: fakeObsidian };

// Load DagLayout from main.js by eval-ing into global scope
const src = fs.readFileSync("canvas-watcher-plugin/main.js", "utf8");
const match = src.match(/\/\/ --- DAG Layout Engine ---([\s\S]*?)\/\/ --- Core processing/);
const code = match[1].replace("const DagLayout =", "global.DagLayout =");
eval(code);

const canvas = JSON.parse(fs.readFileSync("layout-fixtures/stress-test.canvas", "utf8"));
const result = DagLayout.organize(canvas);

const groups = result.nodes.filter(n => n.type === "group");
const tasks  = result.nodes.filter(n => n.type === "text" && !/^##\s*Legend/.test(n.text || "") && !["canvas-errors","canvas-warnings"].includes(n.id));

// Check group overlaps
let overlaps = 0;
for (let i = 0; i < groups.length; i++) {
  for (let j = i + 1; j < groups.length; j++) {
    const a = groups[i], b = groups[j];
    const ox = a.x < b.x + b.width  && b.x < a.x + a.width;
    const oy = a.y < b.y + b.height && b.y < a.y + a.height;
    if (ox && oy) {
      console.log(`OVERLAP: "${a.label}" vs "${b.label}"`);
      overlaps++;
    }
  }
}
if (!overlaps) console.log("No group overlaps ✓");

console.log("\nGroup bounds:");
for (const g of groups.sort((a,b) => a.x - b.x))
  console.log(`  "${g.label}": x=${g.x} y=${g.y} w=${g.width} h=${g.height} → right=${g.x+g.width}`);

// Check depth alignment: cards at same y-band should be at same y
const yBands = {};
for (const t of tasks) {
  const key = Math.round(t.y / 10) * 10;
  if (!yBands[key]) yBands[key] = [];
  yBands[key].push({ id: t.id, y: t.y, text: (t.text||"").split("\n")[0].slice(0,30) });
}
const yVals = Object.keys(yBands).map(Number).sort((a,b)=>a-b);
console.log(`\nDepth rows (${yVals.length} distinct y-bands):`);
for (const y of yVals) {
  const band = yBands[y];
  const ys   = [...new Set(band.map(t => t.y))];
  const flag = ys.length > 1 ? " ← misaligned" : "";
  console.log(`  y≈${y}: ${band.length} cards, actual y: ${ys.join(",")}${flag}`);
}
