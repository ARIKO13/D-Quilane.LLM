// Speed test: before/after KV-cache
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');

const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);

const prompts = ['Indonesia', 'Halo', 'English is', '日本語', '한국어'];

(async () => {
    console.log('=== Speed test (KV-cache version) ===');
    for (const p of prompts) {
        const t0 = Date.now();
        const out = await model.generate(p, 50, 0.7, 10, 0.9, 1.1);
        const elapsed = (Date.now() - t0) / 1000;
        const tps = 50 / elapsed;
        console.log(`[${p}] ${elapsed.toFixed(2)}s (${tps.toFixed(1)} tok/s)`);
        console.log(`  -> ${out.substring(0, 80)}...`);
    }
})();
