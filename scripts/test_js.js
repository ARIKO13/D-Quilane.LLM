// Test JS inference engine dengan model JSON hasil export PyTorch
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');

const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);

console.log('=== Model loaded ===');
console.log('Config:', model.config);

(async () => {
    const prompts = ['Indonesia', 'English is', 'La famille'];
    for (const p of prompts) {
        console.time('gen_' + p.replace(/\s/g, '_'));
        const out = await model.generate(p, 40, 0.7, 10);
        console.timeEnd('gen_' + p.replace(/\s/g, '_'));
        console.log(`[${p}] -> ${out}`);
    }
})();
