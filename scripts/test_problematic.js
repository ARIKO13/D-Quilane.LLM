// Test problematic prompts
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');
const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);
const EOS = modelData.config.eos_id;
console.log(`Vocab: ${model.config.vocab_size}, EOS: ${EOS}`);

const problematic = [
    'tokyo itu apa',  // USER COMPLAINT 1
    'ibu kota jepang apa',  // USER COMPLAINT 2
    'halo apa kabar',
    'siapa presiden pertama indonesia',
    'lima dikali lima',
    'Apa ibu kota Jepang?',
    'Jakarta itu apa',
    'presiden pertama indonesia siapa',
    '1 tambah 1',
    'berapa hari seminggu',
];

(async () => {
    for (const p of problematic) {
        const t0 = Date.now();
        const out = await model.generate(p, 100, 0.4, 10, 0.9, 1.2, undefined, EOS);
        const elapsed = (Date.now() - t0) / 1000;
        console.log(`[${p}] (${elapsed.toFixed(2)}s)`);
        console.log(`  -> ${out}`);
    }
})();
