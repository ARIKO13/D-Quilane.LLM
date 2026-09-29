// Final test
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');
const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);
const EOS = modelData.config.eos_id;
console.log(`Vocab: ${model.config.vocab_size}, EOS: ${EOS}, d_model: ${model.config.d_model}, n_layers: ${model.config.n_layers}`);

const prompts = [
    // Previously problematic
    'tokyo itu apa',
    'ibu kota jepang apa',
    // Unseen
    'hai kamu',
    'nama kamu',
    'jepang di mana',
    'apa yang kamu bisa',
    'coba ngomong',
    'jam berapa sekarang',
    'kamu dari mana',
    'lagi apa',
    'test',
    'halo dunia',
    // Standard
    'Halo',
    'Satu tambah satu?',
    'Siapa presiden pertama Indonesia?',
    'Apa ibu kota Jepang?',
];

(async () => {
    console.log('\n=== v0.9 TEST ===');
    for (const p of prompts) {
        const t0 = Date.now();
        const out = await model.generate(p, 100, 0.4, 10, 0.9, 1.2, undefined, EOS);
        const elapsed = (Date.now() - t0) / 1000;
        console.log(`[${p}] (${elapsed.toFixed(2)}s)`);
        console.log(`  -> ${out.substring(0, 100)}`);
    }
})();
