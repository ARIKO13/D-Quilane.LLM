// Audit: test banyak unseen prompts untuk liat pola ngelantur
const { DQuilane } = require('/home/z/my-project/download/llm/web/app.js');
const fs = require('fs');
const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);
const EOS = modelData.config.eos_id;

const prompts = [
    // Seen in SFT
    'Halo',
    'Siapa kamu?',
    'Satu tambah satu?',
    'Apa ibu kota Jepang?',
    // Unseen variations
    'hai kamu',
    'nama kamu',
    'jepang di mana',
    'apa yang kamu bisa',
    'coba ngomong',
    'jam berapa sekarang',
    'kamu dari mana',
    'lagi apa',
    'berbicara',
    'test',
    'halo dunia',
];

(async () => {
    for (const p of prompts) {
        const t0 = Date.now();
        const out = await model.generate(p, 50, 0.4, 10, 0.9, 1.2, undefined, EOS);
        const elapsed = (Date.now() - t0) / 1000;
        console.log(`[${p}] (${elapsed.toFixed(2)}s)`);
        console.log(`  -> ${out.substring(0, 100)}`);
    }
})();
