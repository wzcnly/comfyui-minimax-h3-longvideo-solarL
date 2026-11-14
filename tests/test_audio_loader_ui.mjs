import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = (await readFile(new URL('../web/audio_loader.js', import.meta.url), 'utf8'))
    .replace("import { app } from '/scripts/app.js';", 'const app = { registerExtension() {} };');
const { prepareAudioLoader } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

// Simulate Comfy.UploadAudio running either before or after this extension.
for (const uploadFirst of [true, false]) {
    const data = { name: 'SolarLLoadAudioUpload', input: { required: {
        audio: [['none', 'test.wav'], { audio_upload: true }],
        start_seconds: ['FLOAT', { default: 0 }],
        duration_seconds: ['FLOAT', { default: 0 }],
    } } };
    if (uploadFirst) data.input.required.upload = ['AUDIOUPLOAD', {}];
    prepareAudioLoader(data);
    if (!uploadFirst) data.input.required.upload = ['AUDIOUPLOAD', {}];
    prepareAudioLoader(data); // Repeated registration must remain safe.
    const widgets = [];
    for (const [name, [type]] of Object.entries(data.input.required)) {
        if (type === 'AUDIOUPLOAD') {
            const preview = widgets.find(w => w.name === 'audioUI');
            assert.ok(preview, 'audioUI must exist when the upload widget initializes');
            preview.element.src = '/view?filename=test.wav';
        } else {
            widgets.push({ name, element: type === 'AUDIO_UI' ? { src: '' } : undefined });
        }
    }
    assert.deepEqual(Object.keys(data.input.required), ['audio', 'start_seconds', 'duration_seconds', 'audioUI', 'upload']);
    assert.equal(widgets.find(w => w.name === 'audioUI').element.src, '/view?filename=test.wav');
}
const unrelated = { name: 'LoadAudio', input: { required: {} } };
prepareAudioLoader(unrelated);
assert.deepEqual(unrelated.input.required, {});
console.log('PASS: custom audio loader creates its player before upload initialization in either extension order');
