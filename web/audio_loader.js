import { app } from '/scripts/app.js';

export function prepareAudioLoader(nodeData) {
    if (nodeData.name !== 'SolarLLoadAudioUpload') return;
    const required = nodeData.input.required;
    // Comfy.UploadAudio requires audioUI to exist while its upload widget is
    // constructed. Comfy.AudioWidget only adds it to built-in node names.
    const { audioUI, upload, ...inputs } = required;
    nodeData.input.required = {
        ...inputs,
        audioUI: audioUI ?? ['AUDIO_UI', {}],
        ...(upload ? { upload } : {}),
    };
}

app.registerExtension({
    name: 'solarL.H3.AudioLoader',
    beforeRegisterNodeDef(_nodeType, nodeData) {
        prepareAudioLoader(nodeData);
    },
});
