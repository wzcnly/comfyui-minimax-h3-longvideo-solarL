import { app } from '/scripts/app.js';

const groups = [
    ['ref_image', 'IMAGE', 9], ['ref_video', 'IMAGE', 3],
    ['ref_video_audio', 'AUDIO', 3], ['ref_audio', 'AUDIO', 3],
];
const aliases = { source_video: ['ref_video', 0], depth_video: ['ref_video', 1],
    source_audio: ['ref_video_audio', 0] };
const connected = (node, input) => input != null
    && node.isInputConnected(node.inputs.indexOf(input));
function referenceSlot(name) {
    if (aliases[name]) return aliases[name];
    for (const [prefix, , limit] of groups) {
        const match = name.match(new RegExp(`^${prefix}_(\\d+)$`));
        if (match && Number(match[1]) < limit) return [prefix, Number(match[1])];
    }
    return null;
}

export function reconcileReferences(node) {
    if (node._solarLReconciling) return;
    node._solarLReconciling = true;
    try {
        // Migrate linked legacy names in place, preserving their actual link IDs.
        for (const [oldName, [prefix, index]] of Object.entries(aliases)) {
            const old = node.inputs?.find(i => i.name === oldName);
            if (!old || !connected(node, old)) continue;
            const name = `${prefix}_${index}`;
            const current = node.inputs.find(i => i.name === name);
            if (connected(node, current)) {
                // Never discard a second connected cable; backend reports the conflict.
                old.label = `${name}（重复连接，请保留一路）`;
                continue;
            }
            if (current) node.removeInput(node.inputs.indexOf(current));
            old.name = name;
            delete old.label;
        }
        for (let i = (node.inputs?.length ?? 0) - 1; i >= 0; i--) {
            const input = node.inputs[i];
            if (aliases[input.name] && !connected(node, input)) node.removeInput(i);
        }
        for (const [prefix, type, limit] of groups) {
            const linked = (node.inputs ?? []).filter(i => connected(node, i) && referenceSlot(i.name)?.[0] === prefix);
            const highest = Math.max(-1, ...linked.map(i => referenceSlot(i.name)[1]));
            const visible = Math.min(limit, highest + 2);
            for (let i = node.inputs.length - 1; i >= 0; i--) {
                const input = node.inputs[i];
                const slot = referenceSlot(input.name);
                if (!slot || slot[0] !== prefix || connected(node, input)) continue;
                const legacy = linked.some(p => aliases[p.name]?.[0] === prefix && aliases[p.name][1] === slot[1]);
                if (slot[1] >= visible || legacy) node.removeInput(i);
            }
            for (let i = 0; i < visible; i++) {
                const present = node.inputs.some(p => {
                    const slot = referenceSlot(p.name);
                    return slot?.[0] === prefix && slot[1] === i;
                });
                if (!present) node.addInput(`${prefix}_${i}`, type);
            }
        }
        // Keep surviving sockets in place. addInput/removeInput own link indices;
        // raw sorting breaks frontends whose link IDs are derived from slot indices.
        node.setDirtyCanvas?.(true, true);
    } finally { node._solarLReconciling = false; }
}

const secondPassWidgets = ['enable_second_pass','upscaler_model','upscale_device',
    'upscale_precision','upscale_mode','upscale_factor','upscale_width','upscale_height',
    'upscale_megapixels','cfg_2','highres_tiling','tile_count','tile_overlap'];
const selfLiftWidgets = ['enable_second_pass','upscaler_model','upscale_mode','upscale_factor',
    'upscale_width','upscale_height','upscale_megapixels','cfg_2','transition_step',
    'lowres_scale','rho','w_min','w_max','highres_tiling','upscaler_unload'];

export function arrangeSecondPassWidgets(node) {
    if (!node.widgets?.some(w => w.name === 'prompt')) return;
    // Retain the backend's positional order for old workflow files.
    if (!node._solarLCanonicalWidgets) node._solarLCanonicalWidgets = [...node.widgets];
    const names = node.widgets.some(w=>w.name==='transition_step') ? selfLiftWidgets : secondPassWidgets;
    const second = names.map(name => node.widgets.find(w => w.name === name)).filter(Boolean);
    const rest = node.widgets.filter(w => !second.includes(w));
    const afterPrompt = rest.findIndex(w => w.name === 'prompt') + 1;
    rest.splice(afterPrompt,0,...second);
    node.widgets.splice(0,node.widgets.length,...rest);
}

export function updateModeWidgets(node) {
    arrangeSecondPassWidgets(node);
    const mode = node.widgets?.find(w => w.name === 'upscale_mode')?.value ?? 'scale by multiplier';
    const active = { upscale_factor: mode === 'scale by multiplier',
        upscale_width: mode === 'target dimensions', upscale_height: mode === 'target dimensions',
        upscale_megapixels: mode === 'megapixels' };
    for (const widget of node.widgets ?? []) {
        if (!(widget.name in active)) continue;
        // Widgets converted to linked inputs already have their own visibility handling.
        if (node.inputs?.some(i => i.widget?.name === widget.name && connected(node, i))) continue;
        if (!widget._solarLOriginal) widget._solarLOriginal = {
            type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden,
            draw: widget.draw,
            computedHeight: widget.computedHeight,
            optionsHidden: widget.options?.hidden,
            optionsCanvasOnly: widget.options?.canvasOnly,
        };
        if (active[widget.name]) {
            const original = widget._solarLOriginal;
            for (const key of ['type','computeSize','hidden','draw','computedHeight']) {
                if (original[key] === undefined) delete widget[key];
                else widget[key] = original[key];
            }
            widget.hidden = original.hidden ?? false;
            widget.options ||= {};
            widget.options.hidden = original.optionsHidden ?? false;
            widget.options.canvasOnly = original.optionsCanvasOnly ?? false;
        } else {
            widget.type = 'converted-widget';
            widget.hidden = true;
            widget.draw = () => {};
            widget.computeSize = () => [0, -4];
            widget.computedHeight = 0;
            widget.options ||= {};
            widget.options.hidden = true;
            widget.options.canvasOnly = true;
        }
        if (widget._state) {
            widget._state.type = widget.type;
            widget._state.hidden = widget.hidden;
        }
        for (const key of ['element', 'inputEl']) {
            if (widget[key]?.style) widget[key].style.display = active[widget.name] ? '' : 'none';
        }
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: 'solarL.H3LongVideo.ReferenceInputs',
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!['MiniMaxH3LongVideoSampler-solarL','MiniMaxH3SelfLiftLongVideoSampler-solarL'].includes(nodeData.name)) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(...args) {
            if (this._solarLCanonicalWidgets) {
                this.widgets.splice(0,this.widgets.length,...this._solarLCanonicalWidgets);
            }
            const savedValues = args[0]?.widgets_values;
            const isLegacySelfLiftLayout = nodeData.name === 'MiniMaxH3SelfLiftLongVideoSampler-solarL'
                && savedValues?.length === 16
                && ['official_image_guide','motion_context'].includes(savedValues[6])
                && typeof savedValues[7] === 'string'
                && typeof savedValues[8] === 'number';
            if (isLegacySelfLiftLayout) {
                const oldNames = ['seed','control_after_generate','cfg_1','target_frames','segment_frames',
                    'overlap_frames','continuation_mode','prompt','transition_step','lowres_scale','rho',
                    'w_min','w_max','upscaler_model','highres_tiling','upscaler_unload'];
                const previous = new Map(oldNames.map((name,i)=>[name,savedValues[i]]));
                // Preserve the legacy workflow's actual SelfLift settings while mapping
                // its prompt-before-controls layout back to the backend widget order.
                args[0] = {...args[0], widgets_values:this.widgets.map(w=>previous.has(w.name) ? previous.get(w.name) : w.value)};
            }
            const result = configure?.apply(this,args);
            arrangeSecondPassWidgets(this);
            return result;
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(info,...args) {
            const result = serialize?.call(this,info,...args);
            if (Array.isArray(info.widgets_values) && this._solarLCanonicalWidgets) {
                const values = [...info.widgets_values];
                info.widgets_values = this._solarLCanonicalWidgets.map(widget => values[this.widgets.indexOf(widget)]);
            }
            return result;
        };
        const schedule = (node) => {
            if (node._solarLReconciling || node._solarLQueued) return;
            node._solarLQueued = true;
            queueMicrotask(() => {
                node._solarLQueued = false;
                updateModeWidgets(node);
                reconcileReferences(node);
            });
        };
        for (const hook of ['onNodeCreated', 'onConfigure', 'onConnectionsChange', 'onWidgetChanged']) {
            const previous = nodeType.prototype[hook];
            nodeType.prototype[hook] = function(...args) {
                const result = previous?.apply(this, args);
                schedule(this);
                if (hook === 'onNodeCreated' || hook === 'onConfigure') {
                    const widget = this.widgets?.find(w => w.name === 'upscale_mode');
                    if (widget && !widget._solarLModeCallback) {
                        widget._solarLModeCallback = true;
                        const callback = widget.callback;
                        widget.callback = (...values) => {
                            callback?.apply(widget, values);
                            schedule(this);
                        };
                    }
                }
                return result;
            };
        }
        // Other extensions and recent frontends can replace widget callbacks.
        // Check the visible mode during drawing as well; only update on change.
        const draw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(...args) {
            const mode = this.widgets?.find(w => w.name === 'upscale_mode')?.value;
            const signature = `${mode}:${this.widgets?.length}:${this.inputs?.length}`;
            if (this._solarLModeSignature !== signature) {
                this._solarLModeSignature = signature;
                updateModeWidgets(this);
            }
            return draw?.apply(this,args);
        };
    },
});
