import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';

const TARGETS = new Set(['MiniMaxH3LongVideoSampler-solarL', 'MiniMaxH3SelfLiftLongVideoSampler-solarL']);

export function addCacheButtons(node) {
    if (node.__solarLCacheButtons) return;
    const field = node.widgets?.find(w => w.name === 'cache_directory');
    if (!field) return;
    node.__solarLCacheButtons = true;
    const update = value => {
        field.value = value;
        field.callback?.(value);
        node.setDirtyCanvas?.(true, true);
    };
    let pending = false;
    const select = node.addWidget('button', '选择缓存文件夹…', null, async () => {
        if (pending) return;
        pending = true;
        select.name = '请在系统窗口中选择…';
        try {
            const response = await api.fetchApi('/solarL/cache-directory/choose', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ initial: field.value || '' }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || '目录选择失败');
            if (!data.cancelled && data.path) update(data.path);
        } catch (error) {
            app.ui.dialog.show(String(error.message || error));
        } finally {
            pending = false;
            select.name = '选择缓存文件夹…';
            node.setDirtyCanvas?.(true, true);
        }
    }, { serialize: false });
    const clear = node.addWidget('button', '清空关闭缓存', null, () => {
        if (pending) return;
        update('');
    }, { serialize: false });
    select.serialize = clear.serialize = false;
}

app.registerExtension({
    name: 'solarL.CacheDirectoryPicker',
    async beforeRegisterNodeDef(NodeType, data) {
        if (!TARGETS.has(data.name)) return;
        const created = NodeType.prototype.onNodeCreated;
        NodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            addCacheButtons(this);
            return result;
        };
    },
});
