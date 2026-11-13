"""Derive a local-edit example from the existing 2080Ti model-loading workflow."""
import copy
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
graph = json.loads((root/'examples/2080ti_内部循环_终版1.json').read_text(encoding='utf-8'))
nodes = {n['id']: n for n in graph['nodes']}
old = nodes[6600]
old_inputs = {p['name']: p.get('link') for p in old['inputs']}
edges = {link[0]: link for link in graph['links']}
old.update(type='MiniMaxH3LocalEditSampler-solarL', title='接续与补段 · 实验初版', size=[400, 360],
           inputs=[{'name': name, 'type': kind, 'link': None} for name, kind in
                   [('edit_plan', 'SOLARL_H3_EDIT_PLAN'), ('model', 'MODEL'), ('positive', 'CONDITIONING'),
                    ('latent', 'LATENT'), ('vae', 'VAE'), ('audio_vae', 'VAE'), ('sampler', 'SAMPLER'),
                    ('sigmas', 'SIGMAS'), ('negative', 'CONDITIONING')]],
           outputs=[{'name': name, 'type': kind, 'links': []} for name, kind in
                    [('patch_images', 'IMAGE'), ('composed_images', 'IMAGE'), ('patch_audio', 'AUDIO'),
                     ('composed_audio', 'AUDIO'), ('fps', 'FLOAT'), ('frame_count', 'INT'),
                     ('window_latent', 'LATENT'), ('report', 'STRING')]],
           widgets_values=[42, 'fixed', 1.0], properties={'Node name for S&R': 'MiniMaxH3LocalEditSampler-solarL'})
old.pop('widgets_values_named', None)
nodes[6630] = dict(id=6630, type='MiniMaxH3LocalEditPlan-solarL', title='选择续写／等长替换／插入／双端补段',
                   pos=[6100, 200], size=[400, 390], flags={}, order=15, mode=0,
                   inputs=[{'name': n, 'type': t, 'link': None} for n, t in
                           [('source_video', 'IMAGE'), ('following_video', 'IMAGE'),
                            ('source_audio', 'AUDIO'), ('following_audio', 'AUDIO')]],
                   outputs=[{'name': 'edit_plan', 'type': 'SOLARL_H3_EDIT_PLAN', 'links': []},
                            {'name': 'report', 'type': 'STRING', 'links': []}],
                   widgets_values=['extend', 24, 72, 48, 22, 'preserve_source', 'auto', 0],
                   properties={'Node name for S&R': 'MiniMaxH3LocalEditPlan-solarL'})
links = [copy.deepcopy(link) for link in graph['links']
         if 6600 not in (link[1], link[3])
         and not (link[3] == 261 and link[4] in (3, 4, 5, 7, 9, 15))
         and link[3] != 6608]
for name, previous in [('model', 'model'), ('positive', 'positive'), ('latent', 'latent'),
                       ('vae', 'vae'), ('audio_vae', 'audio_vae'), ('sampler', 'sampler_1'), ('sigmas', 'sigmas_1')]:
    edge = edges[old_inputs[previous]]
    slot = next(i for i, p in enumerate(old['inputs']) if p['name'] == name)
    links.append([0, edge[1], edge[2], 6600, slot, old['inputs'][slot]['type']])
links += [[0, 6621, 0, 6630, 0, 'IMAGE'], [0, 6621, 2, 6630, 2, 'AUDIO'],
          [0, 6630, 0, 6600, 0, 'SOLARL_H3_EDIT_PLAN'], [0, 6600, 1, 6625, 0, 'IMAGE'],
          [0, 6600, 3, 6625, 1, 'AUDIO'], [0, 6600, 4, 6625, 2, 'FLOAT'],
          [0, 6600, 7, 6608, 0, 'STRING']]
# Keep only ancestors of saved output and report; remove unused model passes and images.
keep = {6303, 6608}
while True:
    more = {link[1] for link in links if link[3] in keep}
    if more <= keep: break
    keep |= more
links = [link for link in links if link[1] in keep and link[3] in keep]
for node in nodes.values():
    for p in node.get('inputs', []): p['link'] = None
    for p in node.get('outputs', []): p['links'] = []
for index, link in enumerate(links, 1):
    link[0] = index
    nodes[link[1]]['outputs'][link[2]]['links'].append(index)
    target = nodes[link[3]]['inputs'][link[4]]
    assert target['link'] is None
    target['link'] = index
nodes[261]['widgets_values'][3] = 124
nodes[261].get('widgets_values_named', {})['length'] = 124
nodes[138]['widgets_values'] = ['保持同一人物、场景和镜头运动，动作自然连贯地延续。']
nodes[138].pop('widgets_values_named', None)
nodes[6303]['widgets_values'][0] = 'H3_local_edit_experimental'
nodes[6303].get('widgets_values_named', {})['filename_prefix'] = 'H3_local_edit_experimental'
nodes[6621]['widgets_values']['video'] = ''
nodes[6621]['widgets_values'].pop('videopreview', None)
nodes[6621].pop('widgets_values_named', None)
graph.update(nodes=[n for n in nodes.values() if n['id'] in keep], links=links, groups=[],
             last_node_id=6630, last_link_id=len(links), revision=0)
graph.pop('id', None)
(root/'examples/2080ti_局部接续与补段_实验初版.json').write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'PASS: generated example, {len(keep)} nodes, {len(links)} consistent links; UI/GPU not executed.')
